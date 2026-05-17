#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import multiprocessing
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GA_Update
from osgeo_utils.gdal_calc import Calc as gdal_calc

# add project dir to pythonpath (parent of this file is the project root)
project_dir = os.path.dirname(os.path.realpath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from core import iMad, radcal, register

header = '''
==============================================================

ArrNorm - Automatic Relative Radiometric Normalization

Some code base on: Dr. Mort Canty
                   https://github.com/mortcanty/CRCDocker

Copyright (c) SMByC-IDEAM
Author: Xavier C. Llano <xavier.corredor.llano@gmail.com>
Sistema de Monitoreo de Bosques y Carbono - SMByC
IDEAM, Colombia

==============================================================
'''


class Normalization:
    def __init__(self, count, img_ref, img_target, max_iters, threshold,
                 reg, onlyreg, noneg, mask, warpband, chunksize):
        self.count = count
        self.img_ref = img_ref
        self.img_target = img_target
        self.ref_text = f"Image ({self.count + 1})"
        self.max_iters = max_iters
        self.threshold = threshold
        self.reg = reg
        self.onlyreg = onlyreg
        self.noneg = noneg
        self.mask = mask
        self.warpband = warpband
        self.chunksize = chunksize

        self.img_ref_clip = None
        self.img_imad = None
        self.img_norm = None

        # Output dtype is the higher-precision of ref and target
        ref_ds = gdal.Open(self.img_ref, GA_ReadOnly)
        ref_dtype_code = ref_ds.GetRasterBand(1).DataType
        ref_ds = None

        target_ds = gdal.Open(self.img_target, GA_ReadOnly)
        target_dtype_code = target_ds.GetRasterBand(1).DataType
        target_ds = None

        self.out_dtype = max(ref_dtype_code, target_dtype_code)

    def run(self):
        print(f"\nPROCESSING IMAGE: {os.path.basename(self.img_target)} ({self.count + 1})")

        self.clipper()

        if self.reg:
            self.register()
            if self.onlyreg:
                print(f'\nDONE: {self.ref_text} PROCESSED\n'
                      f'      register successfully for:  {os.path.basename(self.img_target)}\n')
                return

        self.imad()
        self.radcal()
        if self.noneg:
            self.no_negative_value(self.img_norm)

        if self.mask:
            self.make_mask()
            self.apply_mask()

        self.clean()

        print(f'\nDONE: {self.ref_text} PROCESSED\n'
              f'      ArrNorm successfully for:  {os.path.basename(self.img_target)}\n'
              f'      image normalized saved in: {os.path.basename(self.img_norm)}\n')

    def clipper(self):
        """Reproject and clip the reference image onto the target's exact pixel grid.

        The output reference clip will have:
          - the same CRS as the target,
          - the same upper-left origin (to full floating-point precision),
          - the same pixel width and height (cols x rows),
        guaranteeing pixel-for-pixel spatial coincidence required for IR-MAD.

        Uses gdal.Warp (not gdal.Translate) so that CRS reprojection is
        applied in the same step as the extent clip, and the output
        width/height is forced to match the target exactly.
        """
        print(f"\n======================================\n"
              f"Clipping/aligning the ref image with target: "
              f"{self.ref_text} {os.path.basename(self.img_ref)}")

        target_ds = gdal.Open(self.img_target, GA_ReadOnly)
        target_proj = target_ds.GetProjection()
        target_gt = target_ds.GetGeoTransform()   # (originX, pixW, rot, originY, rot, pixH)
        target_cols = target_ds.RasterXSize
        target_rows = target_ds.RasterYSize
        target_ds = None

        ref_ds = gdal.Open(self.img_ref, GA_ReadOnly)
        ref_proj = ref_ds.GetProjection()
        ref_gt = ref_ds.GetGeoTransform()
        ref_cols = ref_ds.RasterXSize
        ref_rows = ref_ds.RasterYSize
        ref_ds = None

        # Fast-path: if reference is already on the identical pixel grid
        # (same CRS, same geotransform, same dimensions) skip clipping.
        if (ref_proj == target_proj
                and ref_gt == target_gt
                and ref_cols == target_cols
                and ref_rows == target_rows):
            print("Reference already perfectly aligned with target "
                  "(same CRS, geotransform and dimensions). No clipping needed.")
            self.img_ref_clip = self.img_ref
            return

        print(f"Reprojecting/aligning reference to target pixel grid "
              f"(resolution: {target_gt[1]:.6g} x {abs(target_gt[5]):.6g}, "
              f"dimensions: {target_cols} x {target_rows})")

        filename, ext = os.path.splitext(os.path.basename(self.img_ref))
        self.img_ref_clip = os.path.join(
            os.path.dirname(os.path.abspath(self.img_target)),
            filename + "_" + os.path.splitext(os.path.basename(self.img_target))[0] + "_clip" + ext
        )

        try:
            # Derive exact output bounds from target geotransform.
            # Using raw GT values avoids float-rounding from intermediate
            # string/float conversions (gt[5] is negative for north-up rasters).
            xmin = target_gt[0]
            ymax = target_gt[3]
            xmax = xmin + target_gt[1] * target_cols
            ymin = ymax + target_gt[5] * target_rows

            # gdal.Warp with:
            #   dstSRS       → reproject to target CRS (no-op if already same)
            #   outputBounds → clip/pad to exact target extent
            #   width/height → force exactly target_cols x target_rows pixels
            result = gdal.Warp(
                self.img_ref_clip, self.img_ref,
                format='GTiff',
                dstSRS=target_proj,
                outputBounds=(xmin, ymin, xmax, ymax),  # (minX, minY, maxX, maxY)
                width=target_cols,
                height=target_rows,
                resampleAlg=gdal.GRA_Bilinear,
            )
            if result is None:
                raise RuntimeError("gdal.Warp returned None — check GDAL error log.")
            result = None
            print(f'Reference aligned successfully: {os.path.basename(self.img_ref_clip)}')
        except Exception as e:
            self.clean()
            print(f'\nError clipping/reprojecting reference image: {e}')
            sys.exit(1)

    def imad(self):
        print(f"\n======================================\n"
              f"iMad process for: {self.ref_text} {os.path.basename(self.img_target)}")
        img_target = self.img_target_reg if self.reg else self.img_target
        self.img_imad = iMad.main(self.img_ref_clip, img_target,
                                  ref_text=self.ref_text, max_iters=self.max_iters)

    def register(self):
        print(f"\n======================================\n"
              f"Registration image-image in frequency domain: "
              f"{self.ref_text} {os.path.basename(self.img_target)}")
        self.img_target_reg = register.main(self.img_ref_clip, self.img_target,
                                            self.warpband, self.chunksize)

    def radcal(self):
        print(f"\n======================================\n"
              f"Radcal process for: {self.ref_text} {os.path.basename(self.img_target)}"
              f" with iMad image: {os.path.basename(self.img_imad)}")
        self.img_norm = radcal.main(self.img_imad, ncpThresh=self.threshold,
                                    out_dtype=self.out_dtype)

    def no_negative_value(self, image):
        print(f'\n======================================\n'
              f'Converting negative values for: {self.ref_text} {os.path.basename(image)}')
        tmp_image = image.replace('.tif', '_tmp.tif')
        try:
            gdal_calc(A=image, outfile=tmp_image, calc="A*(A>=0)", NoDataValue=0,
                      allBands="A", overwrite=True, quiet=True,
                      creation_options=["BIGTIFF=YES"])
            os.remove(image)
            os.rename(tmp_image, image)
            print(f'Negative values converted successfully: {os.path.basename(image)}')
        except Exception as e:
            self.clean()
            print(f'\nError converting values: {e}')
            sys.exit(1)

    def make_mask(self):
        img_to_process = self.img_target_reg if self.reg else self.img_target
        print(f'\n======================================\n'
              f'Making mask for: {self.ref_text} {os.path.basename(img_to_process)}')
        filename, ext = os.path.splitext(os.path.basename(img_to_process))
        self.mask_file = os.path.join(
            os.path.dirname(os.path.abspath(img_to_process)),
            filename + "_mask" + ext)
        try:
            gdal_calc(A=img_to_process, outfile=self.mask_file, type=gdal.GDT_Byte,
                      calc="1*(A>0)", overwrite=True, quiet=True,
                      creation_options=["COMPRESS=PACKBITS", "NBITS=1"])
            colors = gdal.ColorTable()
            colors.SetColorEntry(0, (0, 0, 0, 255))
            colors.SetColorEntry(1, (0, 255, 0, 255))
            mask_ds = gdal.Open(self.mask_file, GA_Update)
            if mask_ds is not None:
                mask_ds.GetRasterBand(1).SetRasterColorTable(colors)
                mask_ds = None
            print(f'Mask created successfully: {os.path.basename(self.mask_file)}')
        except Exception as e:
            print(f'\nError creating mask: {e}')
            sys.exit(1)

    def apply_mask(self):
        print(f'\n======================================\n'
              f'Applying mask for: {self.ref_text} {os.path.basename(self.img_norm)}')
        tmp_img_norm = self.img_norm.replace('.tif', '_tmp.tif')
        try:
            gdal_calc(A=self.img_norm, B=self.mask_file, outfile=tmp_img_norm,
                      calc="A*(B==1)", NoDataValue=0, allBands="A",
                      overwrite=True, quiet=True, creation_options=["BIGTIFF=YES"])
            os.remove(self.img_norm)
            os.rename(tmp_img_norm, self.img_norm)
            print(f'Mask applied successfully: {os.path.basename(self.mask_file)}')
        except Exception as e:
            self.clean()
            print(f'\nError applying mask: {e}')
            sys.exit(1)

    def clean(self):
        if self.img_imad and os.path.exists(self.img_imad):
            os.remove(self.img_imad)
        # Only delete the clipped reference if it was a temporary file,
        # not when the reference was already aligned and used directly.
        if (self.img_ref_clip
                and self.img_ref_clip != self.img_ref
                and os.path.exists(self.img_ref_clip)):
            os.remove(self.img_ref_clip)


def process(norm_class, args):
    norm_instance = norm_class(*args)
    norm_instance.run()


def meta_process(args):
    process(*args)


def main():
    multiprocessing.freeze_support()

    arguments = argparse.ArgumentParser(
        prog="arrnorm",
        description="Automatic relative radiometric normalization",
        epilog="Xavier Corredor Llano <xcorredorl@ideam.gov.co>\n"
               "Sistema de Monitoreo de Bosques y Carbono - SMBYC\n"
               "IDEAM, Colombia",
        formatter_class=argparse.RawTextHelpFormatter)

    arguments.add_argument('-ref', type=str,
                           help='reference image for iMad normalize', required=True)
    arguments.add_argument('-i', type=int, default=30,
                           help='number of iterations', required=False)
    arguments.add_argument('-t', type=float, default=0.95,
                           help='no-change probability threshold', required=False)
    arguments.add_argument('-p', type=int, default=multiprocessing.cpu_count(),
                           help='number of process/threads', required=False)
    arguments.add_argument('-m', action='store_true', default=False,
                           help='create and apply nodata mask', required=False)
    arguments.add_argument('-reg', action='store_true', default=False,
                           help='registration image-image in frequency domain', required=False)
    arguments.add_argument('-noneg', action='store_true', default=False,
                           help='convert all negative values to nodata (0) for output', required=False)
    arguments.add_argument('-warpband', type=int, default=2,
                           help='number of target band for make registration, requires "-reg"',
                           required=False)
    arguments.add_argument('-chunksize', type=int, default=None,
                           help='chunk size for make registration by chunks, requires "-reg"',
                           required=False)
    arguments.add_argument('-onlyreg', action='store_true', default=False,
                           help='only makes registration, no iMad normalize, requires "-reg"',
                           required=False)
    arguments.add_argument('images', type=str, nargs='+',
                           help='images to apply the iMad normalization')

    arg = arguments.parse_args()

    print(header)
    number_of_processes = arg.p
    print(f'Reference image: {os.path.basename(arg.ref)}')
    print(f'Creating {number_of_processes} multiprocesses')

    with multiprocessing.Pool(number_of_processes) as pool:
        TASKS = [(Normalization, (img_count, arg.ref, img_target, arg.i, arg.t,
                                  arg.reg, arg.onlyreg, arg.noneg, arg.m,
                                  arg.warpband, arg.chunksize))
                 for img_count, img_target in enumerate(arg.images)]
        for _ in pool.imap(meta_process, TASKS):
            pass

    print(f'\nFINISH: successfully process for {len(arg.images)} images\n')


if __name__ == '__main__':
    main()
