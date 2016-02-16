#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from subprocess import call

from arrNorm import iMad, radcal

# ==============================================================================
# PARSER AND CHECK ARGUMENTS

# Create parser arguments
arguments = argparse.ArgumentParser(
    prog="arrnorm",
    description="Automatic relative radiometric normalization",
    epilog="SMBYC-IDEAM - Xavier Corredor Llano <xcorredorl@ideam.gov.co>",
    formatter_class=argparse.RawTextHelpFormatter)

arguments.add_argument('-ref', type=str, required=True,
                       help='reference image for iMad normalize')

arguments.add_argument('-i', type=int, default=25,
                       help='number of iterations', required=False)

arguments.add_argument('-t', type=float, default=0.95,
                       help='no-change probability threshold', required=False)


def check_mask_option(option):
    if option in ['yes', 'Yes', 'YES']:
        return True
    if option in ['no', 'No', 'NO']:
        return False
    raise argparse.ArgumentTypeError('mask option invalid, should be: "yes" or "no"')

arguments.add_argument('-m', type=check_mask_option, default='yes',
                       help='create and apply nodata mask', required=False)

arguments.add_argument('images', type=str, nargs='+',
                       help='images to apply the iMad normalization')

arg = arguments.parse_args()

header = '''
==============================================================

arrNorm - Automatic Relative Radiometric Normalization

Code base on: Dr. Mort Canty
              https://github.com/mortcanty/CRCDocker

Copyright (c) 2016 SMBYC-IDEAM
Authors: Xavier Corredor Llano <xcorredorl@ideam.gov.co>
Instituto de Hidrología, Meteorología y Estudios Ambientales
Sistema de Monitoreo de Bosques y Carbono - SMBYC

==============================================================
'''
print(header)

# ==============================================================================
# PROCESS IMAGES

for img_count, img_target in enumerate(arg.images):

    print("\nPROCESSING IMAGE: {target} ({count})".format(
        target=os.path.basename(img_target),
        count=str(img_count+1)+'/'+str(len(arg.images))
    ))

    # ======================================
    # iMad process

    print("\n======================================\n"
          "iMad process for: ", os.path.basename(img_target))
    img_imad = iMad.main(arg.ref, img_target, niter=arg.i)

    # ======================================
    # Radcal process

    print("\n======================================\n"
          "Radcal process for: ", os.path.basename(img_target),
          " with iMad image: ", os.path.basename(img_imad))
    img_norm = radcal.main(img_imad, ncpThresh=arg.t)

    # ======================================
    # Make mask

    if arg.m:
        print('\n======================================\n'
              'Making mask:')
        filename, ext = os.path.splitext(os.path.basename(img_target))
        mask_file = os.path.join(os.path.dirname(os.path.abspath(img_target)),
                                 filename+"_mask"+ext)
        return_code = call('gdal_calc.py -A '+img_target+' --outfile='+mask_file+' --calc="1*(A>0)" --NoDataValue=0', shell=True)
        if return_code == 0:  # successfully
            print('Mask created successfully: ' + os.path.basename(mask_file))
        else:
            print('\nError creating mask: ' + str(return_code))
            sys.exit(1)

    # ======================================
    # Apply mask to image normalized

    if arg.m:
        print('\n======================================\n'
              'Applying mask:')
        return_code = call('gdal_calc.py -A '+img_norm+' -B '+mask_file+' --outfile='+img_norm+' --calc="A*(B==1)" --NoDataValue=0  --allBands=A  --overwrite', shell=True)
        if return_code == 0:  # successfully
            print('Mask applied successfully: ' + os.path.basename(mask_file))
        else:
            print('\nError applied mask: ' + str(return_code))
            sys.exit(1)

    print('\nDONE: arrNorm successfully for:  {img_orig}\n'
          '      image normalized saved in: {img_norm}\n'.format(
        img_orig=os.path.basename(img_target),
        img_norm=os.path.basename(img_norm)))
