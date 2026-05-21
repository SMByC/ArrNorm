#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from osgeo import gdal

DEFAULT_BLOCK_ROWS = 256


def _iter_row_blocks(rows, block_rows=DEFAULT_BLOCK_ROWS):
    for y in range(0, rows, block_rows):
        yield y, min(block_rows, rows - y)


def _copy_spatial_metadata(src_ds, dst_ds):
    gt = src_ds.GetGeoTransform()
    if gt is not None:
        dst_ds.SetGeoTransform(gt)
    proj = src_ds.GetProjection()
    if proj is not None:
        dst_ds.SetProjection(proj)


def _is_float_dtype(gdal_dtype):
    return gdal_dtype in (gdal.GDT_Float32, gdal.GDT_Float64)


def _safe_neq(data, nodata_value, is_float):
    if is_float and np.isnan(nodata_value):
        return ~np.isnan(data)
    return data != nodata_value


def no_negative_value(input_path, output_path, nodata_value=None,
                      creation_options=None, block_rows=DEFAULT_BLOCK_ROWS):
    src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Cannot open raster: {input_path}")

    driver = gdal.GetDriverByName('GTiff')
    nbands = src_ds.RasterCount
    cols, rows = src_ds.RasterXSize, src_ds.RasterYSize
    dtype = src_ds.GetRasterBand(1).DataType

    co = list(creation_options or [])
    dst_ds = driver.Create(output_path, cols, rows, nbands, dtype, co)
    _copy_spatial_metadata(src_ds, dst_ds)

    for b in range(1, nbands + 1):
        src_band = src_ds.GetRasterBand(b)
        out_band = dst_ds.GetRasterBand(b)

        src_nodata = src_band.GetNoDataValue()
        is_float = _is_float_dtype(src_band.DataType)

        out_nodata = (
            nodata_value if nodata_value is not None
            else (src_nodata if src_nodata is not None else 0)
        )
        out_band.SetNoDataValue(float(out_nodata))

        for y_off, n_rows in _iter_row_blocks(rows, block_rows):
            data = src_band.ReadAsArray(0, y_off, cols, n_rows)

            if src_nodata is not None:
                valid = _safe_neq(data, src_nodata, is_float)
                negative_valid = valid & (data < 0)
                is_nodata = ~valid
                result = np.where(negative_valid | is_nodata, out_nodata, data)
            else:
                result = np.where(data < 0, out_nodata, data)

            out_band.WriteArray(result, 0, y_off)

        desc = src_band.GetDescription()
        if desc:
            out_band.SetDescription(desc)
        out_band.FlushCache()

    src_ds = dst_ds = None


def make_mask(input_path, output_path, nodata_value,
              block_rows=DEFAULT_BLOCK_ROWS):
    src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Cannot open raster: {input_path}")

    driver = gdal.GetDriverByName('GTiff')
    cols, rows = src_ds.RasterXSize, src_ds.RasterYSize
    src_band = src_ds.GetRasterBand(1)
    src_dtype = src_band.DataType
    is_float = _is_float_dtype(src_dtype)

    dst_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte,
                           ["COMPRESS=PACKBITS", "NBITS=1"])
    _copy_spatial_metadata(src_ds, dst_ds)

    out_band = dst_ds.GetRasterBand(1)

    for y_off, n_rows in _iter_row_blocks(rows, block_rows):
        data = src_band.ReadAsArray(0, y_off, cols, n_rows)
        mask = _safe_neq(data, nodata_value, is_float).astype(np.uint8)
        out_band.WriteArray(mask, 0, y_off)

    colors = gdal.ColorTable()
    colors.SetColorEntry(0, (0, 0, 0, 255))
    colors.SetColorEntry(1, (0, 255, 0, 255))
    out_band.SetRasterColorTable(colors)
    out_band.FlushCache()

    src_ds = dst_ds = None


def apply_mask(image_path, mask_path, output_path, nodata_value,
               creation_options=None, block_rows=DEFAULT_BLOCK_ROWS):
    img_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if img_ds is None:
        raise RuntimeError(f"Cannot open raster: {image_path}")

    mask_ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    if mask_ds is None:
        raise RuntimeError(f"Cannot open mask: {mask_path}")

    driver = gdal.GetDriverByName('GTiff')
    nbands = img_ds.RasterCount
    cols, rows = img_ds.RasterXSize, img_ds.RasterYSize
    dtype = img_ds.GetRasterBand(1).DataType

    if mask_ds.RasterXSize != cols or mask_ds.RasterYSize != rows:
        raise RuntimeError(
            f"Mask dimensions ({mask_ds.RasterXSize}x{mask_ds.RasterYSize}) "
            f"don't match image ({cols}x{rows})")

    co = list(creation_options or [])
    dst_ds = driver.Create(output_path, cols, rows, nbands, dtype, co)
    _copy_spatial_metadata(img_ds, dst_ds)

    mask_band = mask_ds.GetRasterBand(1)

    for b in range(1, nbands + 1):
        src_band = img_ds.GetRasterBand(b)
        out_band = dst_ds.GetRasterBand(b)
        out_band.SetNoDataValue(float(nodata_value))

        src_nodata = src_band.GetNoDataValue()
        is_float = _is_float_dtype(src_band.DataType)

        for y_off, n_rows in _iter_row_blocks(rows, block_rows):
            img_data = src_band.ReadAsArray(0, y_off, cols, n_rows)
            mask_data = mask_band.ReadAsArray(0, y_off, cols, n_rows)
            result = img_data * (mask_data == 1)
            if src_nodata is not None:
                valid = _safe_neq(img_data, src_nodata, is_float)
                result = np.where(valid, result, nodata_value)
            out_band.WriteArray(result, 0, y_off)

        desc = src_band.GetDescription()
        if desc:
            out_band.SetDescription(desc)
        out_band.FlushCache()

    img_ds = mask_ds = dst_ds = None
