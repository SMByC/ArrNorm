"""Unit tests for core.radcal helpers."""
import numpy as np
import pytest
from osgeo import gdal

from core.radcal import _clip_for_dtype


class TestClipForDtype:

    def test_byte_clips_above_255(self):
        arr = np.array([0.0, 127.0, 255.0, 256.0, 1000.0])
        np.testing.assert_array_equal(
            _clip_for_dtype(arr, gdal.GDT_Byte),
            [0, 127, 255, 255, 255],
        )

    def test_byte_clips_below_0(self):
        arr = np.array([-500.0, -1.0, 0.0, 10.0])
        np.testing.assert_array_equal(
            _clip_for_dtype(arr, gdal.GDT_Byte),
            [0, 0, 0, 10],
        )

    def test_uint16_clips_above(self):
        arr = np.array([0.0, 32768.0, 65535.0, 65536.0, 1e9])
        np.testing.assert_array_equal(
            _clip_for_dtype(arr, gdal.GDT_UInt16),
            [0, 32768, 65535, 65535, 65535],
        )

    def test_uint16_clips_below(self):
        arr = np.array([-1.0, 0.0, 100.0])
        np.testing.assert_array_equal(
            _clip_for_dtype(arr, gdal.GDT_UInt16),
            [0, 0, 100],
        )

    def test_int16_clips_both_ends(self):
        arr = np.array([-32769.0, -32768.0, 0.0, 32767.0, 32768.0])
        np.testing.assert_array_equal(
            _clip_for_dtype(arr, gdal.GDT_Int16),
            [-32768, -32768, 0, 32767, 32767],
        )

    def test_uint16_in_range_unchanged(self):
        arr = np.array([0.0, 100.0, 1000.0, 50000.0, 65535.0])
        np.testing.assert_array_equal(
            _clip_for_dtype(arr, gdal.GDT_UInt16),
            arr,
        )

    def test_float32_passthrough(self):
        arr = np.array([-1e10, 0.0, 1e10])
        result = _clip_for_dtype(arr, gdal.GDT_Float32)
        np.testing.assert_array_equal(result, arr)

    def test_float64_passthrough(self):
        arr = np.array([-1e15, 0.0, 1e15])
        result = _clip_for_dtype(arr, gdal.GDT_Float64)
        np.testing.assert_array_equal(result, arr)

    def test_returns_numpy_array(self):
        arr = np.array([0.0, 100.0, 300.0])
        result = _clip_for_dtype(arr, gdal.GDT_Byte)
        assert isinstance(result, np.ndarray)

    def test_uint32_clips(self):
        arr = np.array([-1.0, 0.0, 4294967295.0, 4294967296.0])
        np.testing.assert_array_equal(
            _clip_for_dtype(arr, gdal.GDT_UInt32),
            [0, 0, 4294967295, 4294967295],
        )
