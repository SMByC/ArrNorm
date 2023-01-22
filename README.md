# ArrNorm

ArrNorm is a command line program for apply the radiometric normalization to the target image based on reference image using the IR-MAD algorithm to locate invariant/variant pixels for a relative radiometric normalization.

The algorithm takes advantage of the linear and affine invariance of the Multivariate alteration detection (MAD)
transformation to perform a relative radiometric normalization of the images involved in the transformation, using the correlation of the iteratively reweighted MAD (IR-MAD) [1]

Stop condition is set by max iteration or with a minimum no-change probability threshold. With more iterations the algorithm try to find a better match to the reference image, decreasing the delta, the plugin select the best delta for the final result. However, after several iterations the changes in the delta are imperceptible.

[1] M. J. Canty (2014): Image Analysis, Classification and Change Detection in Remote Sensing, with Algorithms for ENVI/IDL and Python (Third Revised Edition), Taylor and Francis CRC Press.

![](example.jpg)

<figcaption>Fig.1 - Example of a Landsat image normalization, using a multi-year average (reference) to normalize a scene. Those pixel changes are because, some remote sensing imagery pixel values are affected by different causes such as: sensor angle, sun position, clouds and seasons. Be sure that you are using the same 'style' for all layers in Qgis (use copy/paste style for that) to visualize and check the normalization visually.</figcaption>

---

> Check the [ArrNorm-Qgis-processing](https://github.com/SMByC/ArrNorm-Qgis-processing) implementation of Arrnorm as a Qgis processing.

## Parameters

* **--ref** *reference image*: The reference image to normalize the target image.
* **-p** *number of threads*: Number of threads to use (default: 1).
* **-i** *max iterations*: Maximum number of iterations (default: 30).
* **-m** *create mask*: Create a mask file with the valid data (default: False).

For other parameters check the help:

```bash
$ arrnorm -h
```

Example:

```bash
$ arrnorm -p 4 -i 20 -m --ref reference.tif *.tif
```

## About us

ArrNorm was developing, designed and implemented by the Group of Forest and Carbon Monitoring System (SMByC), operated by the Institute of Hydrology, Meteorology and Environmental Studies (IDEAM) - Colombia.

Author and developer: *Xavier C. Llano* *<xavier.corredor.llano@gmail.com>*  
Theoretical support, tester and product verification: SMByC-PDI group

## License

ArrNorm is a free/libre software and is licensed under the GNU General Public License.
