#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from arrNorm import iMad, radcal

# ==============================================================================
# PARSER AND CHECK ARGUMENTS

# Create parser arguments
arguments = argparse.ArgumentParser(
    prog="normalize",
    description="iMad normalize",
    epilog="SMBYC - Xavier Corredor Llano <xcorredorl@ideam.gov.co>",
    formatter_class=argparse.RawTextHelpFormatter)

arguments.add_argument('-ref', type=str, required=True,
                       help='reference image for iMad normalize')

arguments.add_argument('-i', type=int, default=25,
                       help='number of iterations', required=False)

# set path for save the results
arguments.add_argument('-t', type=float, default=0.95,
                       help='no-change probability threshold', required=False)

arguments.add_argument('images', type=str, nargs='+',
                       help='images to apply the iMad normalization')

arg = arguments.parse_args()

print("\nAutomatic relative radiometric normalization")

# ==============================================================================
# PROCESS IMAGES

for img_target in arg.images:

    print("\niMad process for: ", img_target)
    img_imad = iMad.main(arg.ref, img_target)

    print("\nRadcal process for: ", img_target, " with iMad image: ", img_imad)
    img_norm = radcal.main(img_imad)