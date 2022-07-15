#
# Copyright 2022 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description='Obtain a sigma image from a model map file', prog='modelrss')
    parser.add_argument("model", help="Model map file")

    args = parser.parse_args(args=args)
