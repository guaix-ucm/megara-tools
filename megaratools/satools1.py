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

    parser = argparse.ArgumentParser(description='Aperture photometry', prog='phot_rss')
    parser.add_argument("rss", help="Input table with list of RSS files", type=argparse.FileType('rb'))
    parser.add_argument('-r', metavar='Number of rings', default=2)

    args = parser.parse_args(args=args)
