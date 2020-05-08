#
# Copyright 2019-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from __future__ import print_function
import numpy as np
import math
import sys
import re

def main(args=None):
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import argparse
    from megaradrp.datamodel import MegaraDataModel

# Parser
    parser = argparse.ArgumentParser(description='Extract spectrum based on fiber IDs',prog='extract_spectrum')
    parser.add_argument('-s', '--spectrum', metavar='RSS-SPECTRUM', help='RSS FITS spectrum', type=argparse.FileType('rb'))
    parser.add_argument('-t', '--ids-table', metavar='INPUT-TABLE', help='File with list of IDs', type=argparse.FileType('r'))
    parser.add_argument('-c', '--column', metavar='COLUMN', help='Column to select from table', type=int) 
    parser.add_argument('-g', '--grep-string', metavar='GREP-STRING', help='String to do grep in table')
    parser.add_argument('-o', '--output', metavar='OUTPUT-SPECTRUM', help='Output 1D spectrum', type=argparse.FileType('wb'))
    parser.add_argument('-p', '--plot', default=False, action="store_true" , help='Plot spectrum instead?')

    args = parser.parse_args(args=args)

    spec = args.spectrum # Input RSS FITS spectrum
    itable = args.ids_table # Input File with list of IDs
    col = args.column-1 # Column to select from table
    gstring = args.grep_string # String to do grep in table
    ofile = args.output # Output 1D spectrum   

# Reading spectrum
    rss = fits.open(spec)
    rss_data = rss[0].data
    prihdr = rss[0].header

    crval = prihdr['CRVAL1']
    crpix = prihdr['CRPIX1']
    cdelt = prihdr['CDELT1']

    datamodel = MegaraDataModel()

    fiberconf = datamodel.get_fiberconf(rss)
    conected = fiberconf.conected_fibers()
    rows = [conf.fibid - 1 for conf in conected]
    # Put in sellist all rows that need to be added [fibid=row_in_sellist(that starts in zero)+1] -> row_in_sellist = fibid-1 

    sellist = []
    for line in open(itable.name, 'r'):
        if re.search(gstring, line):
           sellist.append(int(line.split()[col])-1)
           if line == None:
              print ('no matches found')
              sys.exit(1) 

    region = rss_data[sellist, :]

    wave = []
    newima = np.sum(region, axis=0)
    for i in range(0,len(newima)):
        lambda_i = crval + i*cdelt
        wave.append(lambda_i)

    if args.plot == True:
# Plot spectrum
       plt.figure()
       plt.plot(wave, newima, 'red', label = 'extracted spectrum')
       plt.legend()
       plt.xlabel('wavelength')
       plt.ylabel('flux [Jy]')
       plt.show()
    else:
# Save spectrum
       ext_spectrum = fits.PrimaryHDU(newima, header=rss[0].header)
       ext_spectrum.header.remove('CRVAL2', ignore_missing=True, remove_all=True)
       ext_spectrum.header.remove('CRPIX2', ignore_missing=True, remove_all=True)
       ext_spectrum.header.remove('CDELT2', ignore_missing=True, remove_all=True)
       ext_spectrum.header.remove('CTYPE2', ignore_missing=True, remove_all=True)
       ext_spectrum.writeto(ofile, overwrite=True)

#    spec = copy_img(rss)
    
if __name__ == '__main__':

    main()

