#
# Copyright 2019-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import argparse
import sys
import csv

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from astropy.io import fits

from .analyze import axvlines

plt.rcParams.update({'font.size': 10})


def main(args=None):
# Parser
    parser = argparse.ArgumentParser(description='Input spectrum and table',prog='plot_spectrum')
    parser.add_argument('-s', '--spectrum', metavar='SPECTRUM/FILE_LIST', help='FITS spectrum / list of FITS spectra', type=argparse.FileType('rb'))
    parser.add_argument('-l', '--is-a-list', default=False, action="store_true", help='Use for -s being a list of FITS spectra')
    parser.add_argument('-t', '--std-table', metavar='STD-TABLE', help='Standard-star spectrum table', type=argparse.FileType('r'))
    parser.add_argument('-c', '--catalog', metavar='LINECAT-TABLE', help='Cataloged lines CSV table', type=argparse.FileType('r'))
    parser.add_argument('-z', '--redshift', metavar='LINECAT-Z', help='Redshift for catalog lines', type=float)
    parser.add_argument('-o', '--output', metavar='OUTPUT-PDF', help='Output PDF', type=argparse.FileType('w'))
    parser.add_argument('-e', '--efficiency', default=False, action="store_true" , help='Efficiency?')
    parser.add_argument('-p', '--plot', default=False, action="store_true" , help='Plot spectrum?')
    parser.add_argument('-n', '--no-legend', default=False, action="store_true", help='Legend?')
    parser.add_argument('-L1', '--min-lambda', metavar='INITIAL LAMBDA', help='Initial (rest-frame) lambda to plot', type=float)
    parser.add_argument('-L2', '--max-lambda', metavar='LAST LAMBDA', help='Last (rest-frame) lambda to plot', type=float)
    parser.add_argument('-F1', '--min-flambda', metavar='YMIN FLUX', help='Minimum flux to plot', type=float)
    parser.add_argument('-F2', '--max-flambda', metavar='YMAX FLUX', help='Maximum flux to plot', type=float)
    parser.add_argument('-T', '--title', metavar='PLOT TITLE', help='Title of the plot')

    args = parser.parse_args(args=args)

# Constants
    c_amstrong = 2.99792e+18  # Light speed in AA/s
    gtceffarea = 7.3e+5 # GTC effective area in cm**2
# Internal parameters
    plosses = 0.80 # Pupil losses
    tlosses = 0.80 # Telescope losses (relative to effective area)

    if args.redshift is not None:
       z=float(args.redshift)
    else:
       z=0.

# Plotting
    if args.std_table!=None or args.spectrum!=None:
       plt.figure()

    if args.spectrum!=None:   
# Reading spectrum/spectra
       if args.is_a_list == True:
          list_files = args.spectrum.readlines() 
          list_files = [str(x.strip(),'utf-8') for x in list_files] 
       else: 
          list_files = [args.spectrum.name] 
       for ifile in list_files:
          hdulist = fits.open(ifile)
          tbdata = hdulist[0].data
          prihdr = hdulist[0].header
          lambda0 = prihdr['CRVAL1']
          cdelt = prihdr['CDELT1']
          crpix = prihdr['CRPIX1']
          if ('VPH' in prihdr): 
             vph = prihdr['VPH'] 
          else:
             vph = 'LR'

          lambda_fin = lambda0 + (len(tbdata))*cdelt
       
          if 'BUNIT' in prihdr:
             bunit = prihdr['BUNIT']
          else:
             bunit = 'Jy' # To take into account that the QLA-generated 1D spectra have not BUNIT keyword
          if 'PIXLIMF1' in prihdr:
             pf1 = prihdr['PIXLIMF1'] # Sensitivity function computed in this region (beginning)
          else: 
             pf1 = crpix 
          if 'PIXLIMF2'in prihdr:
             pf2 = prihdr['PIXLIMF2'] # Sensitivity function computed in this region (end)
          else:
             pf2 = len(tbdata)
          if 'PIXLIMM1' in prihdr:
             pm1 = prihdr['PIXLIMM1'] # All fibers include this region (beginning)
          else:
             pm1 = crpix
          if 'PIXLIMM2' in prihdr:
             pm2 = prihdr['PIXLIMM2'] # All fibers include this region (end)
          else:
             pm2 = len(tbdata)

          flux = []
          wave = []
          for i in range(0,len(tbdata)):
              lambda_i = lambda0 + i*cdelt
              wave.append(lambda_i)
              if bunit == 'Jy':
                 flux_vector = (1e-23*tbdata[i]*c_amstrong)/(lambda_i**2)
              elif bunit == 'ELECTRON' or 'CGS' or 'cgs' in bunit:
                 if args.efficiency == True: 
                    eperjy = (gtceffarea*(1.51e+3/lambda_i))*cdelt # Photons to be collected /GTC/sec/pix/Jy
                    flux_vector = tbdata[i]/(eperjy*plosses*tlosses)
                 else:
                    flux_vector = tbdata[i]
              else:
                 print ('unknown or not defined BUNIT [Jy, ELECTRON]')
                 sys.exit(1)
              flux.append(flux_vector)
          lflines = [lambda0 + (pf1 - crpix) * cdelt, lambda0 + (pf2 - crpix) * cdelt]
          lmlines = [lambda0 + (pm1 - crpix) * cdelt, lambda0 + (pm2 - crpix) * cdelt]
          if "LR" in vph:
              plt.plot(wave, flux, 'blue', label = 'input spectrum')
          else: 
              if "MR" in vph:
                  plt.plot(wave, flux, 'green', label = 'input spectrum')
              else:
                  plt.plot(wave, flux, 'red', label = 'input spectrum')
          if (args.no_legend) == False:
              axvlines(lmlines, color='cyan', label = 'All-fiber range', linestyle = '-')
              axvlines(lflines, color='brown', label = 'Sensitivity range', linestyle = '--')

    if args.std_table!=None:
# Reading standard-star spectrum table
       xdat, ydat = np.loadtxt(args.std_table, skiprows=1, usecols=(0, 1), unpack=True)
       fydat = []
       for i in range(0,len(ydat)):
           flux_value = (1e-23*(3631.0*10.**(-0.4*ydat[i]))*c_amstrong)/(xdat[i]**2)
           fydat.append(flux_value)
       plt.plot(xdat, fydat, 'red', label = 'tabulated spectrum')

    if args.catalog!=None:
# Reading input line catalog
        with open(args.catalog.name, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            slines=[] # Wavelengths
            llines=[] # Labels
            ylines=[] # Y positions

            c=0
            a = np.array(list(spamreader))
            for i in a[:,1]:
                if ((args.min_lambda is None or (float(i) > (args.min_lambda*(1+z)))) and (args.max_lambda is None or (float(i) < (args.max_lambda*(1+z))))):
                    slines.append(float(i)*(1+z))
                    llines.append(str(a[c,0]))
                    ylines.append(random.uniform(0.9,0.98))
                c += 1
            axvlines(slines, color='green', label=None, linestyle = ':')
            trans = transforms.blended_transform_factory(plt.gca().transData, plt.gca().transAxes)
            for i in range(0,len(slines)):
                plt.text(slines[i],ylines[i],llines[i],transform=trans,fontsize='smaller')

# Plotting
    plt.title(args.title, fontsize='smaller')
    if args.spectrum!=None:
       if bunit == 'Jy' or 'CGS' or 'cgs' in bunit:
          plt.ylabel('flux [erg s$^{-1}$ cm$^{-2}$ A$^{-1}$]')
       else:
          if args.efficiency == True:
             plt.ylabel('Efficiency')
          else:
             plt.ylabel('ELECTRON')
    elif args.std_table!=None:
         plt.ylabel('flux [erg s$^{-1}$ cm$^{-2}$ A$^{-1}$]')

    if args.min_lambda != None:
        plt.xlim(left=((1+z)*args.min_lambda))
    if args.max_lambda != None:
        plt.xlim(right=((1+z)*args.max_lambda))
    if args.min_flambda != None:
        plt.ylim(bottom=args.min_flambda)
    if args.max_flambda != None:
        plt.ylim(top=args.max_flambda)

    if args.std_table!=None or args.spectrum!=None:
       plt.xlabel('wavelength')
       if args.no_legend == True:
          plt.legend('', frameon=False)
       else:
          plt.legend()
       if args.plot == True:
          plt.show()
       else:
          if args.output!=None:
             plt.savefig(args.output.name)
          else:
             plt.show()

if __name__ == '__main__':

    main()
