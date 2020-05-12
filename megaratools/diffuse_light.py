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
import matplotlib.pyplot as plt

import numina.types.structured as structured

def main(args=None):
    from astropy.io import fits
    import argparse
    from megaradrp.datamodel import MegaraDataModel
    from matplotlib.backends.backend_pdf import PdfPages

# Parser
    parser = argparse.ArgumentParser(description='Cleaning of diffuse light from a reduced (non-RSS) MEGARA image',prog='clean_diffuse_light')
    parser.add_argument('-i', '--input', metavar='INPUT-IMAGE', help='Reduced FITS image', type=argparse.FileType('rb'))
    parser.add_argument('-o', '--output', metavar='OUTPUT-IMAGE', default='model.fits', help='Output diffuse-light FITS image', type=argparse.FileType('wb'))
    parser.add_argument('-r', '--residuals', metavar='RESIDUALS-IMAGE', default='residuals.fits', help='Output residual FITS image', type=argparse.FileType('wb'))
    parser.add_argument('-t', '--traces', metavar='MASTER-TRACES', help='Master traces JSON file', type=argparse.FileType('r'))
    parser.add_argument('-s', '--shift', metavar='SHIFT-TRACES', default=0., help='Traces shift', type=float)
    parser.add_argument('-w', '--window', metavar='SEARCH-WINDOW', default=6., help='Window around traces to search for non-illuminated fibers', type=float)
    parser.add_argument('-d', '--degree', metavar='DEGREE-POLY-COLS', default=4., help='Degree of polynomial fit for columns', type=float)
    parser.add_argument('-d2', '--degree-rows', metavar='DEGREE-POLY-ROWS', default=4., help='Degree of polynomial fit for rows', type=float)
    parser.add_argument('-p', '--outplot', metavar='OUTPUT-PLOT', help='Output plots', type=argparse.FileType('w'))
    parser.add_argument('-b', '--binning', metavar='SPECTRAL-BINNING', default=50, help='Binning in the spectral direction', type=int)
    parser.add_argument('-e', '--exclude', metavar='EXCLUDE-REGION', help='Exclude region (c1 c2 r1 r2), e.g. 2407 2720 0 164', nargs='+', type=int)
    parser.add_argument('-2D', '--two-dimensional', default=False, action="store_true", help='Two-dimensional fitting?')

    args = parser.parse_args(args=args)

    inputimage = args.input # Input reduced FITS image
    outputimage = args.output # Output diffuse-light FITS image
    resimage = args.residuals # Output residuals FITS image
    mtraces = args.traces
    otraces = args.shift

    hdulist = fits.open(inputimage)
    image_header = hdulist[0].header
    image2d = hdulist[0].data
 

    if args.exclude is not None:
        x1, x2, y1, y2 = args.exclude
        image2d[y1:y2,x1:x2] = np.NaN
    
    if 'naxis1' in image_header:
        naxis1 = image_header['naxis1']
    else:
        naxis1 = 4096
    if 'naxis2' in image_header:
        naxis2 = image_header['naxis2']
    else:
        naxis2 = 4112
    
    apers = structured.open(mtraces.name)

    columns = np.arange(0,naxis1)
    rows = np.arange(0,naxis2)
    model = np.empty_like(image2d)
    model2 = np.empty_like(image2d)
    bmodel = np.empty_like(image2d)
    
    if args.outplot!=None:
        pdf_pages = PdfPages(args.outplot.name)

    for i in columns[args.binning::args.binning]:
        fiberspos = [cc.polynomial(i) for cc in apers.contents if cc.valid]
        fluxes = np.nanmean(image2d[:,i-args.binning:i+args.binning],axis=1)

        lrows=[]
        for row in rows:
            for fibpos in fiberspos:
                if (abs(fibpos+args.shift-row) <= args.window):
                    lrows.append(row)
            if (np.isnan(fluxes[row])):
                lrows.append(row)
            
        frows=np.delete(rows,lrows)
        ffluxes=np.delete(fluxes,lrows)  
        plt.clf()
        plt.plot(rows,fluxes)
        plt.plot(frows,ffluxes,'.', color='black')
    
        bsize = min(i+args.binning,naxis1)-(i-args.binning)
        dfit=np.polyfit(frows,ffluxes,args.degree)
        pdfit = np.poly1d(dfit)
        plt.plot(rows,pdfit(rows))
        tmp = np.tile(pdfit(rows).reshape(naxis2,1),bsize)
        model[:,i-args.binning:i+args.binning]=tmp
        print('column:',i)
           
        plt.title('column: '+str(i+1))
        plt.xlabel('Y-position')
        plt.ylabel('Flux')
        if args.outplot!=None:
             pdf_pages.savefig()
        else:
             plt.show()

    if (args.two_dimensional):
        for i in rows[args.binning::args.binning]:
            fluxes = np.nanmean(model[i-args.binning:i+args.binning,:],axis=0)
            plt.clf()
            plt.plot(columns,fluxes)
            bsize = min(i+args.binning,naxis2)-(i-args.binning)
            dfit=np.polyfit(columns,fluxes,args.degree_rows)
            pdfit = np.poly1d(dfit)
            plt.plot(columns,pdfit(columns))
            tmp = np.transpose(np.tile(pdfit(columns).reshape(naxis1,1),bsize))
            model2[i-args.binning:i+args.binning,:]=tmp
            print('row:',i)

            plt.title('row: '+str(i+1))
            plt.xlabel('X-position')
            plt.ylabel('Flux')
            if args.outplot!=None:
                 pdf_pages.savefig()
            else:
                 plt.show()

    if args.outplot!=None:
        pdf_pages.close()
    plt.close()

    if (args.two_dimensional):
        bmodel = model2
    else:
        bmodel = model

    hdulist[0].data = bmodel
    hdulist.writeto(args.output.name, overwrite = True)
    hdulist[0].data = image2d - bmodel
    hdulist.writeto(args.residuals.name, overwrite = True)
    hdulist.close()

if __name__ == '__main__':

    main()

