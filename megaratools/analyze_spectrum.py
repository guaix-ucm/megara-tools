#
# Copyright 2019-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from astropy.io import fits
import argparse
import sys
import csv
from lmfit import minimize, Parameters

from .analyze import axvlines
from .analyze import gaussfunc, gaussfunc_gh, gauss2func
from .analyze import linfunc


def main(args=None):
# Parser
    parser = argparse.ArgumentParser(description='ANALYZE SPECTRUM',prog='analyze_spectrum')
    parser.add_argument('-s', '--spectrum', metavar='SPECTRUM/FILE_LIST', help='FITS spectrum / list of FITS spectra', type=argparse.FileType('rb'))
    parser.add_argument('-l', '--is-a-list', default=False, action="store_true", help='Use for -s being a list of FITS spectra')
    parser.add_argument('-f', '--method', default=0, choices=[0,1,2], metavar='FITTING FUNCTION (0,1,2)', help='Fitting function (0=gauss_hermite, 1=gauss, 2=double_gauss)', type=int)

    parser.add_argument('-w', '--ctwl', metavar='LINE CENTRAL WAVELENGTH', help='Central rest-frame wavelength for line (AA)',
                        type=float)
    parser.add_argument('-k', '--use-peak', default=False, action="store_true", help='Use peak first guess on central wavelength')
    parser.add_argument('-LW1', '--lcut1', metavar='LOWER WAVELENGTH - LINE', help='Lower rest-frame wavelength for line (AA)',
                        type=float)
    parser.add_argument('-LW2', '--lcut2', metavar='UPPER WAVELENGTH - LINE', help='Upper rest-frame wavelength for line (AA)',
                        type=float)
    parser.add_argument('-CW1', '--ccut1', metavar='LOWER WAVELENGTH - CONT', help='Lower rest-frame wavelength for cont. (AA)',
                        type=float)
    parser.add_argument('-CW2', '--ccut2', metavar='UPPER WAVELENGTH - CONT', help='Upper rest-frame wavelength for cont. (AA)',
                        type=float)
    parser.add_argument('-ECW1', '--eccut1', metavar='EXCLUDE FROM CONT. (LOWER WAVELENGTH)', help='Lower rest-frame wavelength of range to exclude for cont. (AA)',
                        type=float)
    parser.add_argument('-ECW2', '--eccut2', metavar='EXCLUDE FROM CONT. (UPPER WAVELENGTH)', help='Upper rest-frame wavelength of range to exclude for cont. (AA)',
                        type=float)
    parser.add_argument('-PW1', '--pcut1', metavar='LOWER WAVELENGTH - PLOT', help='Lower rest-frame wavelength for plot (AA)',
                        type=float)
    parser.add_argument('-PW2', '--pcut2', metavar='UPPER WAVELENGTH - PLOT', help='Upper rest-frame wavelength for plot (AA)',
                        type=float)
    parser.add_argument('-S2', '--scale-amp2', metavar='SCALE FACTOR FOR AMP2', help='Scale factor for amplitude 2', default=1.0, type=float)
    parser.add_argument('-t', '--spec-table', metavar='SPEC-TABLE', help='Additional spectrum table', type=argparse.FileType('r'))
    parser.add_argument('-c', '--catalog', metavar='LINECAT-TABLE', help='Cataloged lines CSV table', type=argparse.FileType('r'))
    parser.add_argument('-z', '--redshift', metavar='REDSHIFT', help='Redshift for target and catalog lines', type=float)
    parser.add_argument('-o', '--output', metavar='OUTPUT-PDF', help='Output PDF', type=argparse.FileType('w'))
    parser.add_argument('-p', '--plot', default=False, action="store_true" , help='Plot spectrum?')
    parser.add_argument('-n', '--no-legend', default=False, action="store_true", help='Legend?')

    args = parser.parse_args(args=args)
    
# Constants
    c_amstrong = 2.99792e+18  # Light speed in AA/s
    two = 2
    zero = np.float64(0.0)
    
    if args.redshift is not None:
       z=float(args.redshift)
    else:
       z=0.

# Plotting
    if args.spec_table!=None or args.spectrum!=None:
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
                 flux_vector = tbdata[i]
              else:
                 print ('unknown or not defined BUNIT [Jy, ELECTRON]')
                 sys.exit(1)
              flux.append(flux_vector)
  
          plt.xlim(float(args.pcut1),float(args.pcut2))
        
          if "LR" in vph:
              plt.plot(wave, flux, 'blue', label = 'input spectrum')
              R=6000.0
          else: 
              if "MR" in vph:
                  plt.plot(wave, flux, 'green', label = 'input spectrum')
                  R=12000.0
              else:
                  plt.plot(wave, flux, 'red', label = 'input spectrum')
                  R=20000.0

          fcont = []
          wcont = []
          fcont = flux[int((float(args.ccut1)*(1.+z)-lambda0)/cdelt+crpix):int((float(args.ccut2)*(1.+z)-lambda0)/cdelt+crpix)]
# Excluding range
          if (args.eccut1 is not None and args.eccut2 is not None):
              del fcont[int((float(args.eccut1)*(1.+z)-lambda0)/cdelt+crpix)-int((float(args.ccut1)*(1.+z)-lambda0)/cdelt+crpix):int((float(args.eccut2)*(1.+z)-lambda0)/cdelt+crpix)-int((float(args.ccut1)*(1.+z)-lambda0)/cdelt+crpix)]
          wcont = wave[int((float(args.ccut1)*(1.+z)-lambda0)/cdelt+crpix):int((float(args.ccut2)*(1.+z)-lambda0)/cdelt+crpix)]
          if (args.eccut1 is not None and args.eccut2 is not None):
              del wcont[int((float(args.eccut1)*(1.+z)-lambda0)/cdelt+crpix)-int((float(args.ccut1)*(1.+z)-lambda0)/cdelt+crpix):int((float(args.eccut2)*(1.+z)-lambda0)/cdelt+crpix)-int((float(args.ccut1)*(1.+z)-lambda0)/cdelt+crpix)]
          cmean = np.mean(fcont)

# Fitting continuum
          p_lin = Parameters()
          p_lin.add('slope', value=zero, vary=True)
          p_lin.add('yord', value=cmean, vary=True)
          print("FITTING CONTINUUM:")
          print("Input(slope,yord):  %10.3E %10.3E" % (0., cmean))
          err_lin = lambda p, x, y: linfunc(p, x) - y
          fitout_lin=minimize(err_lin,p_lin,args=(wcont,fcont))
          fitted_p_lin = fitout_lin.params
          pars_lin = [fitout_lin.params['slope'].value, fitout_lin.params['yord'].value,
                     fitout_lin.chisqr]
          print("Output(slope,yord): %10.3E %10.3E"%(fitout_lin.params['slope'].value, fitout_lin.params['yord'].value))
          print("Best-fitting chisqr continuum: %10.3E" % (fitout_lin.chisqr))
          fit_con = linfunc(fitted_p_lin, wcont)
          residuals = fcont - fit_con
          rms = np.std(residuals)

# Arrays for line profile
          fline = []
          fpline = []
          wline = []
          fline = flux[int((float(args.lcut1)*(1.+z)-lambda0)/cdelt+crpix):int((float(args.lcut2)*(1.+z)-lambda0)/cdelt+crpix)]
          wline = wave[int((float(args.lcut1)*(1.+z)-lambda0)/cdelt+crpix):int((float(args.lcut2)*(1.+z)-lambda0)/cdelt+crpix)]
          peak = np.amax(fline)
          fit_lin = linfunc(fitted_p_lin, wline)
          lcmean = np.mean(fit_lin) # Mean continuum within the line range
          fpline = fline - fit_lin
#          fpline = fline - cmean
          result = np.where(fline == np.amax(fline))
          lpeak = wline[result[0][0]]

          print ("BASIC NUMBERS:")
          print ("(mean,rms,lpk,pk,S/N)",lcmean,rms,lpeak,peak,peak/rms)

# Initial guess on parameters

          amp = peak-lcmean
          if (args.use_peak) == False:
            center = float(args.ctwl)*(1.+z)
          else:
            center = lpeak
          sigma = (float(args.ctwl)/R)*((1.+z)/2.35)
          skew = 0.0
          kurt = 0.0
          
          amp1 = 0.9*(peak-lcmean)
          sigma1 = (float(args.ctwl)/R)/2.35
          amp2 = 0.1*(peak-lcmean)
          sigma2 = 2.*(float(args.ctwl)/R)/2.35
          if (args.use_peak) == False:
            center1 = float(args.ctwl)*(1.+z)
            center2 = float(args.ctwl)*(1.+z)
          else:
            center1 = lpeak
            center2 = lpeak
                       
          if args.method is 0:
              p_gh=Parameters()
              p_gh.add('amp',value=amp, vary=True)
              p_gh.add('center',value=center, vary=True)
              p_gh.add('sigma',value=sigma, vary=True, min=0.8*sigma)
              p_gh.add('skew',value=skew,vary=True,min=-0.5,max=0.5)
              p_gh.add('kurt',value=kurt,vary=True,min=-0.5,max=0.5)
              print ("FITTING METHOD: GAUSS-HERMITE QUADRATURE")
              print ("Input(i0,l0,sigma,skew,kurt):  %10.3E %5.2f %5.2f %10.3E %10.3E"%(amp, center, sigma, skew, kurt))
              gausserr_gh = lambda p,x,y: gaussfunc_gh(p,x)-y 
          if args.method is 1:
              p_gh=Parameters()
              p_gh.add('amp',value=amp, vary=True)
              p_gh.add('center',value=center, vary=True)
              p_gh.add('sigma',value=1.2*sigma, vary=True, min=sigma)
              print ("FITTING METHOD: SINGLE GAUSSIAN")
              print ("Input(i0,l0,sigma):  %10.3E %5.2f %5.2f"%(amp, center, sigma))
              gausserr_gh = lambda p,x,y: gaussfunc(p,x)-y
          if args.method is 2:
              p_gh=Parameters()
              p_gh.add('amp1',value=amp1, vary=True);
              p_gh.add('center1',value=center1, vary=True);
              p_gh.add('sigma1',value=sigma1, vary=True, min=0.8*sigma1);
              p_gh.add('amp2',value=args.scale_amp2*amp2, vary=True);
              p_gh.add('center2',value=center2, vary=True);
              p_gh.add('sigma2',value=sigma2, vary=True, min=1.5*sigma1);
              print ("FITTING METHOD: DOUBLE GAUSSIAN")
              print ("Input(i1,l1,sig1,i2,l2,sig2):  %10.3E %5.2f %5.2f %10.3E %5.2f %5.2f"%(amp1, center1, sigma1, args.scale_amp2*amp2, center2, sigma2))
              gausserr_gh = lambda p,x,y: gauss2func(p,x)-y
                
          fitout_gh=minimize(gausserr_gh,p_gh,args=(wline,fpline))
          fitted_p_gh = fitout_gh.params
          tmp = fitout_gh
          if args.method is 2:
             tmp = fitted_p_gh
             tmp.add('amp',value=tmp['amp1'].value)
             tmp.add('center',value=tmp['center1'].value)
             tmp.add('sigma',value=tmp['sigma1'].value)
             fitted_p_gh1 = tmp
             fit_gh1=gaussfunc(fitted_p_gh1,wline)
             print ("Flux1 from model: %10.3E+/-%10.3E"%(cdelt*np.sum(fit_gh1), rms*cdelt*np.sqrt(2*len(fit_gh1)+(np.sum(fit_gh1)/lcmean)))) # Errors as in Tresse et al. (1999)
             tmp2 = fitted_p_gh
             tmp2.add('amp',value=tmp2['amp2'].value)
             tmp2.add('center',value=tmp2['center2'].value)
             tmp2.add('sigma',value=tmp2['sigma2'].value)
             fitted_p_gh2 = tmp2
             fit_gh2=gaussfunc(fitted_p_gh2,wline)
             print ("Flux2 from model: %10.3E+/-%10.3E"%(cdelt*np.sum(fit_gh2), rms*cdelt*np.sqrt(2*len(fit_gh2)+(np.sum(fit_gh2)/lcmean)))) # Errors as in Tresse et al. (1999)

          if args.method is 0:
              pars_gh=[fitout_gh.params['amp'].value,fitout_gh.params['center'].value,fitout_gh.params['sigma'].value,fitout_gh.params['skew'].value,fitout_gh.params['kurt'].value,fitout_gh.chisqr]
              fit_gh=gaussfunc_gh(fitted_p_gh,wline)
              print ("Output(i0,l0,sigma,skew,kurt): %10.3E %5.2f %5.2f %10.3E %10.3E"%(fitout_gh.params['amp'].value, fitout_gh.params['center'].value, fitout_gh.params['sigma'].value, fitout_gh.params['skew'].value, fitout_gh.params['kurt'].value))
          if args.method is 1:
              pars_gh=[fitout_gh.params['amp'].value,fitout_gh.params['center'].value,fitout_gh.params['sigma'].value,fitout_gh.chisqr]
              fit_gh=gaussfunc(fitted_p_gh,wline)
              print ("Output(i0,l0,sigma): %10.3E %5.2f %5.2f"%(fitout_gh.params['amp'].value, fitout_gh.params['center'].value, fitout_gh.params['sigma'].value))
          if args.method is 2:
              pars_gh=[fitout_gh.params['amp1'].value,fitout_gh.params['center1'].value,fitout_gh.params['sigma1'].value,fitout_gh.params['amp2'].value,fitout_gh.params['center2'].value,fitout_gh.params['sigma2'].value,fitout_gh.chisqr] 
              fit_gh=gauss2func(fitted_p_gh,wline)
              print ("Output(i1,l1,sig1,i2,l2,sig2): %10.3E %5.2f %5.2f %10.3E %5.2f %5.2f"%(fitout_gh.params['amp1'].value, fitout_gh.params['center1'].value, fitout_gh.params['sigma1'].value, fitout_gh.params['amp2'].value, fitout_gh.params['center2'].value, fitout_gh.params['sigma2'].value))
#          print ("CDELT1: %6.3f Angstroms/pixel"%(cdelt))
          eEWd=(rms*cdelt*(cdelt*np.sum(fpline)/lcmean)/(cdelt*np.sum(fpline)))*np.sqrt(2*len(fpline)+np.sum(fpline)/lcmean+(np.sum(fpline)/lcmean)**2/len(fpline))
          eEWm=(rms*cdelt*(cdelt*np.sum(fit_gh)/lcmean)/(cdelt*np.sum(fit_gh)))*np.sqrt(2*len(fit_gh)+np.sum(fit_gh)/lcmean+(np.sum(fpline)/lcmean)**2/len(fit_gh))
          print ("Flux & EW from data:  %10.3E+/-%10.3E %5.2f+/-%5.2f"%(cdelt*np.sum(fpline), rms*cdelt*np.sqrt(2*len(fpline)+(np.sum(fpline)/lcmean)), cdelt*np.sum(fpline)/lcmean, eEWd)) # Errors as in Tresse et al. (1999)
          print ("Flux & EW from model: %10.3E+/-%10.3E %5.2f+/-%5.2f"%(cdelt*np.sum(fit_gh), rms*cdelt*np.sqrt(2*len(fit_gh)+(np.sum(fit_gh)/lcmean)), cdelt*np.sum(fit_gh)/lcmean, eEWm)) # Errors as in Tresse et al. (1999)
          print ("Best-fitting chisqr: %10.3E"%(fitout_gh.chisqr))

# Here we should write the results to a file and decide whether we should set them first to zero if S/N is below some threshold

          plt.plot(wcont, fit_con, 'red', label = 'Continuum fit')
          resid_gh=fpline-fit_gh
          plt.plot(wline, fit_gh+fit_lin, 'orange', label = 'best fit')

          lflines = [lambda0 + (pf1 - crpix) * cdelt, lambda0 + (pf2 - crpix) * cdelt]
          lmlines = [lambda0 + (pm1 - crpix) * cdelt, lambda0 + (pm2 - crpix) * cdelt]
          lllines = [float(args.lcut1)*(1.+z),float(args.lcut2)*(1.+z)]
          lclines = [float(args.ccut1)*(1.+z),float(args.ccut2)*(1.+z)]
          leclines = [float(args.eccut1)*(1.+z),float(args.eccut2)*(1.+z)]
          cwline = [float(args.ctwl)*(1.+z)]
            
          if (args.no_legend) == False:
              axvlines(lmlines, color='cyan', label = 'All-fiber range', linestyle = '-')
              axvlines(lflines, color='brown', label = 'Sensitivity range', linestyle = '--')
              if (args.use_peak) == False:
                 axvlines(cwline, color='black', label = 'Central wavelength', linestyle = '-')
              axvlines(lllines, color='gray', label = 'Line-fitting range', linestyle = '-')
              axvlines(lclines, color='gray', label = 'Continuum range', linestyle = '--')
              axvlines(leclines, color='gray', linestyle = '--')

    if args.spec_table!=None:
# Reading standard-star spectrum table
       xdat, ydat = np.loadtxt(args.spec_table, skiprows=1, usecols=(0, 1), unpack=True)
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
            a = np.array(list(spamreader))
            for i in a[:,1]:
                slines.append(float(i)*(1+z))
            for i in a[:,0]:
                llines.append(str(i))
                ylines.append(random.uniform(0.9,0.98))
            axvlines(slines, color='green', label=None, linestyle = ':')
            trans = transforms.blended_transform_factory(plt.gca().transData, plt.gca().transAxes)
            for i in range(0,len(slines)):
                plt.text(slines[i],ylines[i],llines[i],transform=trans,fontsize='smaller')

# Plotting
    if args.spectrum!=None:
       if bunit == 'Jy' or 'CGS' or 'cgs' in bunit:
          plt.ylabel('flux [erg s$^{-1}$ cm$^{-2}$ $\AA$$^{-1}$]')
       else:
          plt.ylabel('ELECTRON')
    elif args.spec_table!=None:
         plt.ylabel('flux [erg s$^{-1}$ cm$^{-2}$ $\AA$$^{-1}$]')

    if args.spec_table!=None or args.spectrum!=None:
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
