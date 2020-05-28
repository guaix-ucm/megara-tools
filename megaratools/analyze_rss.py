#
# Copyright 2019-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import argparse, textwrap
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit import minimize, Parameters
from matplotlib.backends.backend_pdf import PdfPages

from .analyze import axvlines
from .analyze import gaussfunc, gauss2func, gaussfunc_gh
from .analyze import linfunc

def main(args=None):
# Parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,description=
        textwrap.dedent('''\
             __________________  ANALYZE_RSS PROGRAM  __________________\n
             ____________  OUTPUT PARAMETER IN OUTPUT FITS  ____________
             Property  channel description\n
        ...  FM        #  0 Fitting method (0=gauss-hermite,1=1gauss,2=2gauss)
        ...  CONTINUUM #  1 Continuum level in cgs
        ...  NOISE     #  2 rms in cgs
        ...  SNR       #  3 S/N at the peak of the line
        ...  FLUXD     #  4 Flux from window_data - window_continuum
        ...  EWD       #  5 Flux from window_data - window_continuum / mean_continuum
        ...  FLUXF     #  6 Flux from best-fitting function(s)
        ...  EWF       #  7 EW from best-fitting function(s)
        ...  H0        #  8 amplitude for methods 0 & 1 & 2 (first gaussian)
        ...  H1        #  9 central lambda for methods 0 & 1 & 2 (first gaussian)
        ...  H2        # 10 sigma (in AA) for methods 0 & 1 & 2 (first gaussian)
        ...  H3        # 11 h3 for method 0
        ...  H4        # 12 h4 for method 0
        ...  H0B       # 13 amplitude for method 2 (second gaussian)
        ...  H1B       # 14 central lambda for method 2 (second gaussian)  
        ...  H2B       # 15 sigma (in AA) for method 2 (second gaussian)
        ...  H1KS      # 16 velocity in km/s from H1 (1st g)
        ...  H2KS      # 17 sigma in km/s from H2 (1st g)
        ...  H2KLC     # 18 sigma in km/s from H2 corrected for instrumental sigma (1st g)
        ...  H1KSB     # 19 velocity in km/s from H1B (2nd g)
        ...  H2KSB     # 20 sigma in km/s from H2B (2nd g)
        ...  H2KLCB    # 21 sigma in km/s from H2 corrected for instrumental sigma (2nd g)
        ...  FLUXF1    # 22 Flux from best-fitting 1st gaussian 
        ...  FLUXF2    # 23 Flux from best-fitting 2nd gaussian 
        ...  EFLUXD    # 24 Error of 4 (Flux from window_data - window_continuum) 
        ...  EEWD      # 25 Error of 5 (Flux from window_data - window_continuum / mean_continuum)
        ...  EFLUXF    # 26 Error of 6 (Flux from best-fitting function(s))
        ...  EEWF      # 27 Error of 7 (EW from best-fitting function(s))
        ...  CHI2      # 28 best-fitting chi^2 (cgs)'''),
        prog='analyze_rss')
    
    parser.add_argument('-s', '--spectrum', metavar='RSS FILE', help='RSS input file', type=argparse.FileType('rb'))
    parser.add_argument('-f', '--method', default=0, choices=[0,1,2], metavar='FITTING FUNCTION (0,1,2)', help='Fitting function (0=gauss_hermite, 1=gauss, 2=double_gauss)', type=int)
    parser.add_argument('-S', '--limsnr', default=5, metavar='MINIMUM S/N', help='Mininum Signal-to-noise ratio in each spaxel', type=float)
    parser.add_argument('-w', '--ctwl', metavar='LINE CENTRAL WAVELENGTH', help='Central rest-frame wavelength for line (AA)',
                        type=float)
    parser.add_argument('-k', '--use-peak', default=True, action="store_true", help='Use peak first guess on central wavelength')
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
    parser.add_argument('-PW1', '--pcut1', metavar='LOWER WAVELENGTH - PLOT', help='Lower (observed) wavelength for plot (AA)',
                        type=float)
    parser.add_argument('-PW2', '--pcut2', metavar='UPPER WAVELENGTH - PLOT', help='Upper (observed) wavelength for plot (AA)',
                        type=float)
    parser.add_argument('-S2', '--scale-amp2', metavar='SCALE FACTOR FOR AMP2', help='Scale factor for amplitude 2', default=1.0, type=float)
    parser.add_argument('-z', '--redshift', metavar='REDSHIFT', help='Redshift for target and catalog lines', type=float)
    parser.add_argument('-o', '--output', metavar='OUTPUT-PDF', help='Output PDF', type=argparse.FileType('bw'))
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help='Verbose mode for fitting results?')
    parser.add_argument('-O', '--output-rss', default="analyze_rss.fits", metavar='OUTPUT RSS FILE', help='Output RSS file', type=argparse.FileType('bw'))
    parser.add_argument('-of', '--output-fibers', metavar='OUTPUT FIBERS LIST', help='Output list of fibers above minimum Signal-to-noise ratio', type=argparse.FileType('bw'))


    args = parser.parse_args(args=args)
    
    plt.rcParams.update({'figure.max_open_warning': 0})
    
# Constants
    c_amstrong = 2.99792e+18  # Light speed in AA/s
    c_km = 2.99792e+5 # Light speed in km/s
    nonfit = float('NaN') # Value assigned to the out parameters when fit is not performed (could be 0. or NaN)
    zero = np.float64(0.0)

    if args.redshift is not None:
       z=float(args.redshift)
    else:
       z=0.

# Plotting

    if args.spectrum!=None:   
       ima = fits.open(args.spectrum)
       prihdr = ima[0].header
       lambda0 = prihdr['CRVAL1']
       cdelt = prihdr['CDELT1']
       crpix = prihdr['CRPIX1']
       vph = prihdr['VPH']
       ny = prihdr['NAXIS1']
       nx = prihdr['NAXIS2'] 
       rss = ima[0].data
    
       if 'BUNIT' in prihdr:
          bunit = prihdr['BUNIT']
       else:
          bunit = 'Jy' # To take into account that old QLA-generated 1D spectra might not BUNIT keyword
                
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

       plt.figure()
       plt.title('Collapsed spectrum')
       plt.ylabel('flux [erg s$^{-1}$ cm$^{-2}$ $\AA$$^{-1}$]')
       plt.xlabel('wavelength')
        
       flux = []
       wave = []
       cdata = np.sum(rss, axis=0)

       for i in range(0,len(cdata)):
           lambda_i = lambda0 + i*cdelt
           wave.append(lambda_i)
           if bunit == 'Jy':
             flux_vector = (1e-23*cdata[i]*c_amstrong)/(lambda_i**2)
           elif bunit == 'ELECTRON' or 'CGS' or 'cgs' in bunit:
             flux_vector = cdata[i]
           else:
              print ('unknown or not defined BUNIT [Jy, ELECTRON]')
              sys.exit(1)
           flux.append(flux_vector)

       plt.xlim(lambda0,lambda0+float(len(cdata))*cdelt)

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

       lflines = [lambda0 + (pf1 - crpix) * cdelt, lambda0 + (pf2 - crpix) * cdelt]
       lmlines = [lambda0 + (pm1 - crpix) * cdelt, lambda0 + (pm2 - crpix) * cdelt]
       lllines = [float(args.lcut1)*(1.+z),float(args.lcut2)*(1.+z)]
       lclines = [float(args.ccut1)*(1.+z),float(args.ccut2)*(1.+z)]
       if (args.eccut1 is not None and args.eccut2 is not None):
        leclines = [float(args.eccut1) * (1. + z), float(args.eccut2) * (1. + z)]
       plines = [float(args.pcut1),float(args.pcut2)]
       cwline = [float(args.ctwl)*(1.+z)]
       axvlines(lmlines, color='cyan', label = 'All-fiber range', linestyle = '-')
       axvlines(lflines, color='brown', label = 'Sensitivity range', linestyle = '--')
       axvlines(cwline, color='black', label = 'Central wavelength', linestyle = '-')
       axvlines(lllines, color='gray', label = 'Line-fitting range', linestyle = '-')
       axvlines(lclines, color='gray', label = 'Continuum range', linestyle = '--')
       if (args.eccut1 is not None and args.eccut2 is not None):
        axvlines(leclines, color='gray', linestyle='--')
       axvlines(plines, color='green', label = 'Plot range', linestyle = '-.')
       plt.legend()

       plt.show()

       answer = input('Continue (Y/n)? ')
       if len(answer) is not 0 and answer[0].lower() == "n":
          sys.exit(1)
            
# Reading spectrum/spectra
       with PdfPages(args.output) as pdf:
           FM = []
           CONTINUUM = []
           NOISE = []
           SNR = []
           FLUXD = [] # Flux from window_data - window_continuum
           EWD = [] # Flux from window_data - window_continuum / mean_continuum
           EFLUXD = [] # Error of Flux from window_data - window_continuum
           EEWD = [] # Error of Flux from window_data - window_continuum / mean_continuum
           FLUXF = [] # Flux from best-fitting function
           EWF = [] # EW from best-fitting function
           EFLUXF = [] # Error of Flux from best-fitting function
           EEWF = [] # Error of EW from best-fitting function
           H0 = []
           H1 = []
           H2 = []
           H3 = []
           H4 = []
           H0B = []
           H1B = []
           H2B = []
           H1KS = []
           H2KS = []
           H2KLC = []
           H1KSB = []
           H2KSB = []
           H2KLCB = []     
           FLUXF1 = []
           FLUXF2 = []
           CHI2 = []
           FIB = [] # list of of fibers with SNR above input threshold
           
           for ispec in range(1,nx+1):
              FM.append(args.method)
              plt.figure()
              tbdata = rss[ispec-1,:]
              lambda_fin = lambda0 + (len(tbdata))*cdelt

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
#              wcont = wave[int((float(args.ccut1)*(1.+z)-lambda0)/cdelt+crpix):int((float(args.ccut2)*(1.+z)-lambda0)/cdelt+crpix)]

# Excluding range
              if (args.eccut1 is not None and args.eccut2 is not None):
                  del fcont[int((float(args.eccut1) * (1. + z) - lambda0) / cdelt + crpix) - int(
                      (float(args.ccut1) * (1. + z) - lambda0) / cdelt + crpix):int(
                      (float(args.eccut2) * (1. + z) - lambda0) / cdelt + crpix) - int(
                      (float(args.ccut1) * (1. + z) - lambda0) / cdelt + crpix)]
              wcont = wave[int((float(args.ccut1) * (1. + z) - lambda0) / cdelt + crpix):int(
                  (float(args.ccut2) * (1. + z) - lambda0) / cdelt + crpix)]
              if (args.eccut1 is not None and args.eccut2 is not None):
                  del wcont[int((float(args.eccut1) * (1. + z) - lambda0) / cdelt + crpix) - int(
                      (float(args.ccut1) * (1. + z) - lambda0) / cdelt + crpix):int(
                      (float(args.eccut2) * (1. + z) - lambda0) / cdelt + crpix) - int(
                      (float(args.ccut1) * (1. + z) - lambda0) / cdelt + crpix)]
              cmean = np.mean(fcont)

# Fitting continuum
              p_lin = Parameters()
              p_lin.add('slope', value=zero, vary=True)
              p_lin.add('yord', value=cmean, vary=True)
              if (args.verbose):
                print("FITTING CONTINUUM:")
                print("Input(slope,yord):  %10.3E %10.3E" % (0., cmean))
              err_lin = lambda p, x, y: linfunc(p, x) - y
              fitout_lin = minimize(err_lin, p_lin, args=(wcont, fcont))
              fitted_p_lin = fitout_lin.params
              pars_lin = [fitout_lin.params['slope'].value, fitout_lin.params['yord'].value,fitout_lin.chisqr]
              if (args.verbose):
                  print("Output(slope,yord): %10.3E %10.3E" % (
              fitout_lin.params['slope'].value, fitout_lin.params['yord'].value))
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
              lcmean = np.mean(fit_lin)  # Mean continuum within the line range
              fpline = fline - fit_lin
#              fpline = fline - cmean
              result = np.where(fline == np.amax(fline))
              lpeak = wline[result[0][0]]
            
              CONTINUUM.append(lcmean)
              NOISE.append(rms)
            
              if (np.isnan(peak/rms)):
                  peak=0.
                  rms=1E-31

              eEWd = (rms * cdelt * (cdelt * np.sum(fpline) / lcmean) / (cdelt * np.sum(fpline))) * np.sqrt(
                  2 * len(fpline) + np.sum(fpline) / lcmean + (np.sum(fpline) / lcmean) ** 2 / len(fpline))

              if np.isfinite(peak/rms) and (peak/rms) >= float(args.limsnr) and ispec!=623:
#                  print(ispec)
                  FIB.append(ispec)
                  SNR.append(peak/rms)
                  FLUXD.append(cdelt*np.sum(fpline))
                  EFLUXD.append(rms * cdelt * np.sqrt(2 * len(fpline) + (np.sum(fpline) / lcmean)))
                  EWD.append(cdelt*np.sum(fpline)/lcmean)
                  EEWD.append(eEWd)
              else:
                  SNR.append(nonfit)
                  FLUXD.append(nonfit)
                  EFLUXD.append(nonfit)
                  EWD.append(nonfit)
                  EEWD.append(nonfit)
                  
              if (args.verbose):
                  print("BASIC NUMBERS:")
                  print ("Fiber: %3d; Stat (mean,rms,lpk,pk,S/N): %10.3E %10.3E %5.2f %10.3E %5.2f"%(ispec,lcmean,rms,lpeak,peak,peak/rms))

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
                  p_gh.add('amp',value=amp, vary=True);
                  p_gh.add('center',value=center, vary=True);
                  p_gh.add('sigma',value=sigma, vary=True, min=0.8*sigma);
                  p_gh.add('skew',value=skew,vary=True,min=-0.5,max=0.5);
                  p_gh.add('kurt',value=kurt,vary=True,min=-0.5,max=0.5);
                  if (args.verbose): 
                    print ("FITTING METHOD: GAUSS-HERMITE QUADRATURE")
                    print ("Input(i0,l0,sigma,skew,kurt):  %10.3E %5.2f %5.2f %10.3E %10.3E"%(amp, center, sigma, skew, kurt))
                  gausserr_gh = lambda p,x,y: gaussfunc_gh(p,x)-y 
              if args.method is 1:
                  p_gh=Parameters()
                  p_gh.add('amp',value=amp, vary=True);
                  p_gh.add('center',value=center, vary=True);
                  p_gh.add('sigma',value=1.2*sigma, vary=True, min=sigma);
                  if (args.verbose): 
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
                  if (args.verbose): 
                    print ("FITTING METHOD: DOUBLE GAUSSIAN")
                    print ("Input(i1,l1,sig1,i2,l2,sig2):  %10.3E %5.2f %5.2f %10.3E %5.2f %5.2f"%(amp1, center1, sigma1, args.scale_amp2*amp2, center2, sigma2))
                  gausserr_gh = lambda p,x,y: gauss2func(p,x)-y

              fitout_gh=minimize(gausserr_gh,p_gh,args=(wline,fpline))
              fitted_p_gh = fitout_gh.params

              if args.method is 0:
                  pars_gh=[fitout_gh.params['amp'].value,fitout_gh.params['center'].value,fitout_gh.params['sigma'].value,fitout_gh.params['skew'].value,fitout_gh.params['kurt'].value,fitout_gh.chisqr]
                  fit_gh=gaussfunc_gh(fitted_p_gh,wline)
                  if (args.verbose): 
                    print ("Output(i0,l0,sigma,skew,kurt): %10.3E %5.2f %5.2f %10.3E %10.3E"%(fitout_gh.params['amp'].value, fitout_gh.params['center'].value, fitout_gh.params['sigma'].value, fitout_gh.params['skew'].value, fitout_gh.params['kurt'].value))
                    print('WARNING::::: out.covar == None :::::: Fit Sucess = '+ str(fitout_gh.success) + ' :: Uncertainties estimated = ' + str(fitout_gh.errorbars))

              if args.method is 1:
                  pars_gh=[fitout_gh.params['amp'].value,fitout_gh.params['center'].value,fitout_gh.params['sigma'].value,fitout_gh.chisqr]
                  fit_gh=gaussfunc(fitted_p_gh,wline)
                  if (args.verbose): 
                    print ("Output(i0,l0,sigma): %10.3E %5.2f %5.2f"%(fitout_gh.params['amp'].value, fitout_gh.params['center'].value, fitout_gh.params['sigma'].value))
                    print('WARNING::::: out.covar == None :::::: Fit Sucess = '+ str(fitout_gh.success) + ' :: Uncertainties estimated = ' + str(fitout_gh.errorbars))

              if args.method is 2:
                  pars_gh=[fitout_gh.params['amp1'].value,fitout_gh.params['center1'].value,fitout_gh.params['sigma1'].value,fitout_gh.params['amp2'].value,fitout_gh.params['center2'].value,fitout_gh.params['sigma2'].value,fitout_gh.chisqr] 
                  fit_gh=gauss2func(fitted_p_gh,wline)
                  if (args.verbose): 
                    print ("Output(i1,l1,sig1,i2,l2,sig2): %10.3E %5.2f %5.2f %10.3E %5.2f %5.2f"%(fitout_gh.params['amp1'].value, fitout_gh.params['center1'].value, fitout_gh.params['sigma1'].value, fitout_gh.params['amp2'].value, fitout_gh.params['center2'].value, fitout_gh.params['sigma2'].value))
                  tmp = fitted_p_gh
                  tmp.add('amp', value=tmp['amp1'].value)
                  tmp.add('center', value=tmp['center1'].value)
                  tmp.add('sigma', value=tmp['sigma1'].value)
                  fitted_p_gh1 = tmp
                  fit_gh1 = gaussfunc(fitted_p_gh1, wline)
                  if (args.verbose):
                      print("Flux1 from model: %10.3E+/-%10.3E" % (cdelt * np.sum(fit_gh1), rms * cdelt * np.sqrt(2 * len(fit_gh1) + (np.sum(fit_gh1) / lcmean))))  # Errors as in Tresse et al. (1999)
                  tmp2 = fitted_p_gh
                  tmp2.add('amp', value=tmp2['amp2'].value)
                  tmp2.add('center', value=tmp2['center2'].value)
                  tmp2.add('sigma', value=tmp2['sigma2'].value)
                  fitted_p_gh2 = tmp2
                  fit_gh2 = gaussfunc(fitted_p_gh2, wline)
                  if (args.verbose):
                      print("Flux2 from model: %10.3E+/-%10.3E" % (cdelt * np.sum(fit_gh2), rms * cdelt * np.sqrt(2 * len(fit_gh2) + (np.sum(fit_gh2) / lcmean))))  # Errors as in Tresse et al. (1999)

              eEWm = (rms * cdelt * (cdelt * np.sum(fit_gh) / lcmean) / (cdelt * np.sum(fit_gh))) * np.sqrt(
                  2 * len(fit_gh) + np.sum(fit_gh) / lcmean + (np.sum(fpline) / lcmean) ** 2 / len(fit_gh))
              if (args.verbose):
                  print ("Best-fitting chisqr: %10.3E"%(fitout_gh.chisqr))
                  print("Flux & EW from data:  %10.3E+/-%10.3E %5.2f+/-%5.2f" % (
                  cdelt * np.sum(fpline), rms * cdelt * np.sqrt(2 * len(fpline) + (np.sum(fpline) / lcmean)),
                  cdelt * np.sum(fpline) / lcmean, eEWd))  # Errors as in Tresse et al. (1999)
                  print("Flux & EW from model: %10.3E+/-%10.3E %5.2f+/-%5.2f" % (
                  cdelt * np.sum(fit_gh), rms * cdelt * np.sqrt(2 * len(fit_gh) + (np.sum(fit_gh) / lcmean)),
                  cdelt * np.sum(fit_gh) / lcmean, eEWm))  # Errors as in Tresse et al. (1999)
                  print("Best-fitting chisqr: %10.3E" % (fitout_gh.chisqr))

              # Here we write the results to a file and decide whether we should set them first to zero if S/N is below some threshold
              if (np.isnan(peak/rms)):
                  peak=0.
                  rms=1E-31

              if (np.isfinite(peak/rms)) and (peak/rms) >= float(args.limsnr) and ispec!=623:
                  FLUXF.append(cdelt*np.sum(fit_gh))
                  EFLUXF.append(rms * cdelt * np.sqrt(2 * len(fit_gh)))
                  EWF.append(cdelt*np.sum(fit_gh)/lcmean)
                  EEWF.append(eEWm)

                  plt.plot(wcont, fit_con, 'red', label='Continuum fit')
                  resid_gh=fpline-fit_gh
                  plt.plot(wline, fit_gh+fit_lin, 'orange', label = 'best fit')

                  lflines = [lambda0 + (pf1 - crpix) * cdelt, lambda0 + (pf2 - crpix) * cdelt]
                  lmlines = [lambda0 + (pm1 - crpix) * cdelt, lambda0 + (pm2 - crpix) * cdelt]
                  lllines = [float(args.lcut1)*(1.+z),float(args.lcut2)*(1.+z)]
                  lclines = [float(args.ccut1)*(1.+z),float(args.ccut2)*(1.+z)]
                  if (args.eccut1 is not None and args.eccut2 is not None):
                    leclines = [float(args.eccut1) * (1. + z), float(args.eccut2) * (1. + z)]
                  cwline = [float(args.ctwl)*(1.+z)]

                  axvlines(lmlines, color='cyan', label = 'All-fiber range', linestyle = '-')
                  axvlines(lflines, color='brown', label = 'Sensitivity range', linestyle = '--')
                  if (args.use_peak) == False:
                     axvlines(cwline, color='black', label = 'Central wavelength', linestyle = '-')
                  axvlines(lllines, color='gray', label = 'Line-fitting range', linestyle = '-')
                  axvlines(lclines, color='gray', label = 'Continuum range', linestyle = '--')
                  if (args.eccut1 is not None and args.eccut2 is not None):
                    axvlines(leclines, color='gray', linestyle='--')

                  if args.method is 0:       
                      H0.append(fitout_gh.params['amp'].value)
                      H1.append(fitout_gh.params['center'].value)
                      H2.append(fitout_gh.params['sigma'].value)
                      H3.append(fitout_gh.params['skew'].value)
                      H4.append(fitout_gh.params['kurt'].value)
                      H0B.append(nonfit)
                      H1B.append(nonfit)
                      H2B.append(nonfit)
                      H1KS.append(((fitout_gh.params['center'].value)/float(args.ctwl)-1.0)*c_km)
                      H2KS.append(((fitout_gh.params['sigma'].value)/float(args.ctwl))*c_km)
                      if (np.isfinite(np.sqrt((((fitout_gh.params['sigma'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2))):
                         H2KLC.append(np.sqrt((((fitout_gh.params['sigma'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2))
                      else:
                         H2KLC.append(0.)
                      H1KSB.append(nonfit)
                      H2KSB.append(nonfit)
                      H2KLCB.append(nonfit)    
                      FLUXF1.append(nonfit)
                      FLUXF2.append(nonfit)
                  if args.method is 1:    
                      H0.append(fitout_gh.params['amp'].value)
                      H1.append(fitout_gh.params['center'].value)
                      H2.append(fitout_gh.params['sigma'].value)
                      H3.append(nonfit)
                      H4.append(nonfit)
                      H0B.append(nonfit)
                      H1B.append(nonfit)
                      H2B.append(nonfit)
                      H1KS.append(((fitout_gh.params['center'].value)/float(args.ctwl)-1.0)*c_km)
                      H2KS.append(((fitout_gh.params['sigma'].value)/float(args.ctwl))*c_km)
                      if (np.isfinite(np.sqrt((((fitout_gh.params['sigma'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2))): 
                         H2KLC.append(np.sqrt((((fitout_gh.params['sigma'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2))
                      else:
                         H2KLC.append(0.)
                      H1KSB.append(nonfit)
                      H2KSB.append(nonfit)
                      H2KLCB.append(nonfit)   
                      FLUXF1.append(nonfit)
                      FLUXF2.append(nonfit)
                  if args.method is 2:       
                      H0.append(fitout_gh.params['amp1'].value)
                      H1.append(fitout_gh.params['center1'].value)
                      H2.append(fitout_gh.params['sigma1'].value)
                      H3.append(nonfit)
                      H4.append(nonfit)
                      H0B.append(fitout_gh.params['amp2'].value)
                      H1B.append(fitout_gh.params['center2'].value)
                      H2B.append(fitout_gh.params['sigma2'].value)
                      H1KS.append(((fitout_gh.params['center1'].value)/float(args.ctwl)-1.0)*c_km)
                      H2KS.append(((fitout_gh.params['sigma1'].value)/float(args.ctwl))*c_km)
                      if (np.isfinite(np.sqrt((((fitout_gh.params['sigma1'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2))): 
                         H2KLC.append(np.sqrt((((fitout_gh.params['sigma1'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2))
                      else:
                         H2KLC.append(0.)
                      H1KSB.append(((fitout_gh.params['center2'].value)/float(args.ctwl)-1.0)*c_km)
                      H2KSB.append(((fitout_gh.params['sigma2'].value)/float(args.ctwl))*c_km)
                      if (np.isfinite(np.sqrt((((fitout_gh.params['sigma2'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2))):
                         H2KLCB.append(np.sqrt((((fitout_gh.params['sigma2'].value)/float(args.ctwl))*c_km)**2-(c_km/(R*2.35))**2)) 
                      else:
                         H2KLCB.append(0.)
                      s1 = fitout_gh.params['sigma1'].value
                      s2 = fitout_gh.params['sigma2'].value
                      FLUXF1.append(1.064*fitout_gh.params['amp1'].value*(2.35*s1))
                      FLUXF2.append(1.064*fitout_gh.params['amp2'].value*(2.35*s2))
                  CHI2.append(fitout_gh.chisqr)
              else:
                  FLUXF.append(nonfit)
                  EFLUXF.append(nonfit)
                  EWF.append(nonfit)
                  EEWF.append(nonfit)
                  H0.append(nonfit)
                  H1.append(nonfit)
                  H2.append(nonfit)
                  H3.append(nonfit)
                  H4.append(nonfit)
                  H0B.append(nonfit)
                  H1B.append(nonfit)
                  H2B.append(nonfit)
                  H1KS.append(nonfit)
                  H2KS.append(nonfit)
                  H2KLC.append(nonfit)
                  H1KSB.append(nonfit)
                  H2KSB.append(nonfit)
                  H2KLCB.append(nonfit)
                  FLUXF1.append(nonfit)
                  FLUXF2.append(nonfit)
                  CHI2.append(nonfit)

              plt.title('fiber: '+str(ispec))
              plt.ylabel('flux [erg s$^{-1}$ cm$^{-2}$ A$^{-1}$]')
              plt.xlabel('wavelength')
              plt.legend()
              pdf.savefig()
              plt.close()

# Storing output RSS file
       data1 = ima[0].data
       all_output = np.zeros([data1.shape[0],29])
        
# Assign each property to one of the 29 channels
       all_output[:,0] = FM        #  0 Fitting method (0=gauss-hermite,1=1gauss,2=2gauss)
       all_output[:,1] = CONTINUUM #  1 Continuum level in cgs
       all_output[:,2] = NOISE     #  2 rms in cgs
       all_output[:,3] = SNR       #  3 S/N at the peak of the line
       all_output[:,4] = FLUXD     #  4 Flux from window_data - window_continuum
       all_output[:,5] = EWD       #  5 Flux from window_data - window_continuum / mean_continuum
       all_output[:,6] = FLUXF     #  6 Flux from best-fitting function
       all_output[:,7] = EWF       #  7 EW from best-fitting function
       all_output[:,8] = H0        #  8 amplitude for methods 0 & 1 & 2 (first gaussian)
       all_output[:,9] = H1        #  9 central lambda for methods 0 & 1 & 2 (first gaussian)
       all_output[:,10] = H2       # 10 sigma (in AA) for methods 0 & 1 & 2 (first gaussian)
       all_output[:,11] = H3       # 11 h3 for method 0
       all_output[:,12] = H4       # 12 h4 for method 0
       all_output[:,13] = H0B      # 13 amplitude for method 2 (second gaussian)
       all_output[:,14] = H1B      # 14 central lambda for method 2 (second gaussian)
       all_output[:,15] = H2B      # 15 sigma (in AA) for method 2 (second gaussian)
       all_output[:,16] = H1KS     # 16 velocity in km/s from H1 (1st g)
       all_output[:,17] = H2KS     # 17 sigma in km/s from H2 (1st g)
       all_output[:,18] = H2KLC    # 18 sigma in km/s from H2 corrected for instrumental sigma (1st g)
       all_output[:,19] = H1KSB    # 19 velocity in km/s from H1B (2nd g)
       all_output[:,20] = H2KSB    # 20 sigma in km/s from H2B (2nd g)
       all_output[:,21] = H2KLCB   # 21 sigma in km/s from H2 corrected for instrumental sigma (2nd g)
       all_output[:,22] = FLUXF1   # 22 Flux from best-fitting 1st gaussian function
       all_output[:,23] = FLUXF2   # 23 Flux from best-fitting 2nd gaussian function
       all_output[:,24] = EFLUXD   # 24 Error of 4 (Flux from window_data - window_continuum)
       all_output[:,25] = EEWD     # 25 Error of 5 (Flux from window_data - window_continuum / mean_continuum)
       all_output[:,26] = EFLUXF   # 26 Error of 6 (Flux from best-fitting function)
       all_output[:,27] = EEWF     # 27 Error of 7 (EW from best-fitting function)
       all_output[:,28] = CHI2     # 28 best-fitting chi^2 (cgs)
    
       ima[0].data = all_output
       ima.writeto(args.output_rss, overwrite = True)

# Store the id of the fibers with SNR above minimum signal-to-noise ratio
       if args.output_fibers!=None: 
         np.savetxt(args.output_fibers, FIB, fmt='%d')

if __name__ == '__main__':

    main()

