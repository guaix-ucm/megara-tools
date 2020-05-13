#
# Copyright 2017-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import math
from pathlib import Path
import numpy as np
from astropy.io import fits

import megaradrp.processing.cube as mcube

def trim_cubes(cube_file, trim_numbers):
    hdu = fits.open(cube_file)
    data = hdu[0].data
    for i in range(trim_numbers[0]):
        data = np.delete(data,0,1)   # Corto filas de pixeles de abajo.
    for i in range(trim_numbers[1]):
        data = np.delete(data,-1,1)  # Corto files de pixeles de arriba.
    for i in range(trim_numbers[2]):
        data = np.delete(data,0,2)   # Corto columnas de pixeles de la izquierda.
    for i in range(trim_numbers[3]):
        data = np.delete(data,-1,2)  # Corto columnas de pixeles de la derecha.
    hdu[0].header['NAXIS1'] = hdu[0].header['NAXIS1'] - 2
    hdu[0].header['NAXIS2'] = hdu[0].header['NAXIS2'] - 3
    hdu[0].header['CRPIX1'] = hdu[0].header['CRPIX1'] - 1
    hdu[0].header['CRPIX2'] = hdu[0].header['CRPIX2'] - 1
    hdu[0].data = data
    path = Path(cube_file)
    new_path = path.parent / ('trimmed_' + path.name)
    hdu.writeto(str(new_path), overwrite=True)

def helio_corr(ifile):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u
    hdu = fits.open(ifile)
    h1_0 = hdu[0].header

    obs_date = h1_0['DATE-OBS'].split('T')[0]

    roque = EarthLocation.from_geodetic(lat=28.7606*u.deg, lon=342.1184*u.deg, height=2326*u.m)
    sc = SkyCoord(ra=h1_0['RADEG']*u.deg, dec=h1_0['DECDEG']*u.deg)
    heliocorr = sc.radial_velocity_correction('heliocentric', obstime=Time(obs_date), location=roque)  
    helio_corr_velocity = heliocorr.to(u.km/u.s).value

    return helio_corr_velocity

def hypercube_dimensions(list_cubes, xoff, yoff, target_scale):
    RA_list = []
    DEC_list = []
    for i, ifile in enumerate(list_cubes):
        hdu = fits.open(ifile)
        RA_list.append(hdu[0].header['RADEG'] + (xoff[i]/3600))
        DEC_list.append(hdu[0].header['DECDEG'] + (yoff[i])/3600)
    pixel_size_x = target_scale/np.cos(hdu[0].header['DECDEG']*np.pi/180) # Este es el que aparece al medir en la imagen # arcsec
    pixel_size_y = target_scale # arcsec sin tener en cuenta posicion del telescopio
   
    ref_data = fits.open(list_cubes[0])[0].data
    data_size_x = ref_data.shape[2]
    data_size_y = ref_data.shape[1]  
    data_size_z = ref_data.shape[0]  # Direccion espectral

    hypercube_size_x = int(round((abs(max(RA_list)-min(RA_list))*3600/pixel_size_x) + data_size_x)) # De centro a centro + otro apuntado entero 
    hypercube_size_y = int(round((abs(max(DEC_list)-min(DEC_list))*3600/pixel_size_y) + data_size_y)) # De centro a centro + otro apuntado entero 
    hypercube_size_z = data_size_z
    return hypercube_size_x, hypercube_size_y, hypercube_size_z, RA_list, DEC_list, pixel_size_x, pixel_size_y, data_size_x, data_size_y

def to_bool(s):
    return True if s == '1' else False

def mask_bin(list_cubes, outfile, xoff, yoff, target_scale):
    hypercube_size_x, hypercube_size_y, hypercube_size_z, RA_list, DEC_list, pixel_size_x, pixel_size_y, data_size_x, data_size_y = hypercube_dimensions(list_cubes, xoff, yoff, target_scale)
    hdu = fits.open(list_cubes[0])
    mask = np.zeros((hypercube_size_y,hypercube_size_x), int)
    for i, ifile in enumerate(list_cubes):
        offset_pixels_x = int(round((abs((RA_list[i]-max(RA_list))*3600/pixel_size_x))))
        offset_pixels_y = int(round((abs((DEC_list[i]-min(DEC_list))*3600/pixel_size_y))))        
        for j in range(data_size_y):
            for k in range(data_size_x):
                mask[j+offset_pixels_y,k+offset_pixels_x] += 2**(i) 
    hdu[0].data = mask
    hdu[0].header['NAXIS'] = 2
    hdu[0].header['NAXIS1'] = mask.shape[1]
    hdu[0].header['NAXIS2'] = mask.shape[0]    
    path = Path(outfile)
    new_path = path.parent / (path.stem + '_mask' + path.suffix)
    hdu.writeto(str(new_path), overwrite=True)
#    hdu.writeto(str(outfile + '_hypercube_mask.fits'), overwrite=True)
    boolean_mask = np.empty(mask.shape,dtype=object)
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            b = bin(mask[j,k])[2:].zfill(len(list_cubes))
            b = list(b)
            b = b[::-1]
            a = []
            for i,val in enumerate(b):
                a.append(to_bool(val))
            boolean_mask[j,k] = np.array(a)
    
    return mask, boolean_mask

def rebin_spec(wave, specin, wavnew):
    from pysynphot import observation
    from pysynphot import spectrum

    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
 
    return obs.binflux

def grid_combined_cube(list_cubes, helio_corr_apply, outfile, xoff, yoff, scale_a, scale_m, target_scale):
    from itertools import compress
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u

    hypercube_size_x, hypercube_size_y, hypercube_size_z, RA_list, DEC_list, pixel_size_x, pixel_size_y, data_size_x, data_size_y = hypercube_dimensions(list_cubes, xoff, yoff, target_scale)

    hypercube = np.zeros([hypercube_size_z,hypercube_size_y,hypercube_size_x])
    alldata = np.zeros((hypercube_size_z,hypercube_size_y,hypercube_size_x,len(list_cubes)), float)

    if helio_corr_apply:
    
        for i, ifile in enumerate(list_cubes):
            offset_pixels_x = int(round((abs((RA_list[i]-max(RA_list))*3600/pixel_size_x))))
            offset_pixels_y = int(round((abs((DEC_list[i]-min(DEC_list))*3600/pixel_size_y))))        
            hdu = fits.open(ifile)
            data_ifile = hdu[0].data
            
            h1_0 = hdu[0].header
            lambda_obs = h1_0['CRVAL3'] + h1_0['CDELT3']*np.arange(h1_0['NAXIS3'])
            lambda_em = np.zeros(data_ifile.shape)
            step = h1_0['CDELT3']
            
            helio_corr_velocity = helio_corr(ifile)
            print('Applying heliocentric correction to', ifile, '\n')
            print('Velocity correction:', helio_corr_velocity)
            
            c = 299792.458 # km/s
            
            z = np.exp(helio_corr_velocity/c) - 1
                
            lambda_em = lambda_obs/(1+z)
            
            # Esto es porque necesito saber todas las velocidades a las que voy a 
            # corregir los espectros para tener el rango completo de longitudes de onda. 
            
            lambda_completa = lambda_em.min() + step*np.arange(len(lambda_obs))
            
            spec_helio_corr = np.zeros((len(lambda_completa), data_ifile.shape[1], data_ifile.shape[2]))
            for j in range(data_ifile.shape[1]):
                for k in range(data_ifile.shape[2]):
                    spec_helio_corr[:,j,k] = rebin_spec(lambda_em, data_ifile[:,j,k], lambda_completa)
                    alldata[:,j+offset_pixels_y,k+offset_pixels_x,i] = spec_helio_corr[:,j,k]*scale_m[i] + scale_a[i]

    else:
        print('No heliocentric correction applied')
        for i, ifile in enumerate(list_cubes):
            offset_pixels_x = int(round((abs((RA_list[i]-max(RA_list))*3600/pixel_size_x))))
            offset_pixels_y = int(round((abs((DEC_list[i]-min(DEC_list))*3600/pixel_size_y))))        
            hdu = fits.open(ifile)
            data_ifile = hdu[0].data
            for j in range(data_size_y):
                for k in range(data_size_x):
                    alldata[:,j+offset_pixels_y,k+offset_pixels_x,i] = data_ifile[:,j,k]*scale_m[i] + scale_a[i]

    mask, boolean_mask = mask_bin(list_cubes, outfile, xoff, yoff, target_scale)       
    
    while_loop_count=0
    flux_calibrated_images_index = [0]
    while while_loop_count <= (len(list_cubes)-1):
        for i in range(len(list_cubes)-1):
            if np.isin(i+1,flux_calibrated_images_index) == False:
                npixels_solape = 0
                indice_solape_0 = []
                indice_solape_1 = []
                for m, valm in enumerate(flux_calibrated_images_index):
                    for j in range(boolean_mask.shape[0]):
                        for k in range(boolean_mask.shape[1]):                    
                            if boolean_mask[j,k][i+1] and boolean_mask[j,k][valm]:
                                indice_solape_0.append(j)
                                indice_solape_1.append(k)
                                npixels_solape+=1
                    if npixels_solape >= 10:
                        flux_calibrated_images_index.append(i+1)
                        flux_factor = sum(np.sum(alldata[400:3900,indice_solape_0,indice_solape_1,valm], axis=1))/sum(np.sum(alldata[400:3900,indice_solape_0,indice_solape_1,i+1], axis=1))
                        alldata[:,:,:,i+1] = flux_factor*alldata[:,:,:,i+1]
                        print(flux_factor, list_cubes[i+1])
                        print('Flux calibrated with', list_cubes[valm])
                        break
        while_loop_count+=1
        
    for i, ifile in enumerate(range(len(list_cubes))[::-1]):
        for j in range(hypercube_size_y):
            for k in range(hypercube_size_x):
                if np.mean(alldata[:,j,k,ifile]) > 0:
                    hypercube[:,j,k] = alldata[:,j,k,ifile]

    hdu[0].data = hypercube
    hdu.writeto(str(outfile), overwrite=True)


def main(args=None):
    import argparse
    import os
    import astropy.io.fits as fits

    # parse command-line options
    parser = argparse.ArgumentParser(prog='convert_rss_cube')
    # positional parameters
    methods = {'nn': 1, 'linear': 2}

    parser.add_argument("rss",
                        help="RSS file / List of RSS files",
                        type=argparse.FileType('rb'))
    parser.add_argument('-l', '--is-a-list', default=False, action="store_true", help='Use for -s being a list of FITS spectra')
    parser.add_argument('-c', '--is-a-cube', default=False, action="store_true", help='Use for -s being a list of cubes (not rss) spectra')
    parser.add_argument('-p', '--pixel-size', type=float, default=0.4,
                        metavar='PIXEL_SIZE',
                        help="Pixel size in arc seconds (default = 0.4)")
    parser.add_argument('-o', '--outfile', default='test',
                        help="Name of the output cube file (default = test")
    parser.add_argument('-d', '--disable-scaling', action='store_true',
                        help="Disable flux conservation")
    parser.add_argument('-m', '--method', action='store', choices=['nn', 'linear'],
                        default='nn', help="Method of interpolation")
    parser.add_argument('--wcs-pa-from-header', action='store_true',
                        help="Use PA angle from header", dest='pa_from_header')
    parser.add_argument('-trim', '--trimming', default=False, action="store_true", help='Use for trimming the cubes')
    parser.add_argument('-hyp', '--hyper', default=False, action="store_true", help='Use for creating the hypercube')
    parser.add_argument('-helio', '--helio', default=False, action="store_true", help='Use for applying heliocentric velocity correction')
    parser.add_argument('-trimn', '--trimming-numbers', nargs='*', default= [1,2,1,1], help='Use for declare the number of rows and columns you want to trim. [Bottom rows, top rows, left column, right column] (default= 1,2,1,1)')

    args = parser.parse_args(args=args)

    conserve_flux = not args.disable_scaling
    hexspline_order = methods[args.method]

    xoff = []
    yoff = []

    if args.is_a_list:
        xoff, yoff, scale_a, scale_m = np.loadtxt(args.rss, usecols=(1, 2, 3, 4), unpack=True)
        args.rss.seek(0)
        list_files = args.rss.readlines()
        list = [str(x.strip(),'utf-8') for x in list_files] 
        list_files = []
        for i in list:
            list_files.append(i.split()[0])
    else: 
        list_files = [args.rss.name]
        xoff.append(0.)
        yoff.append(0.)

    target_scale = args.pixel_size # Arcsec
    if not args.is_a_cube:
        print('\n', 'Creating cube from rss', '\n')
        print('target scale is', target_scale, 'arcsec')
        list_cubes = []
        for i in list_files:
#            cout = str(Path(i).parent) + './cube_' + str(Path(i).name)
            path = Path(i)
            new_path = path.parent / ('cube_' + path.name)
            cout = str(new_path)
            list_cubes.append(cout)
            rss = fits.open(i)

            rss_data = rss[0].data
            ############ Assigning an average value to the dead fibre from its six neighbouring fibres ############
            rss[0].data[622] = (rss_data[619] + rss_data[524] + rss_data[528] + rss_data[184] + rss_data[183] + rss_data[
                621]) / 6.0
            #######################################################################################################

            cube = mcube.create_cube_from_rss(rss, hexspline_order, target_scale, conserve_flux=conserve_flux)
            if not args.pa_from_header:
                print('recompute WCS from IPA')
                cube[0].header = mcube.recompute_wcs(cube[0].header)
            cube.writeto(cout,overwrite=True)

    else:
        list_cubes = list_files
        
    if args.trimming:
        print ('\n', 'Trimming cubes...', '\n')
        list_trimmed_cubes = []
        trim_numbers = args.trimming_numbers
        trim_numbers = [ int(x) for x in trim_numbers ]
        print(trim_numbers)
        for i, ifile in enumerate(list_cubes):
            path = Path(ifile)
            new_path = path.parent / ('trimmed_' + path.name)
            list_trimmed_cubes.append(str(new_path))
#            list_trimmed_cubes.append(str(Path(ifile).parent) + './trimmed_' + str(Path(ifile).name))
            trim_cubes(ifile, trim_numbers)
    
    if args.helio:
        helio_corr_apply = True
    else:
        helio_corr_apply = False
        
    if args.hyper:
        outfile = args.outfile
        print ('\n', 'Creating hypercube...', '\n')
        if args.trimming == True:
            grid_combined_cube(list_trimmed_cubes, helio_corr_apply, outfile, xoff, yoff, scale_a, scale_m, target_scale)
        else:
            grid_combined_cube(list_cubes, helio_corr_apply, outfile, xoff, yoff, scale_a, scale_m, target_scale)

if __name__ == '__main__':
    main()
