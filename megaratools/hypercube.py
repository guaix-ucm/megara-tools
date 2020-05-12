#
# Copyright 2017-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""
Interpolation method based on:
'Hex-Splines: A Novel Spline Family for Hexagonal Lattices'
van de Ville et al. IEEE Transactions on Image Processing 2004, 13, 6
"""

import numpy as np
import math
from astropy.io import fits
from scipy import signal
from scipy.interpolate import RectBivariateSpline
from megaradrp.simulation.convolution import hex_c, square_c, setup_grid
from numina.frame.utils import copy_img
from megaradrp.datamodel import MegaraDataModel
import megaradrp.processing.wcs as mwcs

M_SQRT3 = math.sqrt(3)

PLATESCALE = 1.2120  # arcsec / mm
SCALE = 0.443 # mm

HEX_SCALE = PLATESCALE * SCALE

# Normalized hexagon geometry
H_HEX = 0.5
R_HEX = 1 / M_SQRT3
A_HEX = 0.5 * H_HEX * R_HEX
HA_HEX = 6 * A_HEX # detR0 == ha


def my_atleast_2d(*arys):
    """Equivalent to atleast_2d, adding the newaxis at the end"""
    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if len(ary.shape) == 0:
            result = ary.reshape(1, 1)
        elif len(ary.shape) == 1:
            result = ary[:, np.newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def calc_matrix(nrow, ncol, x0, y0, grid_type=2):

    R0 = np.array([[M_SQRT3 / 2,0], [-0.5,1]]) # Unit scale

    kcol = []
    krow = []
    for i in range(ncol):
        if grid_type == 1:
            s = (i + i % 2) // 2 # depending on type
        else:
            s = i // 2
        for j in range(nrow):
            kcol.append(i)
            krow.append(j+s)

    sl = np.array([kcol, krow]) # x y
    r0l = np.dot(R0, sl)
    # r0l = R0 @ sl
    return r0l

def calc_matrix_from_fiberconf(fiberconf, x0, y0):

    # This should be in FIBERCONF...
    spos1_x = []
    spos1_y = []
    for fiber in fiberconf.conected_fibers():
        spos1_x.append(fiber.x)
        spos1_y.append(fiber.y)
    spos1_x = np.asarray(spos1_x)
    spos1_y = np.asarray(spos1_y)

    # FIXME: workaround
    # FIBER in LOW LEFT corner is 614
    REFID = 614
    ref_fiber = fiberconf.fibers[REFID]
    minx, miny = ref_fiber.x, ref_fiber.y
    if ref_fiber.x < -6:
        # arcsec
        ascale = HEX_SCALE # 0.443 * 1.212
        print('fiber coordinates in arcsec')
    else:
        # mm
        ascale = SCALE
        print('fiber coordinates in mm')
#    print (x0, y0)
    refx, refy = minx / ascale, miny / ascale
    rpos1_x = (spos1_x - minx) / ascale
    rpos1_y = (spos1_y - miny) / ascale
    r0l_1 = np.array([rpos1_x, rpos1_y])
    return r0l_1, (refx, refy)


def calc_grid(x0, y0, scale=1.0):

    G_TYPE = 2
    ncol = 27
    nrow = 21
    r0l = calc_matrix(nrow, ncol, x0, y0, grid_type=G_TYPE)
    # r0l = R0 @ sl
    spos_x = scale * (r0l[0] - r0l[0].max() / 2)
    spos_y = scale * (r0l[1] - r0l[1].max() / 2)

    return spos_x, spos_y


def hexgrid_extremes(r0l, x0, y0, target_scale):
    # geometry
    # ha_hex = 6 * a_hex # detR0 == ha
    # compute extremes of hexgrid to rectangular grid
    # with pixel size 'scale'
    x0min, y0min = r0l.min(axis=1)
    x0max, y0max = r0l.max(axis=1)
    y1min = y0min - H_HEX
    y1max = y0max + H_HEX
    x1min = x0min - R_HEX
    x1max = x0max + R_HEX

    j1min = int(math.floor(x1min / target_scale + 0.5))
    i1min = int(math.floor(y1min / target_scale + 0.5))
    j1max = int(math.ceil(x1max / target_scale - 0.5))
    i1max = int(math.ceil(y1max / target_scale - 0.5))
    return (i1min, i1max), (j1min, j1max)



def create_cube(r0l, x0, y0, zval, target_scale=1.0):
    # geometry

    R1 = target_scale * np.array([[1.0 ,0], [0,1]]) # Unit scale
    detR1 = np.linalg.det(R1)

    # compute extremes of hexgrid to rectangular grid
    # with pixel size 'scale'

    (i1min, i1max), (j1min, j1max) = hexgrid_extremes(r0l, x0, y0, target_scale)

    # Rectangular grid
    mk1 = np.arange(i1min, i1max+1)
    mk2 = np.arange(j1min, j1max+1)
    crow = len(mk1)
    ccol = len(mk2)
    # Result image
    # Third axis
    zval2 = my_atleast_2d(zval)
    # disp axis is last axis...
    dk = np.zeros((crow, ccol, zval2.shape[-1]))
    # print('result shape is ', dk.shape)
    # r1k = R1 @ sk
    sk = np.flipud(np.transpose([np.tile(mk1, len(mk2)), np.repeat(mk2, len(mk1))]).T)  # x y
    r1k = np.dot(R1, sk)

    # Compute convolution of hex and rect kernels
    Dx = 0.005
    Dy = 0.005
    xsize = ysize = 3.0
    xx, yy, xs, ys, xl, yl = setup_grid(xsize, ysize, Dx, Dy)

    hex_kernel = hex_c(xx, yy, rad=R_HEX, ang=0.0)
    square_kernel = square_c(xx, yy, target_scale)
    convolved = signal.fftconvolve(hex_kernel, square_kernel, mode='same')
    kernel = convolved *(Dx *Dy)  / (detR1)
    rbs = RectBivariateSpline(xs, ys, kernel)
    # done

    # Loop to compute integrals...
    # This could be faster
    # zval could be 2D
    for s, r in zip(sk.T, r1k.T):
        allpos = -(r0l - r[:, np.newaxis])
        we = np.abs((rbs.ev(allpos[1], allpos[0])))
        we[we<0] = 0.0
        dk[s[1] - i1min, s[0] - j1min] = np.sum(we[:, np.newaxis] * zval2, axis=0)

    return dk

def create_cube_from_array(rss_data, x0, y0, fiberconf, target_scale_arcsec=1.0, conserve_flux=True):

    target_scale = target_scale_arcsec / HEX_SCALE
    conected = fiberconf.conected_fibers()
    rows = [conf.fibid - 1 for conf in conected]

    rss_data = my_atleast_2d(rss_data)

    region = rss_data[rows, :]

    r0l, (refx, refy) = calc_matrix_from_fiberconf(fiberconf, x0, y0)
    cube_data = create_cube(r0l, x0, y0, region[:, :], target_scale)
    # scale with areas
    if conserve_flux:
        cube_data *= (target_scale ** 2 / HA_HEX)
    result = np.moveaxis(cube_data, 2, 0)
    result.astype('float32')
    return result


def create_cube_from_rss(rss,  x0, y0, target_scale_arcsec=1.0, conserve_flux=True):

    target_scale = target_scale_arcsec / HEX_SCALE
    # print('target scale is', target_scale)

    rss_data = rss[0].data
    
    ############ Assigning an average value to the dead fibre from its six neighbouring fibres ############
    rss_data[622] = (rss_data[619]+rss_data[524]+rss_data[528]+rss_data[184]+rss_data[183]+rss_data[621])/6
    #######################################################################################################

    # Operate on non-SKY fibers

    datamodel = MegaraDataModel()

    fiberconf = datamodel.get_fiberconf(rss)
    conected = fiberconf.conected_fibers()
    rows = [conf.fibid - 1 for conf in conected]
    #
    region = rss_data[rows, :]

    # FIXME: workaround
    # Get FUNIT keyword
    hdr = rss['FIBERS'].header
    funit = hdr.get('FUNIT', 'arcsec')
    pscale = hdr.get('PSCALE', PLATESCALE)
    if funit == 'mm':
        print('fiber coordinates in mm')
        coord_scale = pscale
    else:
        print('fiber coordinates in arcsec')
        coord_scale = 1.0
    # print('PSCALE', pscale)

    # The scale can be 'mm' or 'arcsec'  and it should come from the header
    r0l, (refx, refy) = calc_matrix_from_fiberconf(fiberconf, x0, y0)

    (i1min, i1max), (j1min, j1max) = hexgrid_extremes(r0l, x0, y0, target_scale)
    cube_data = create_cube(r0l, x0, y0, region[:, :], target_scale)

    if conserve_flux:
        # scale with areas
        cube_data *= (target_scale ** 2 / HA_HEX)

    cube = copy_img(rss)
    # Move axis to put WL first
    # so that is last in FITS
    # plt.imshow(cube_data[:, :, 0], origin='lower', interpolation='bicubic')
    # plt.show()

    cube[0].data = np.moveaxis(cube_data, 2, 0)
    cube[0].data.astype('float32')

    # Merge headers
    merge_wcs(rss['FIBERS'].header, rss[0].header, out=cube[0].header)
    # Update values of WCS
    # CRPIX1, CRPIX2
    # CDELT1, CDELT2
    # minx, miny
    # After shifting the array
    # refpixel is -i1min, -j1min
    crpix_x = -refx / target_scale - j1min
    crpix_y = -refy / target_scale - i1min
    # Map the center of original field
    #
    #
    cube[0].header['CRPIX1'] = crpix_x
    cube[0].header['CRPIX2'] = crpix_y
    cube[0].header['CDELT1'] = -target_scale_arcsec / (3600.0)
    cube[0].header['CDELT2'] = target_scale_arcsec / (3600.0)
    # 2D from FIBERS
    # WL from PRIMARY
    # done
    return cube

def recompute_wcs(hdr):
    """Recompute the WCS rotations from IPA """
    ipa = hdr['IPA']
    pa = mwcs.compute_pa_from_ipa(ipa)
    print('IPA angle is:', ipa, 'PA angle is', math.fmod(pa, 360))
    x = hdr['PC1_1']
    y = hdr['PC1_2']
    print('PA from header is:', np.rad2deg(math.atan2(y, x)))
    return mwcs.update_wcs_from_ipa(hdr, pa)


def merge_wcs(hdr_sky, hdr_spec, out=None):

    if out is None:
        hdr = hdr_spec.copy()
    else:
        hdr = out

    # Extend header for third axis
    c_crpix = 'Pixel coordinate of reference point'
    c_cunit = 'Units of coordinate increment and value'
    hdr.set('CUNIT1', comment=c_cunit, after='CDELT1')
    hdr.set('CUNIT2', comment=c_cunit, after='CUNIT1')
    hdr.set('CUNIT3', value=' ',comment=c_cunit, after='CUNIT2')
    hdr.set('CRPIX2', value=1, comment=c_crpix, after='CRPIX1')
    hdr.set('CRPIX3', value=1, comment=c_crpix, after='CRPIX2')
    hdr.set('CDELT3', after='CDELT2')
    hdr.set('CTYPE3', after='CTYPE2')
    hdr.set('CRVAL3', after='CRVAL2')
    c_pc = 'Coordinate transformation matrix element'
    hdr.set('PC1_1', value=1.0, comment=c_pc, after='CRVAL3')
    hdr.set('PC1_2', value=0.0, comment=c_pc, after='PC1_1')
    hdr.set('PC1_3', value=0.0, comment=c_pc, after='PC1_2')
    hdr.set('PC2_1', value=0.0, comment=c_pc, after='PC1_3')
    hdr.set('PC2_2', value=1.0, comment=c_pc, after='PC2_1')
    hdr.set('PC2_3', value=0.0, comment=c_pc, after='PC2_2')
    hdr.set('PC3_1', value=0.0, comment=c_pc, after='PC2_3')
    hdr.set('PC3_2', value=0.0, comment=c_pc, after='PC3_1')
    hdr.set('PC3_3', value=1.0, comment=c_pc, after='PC3_2')

    # Mapping, which keyword comes from each header
    mappings = [('CRPIX3', 'CRPIX1', 0, 0.0),
                ('CDELT3', 'CDELT1', 0, 1.0),
                ('CRVAL3', 'CRVAL1', 0, 0.0),
                ('CTYPE3', 'CTYPE1', 0, ' '),
                ('CRPIX1', 'CRPIX1', 1, 0.0),
                ('CDELT1', 'CDELT1', 1, 1.0),
                ('CRVAL1', 'CRVAL1', 1, 0.0),
                ('CTYPE1', 'CTYPE1', 1, ' '),
                ('CUNIT1', 'CUNIT1', 1, ' '),
                ('PC1_1', 'PC1_1', 1 , 1.0),
                ('PC1_2', 'PC1_2', 1 , 0.0),
                ('CRPIX2', 'CRPIX2', 1, 0.0),
                ('CDELT2', 'CDELT2', 1, 1.0),
                ('CRVAL2', 'CRVAL2', 1, 0.0),
                ('CTYPE2', 'CTYPE2', 1, ' '),
                ('CUNIT2', 'CUNIT2', 1, ' '),
                ('PC2_1', 'PC2_1', 1, 0.0),
                ('PC2_2', 'PC2_2', 1, 1.0),
                ]

    idem_keys = [
        ('LONPOLE', 0.0),
    #    'LATPOLE',
        ('RADESYS', 'FK5')
    #    'EQUINOX'
    ]
    for key, default in idem_keys:
        mp = (key, key, 1, default)
        mappings.append(mp)

    hdr_in = {}
    hdr_in[0] = hdr_spec
    hdr_in[1] = hdr_sky

    for dest, orig, idx, default in mappings:
        hdr_orig = hdr_in[idx]
        hdr[dest] = (hdr_orig.get(orig, default), hdr_orig.comments[orig])

    return hdr



#def combine_cube(list_cubes,stattype):
#    import astropy.io.fits as fits
#    import argparse
#    import numpy as np
#        
##    parser = argparse.ArgumentParser(description='Combining by averaging aligned RSS files',prog='combine_rss')
##    parser.add_argument("rss",help="Input table with list of RSS files",type=argparse.FileType('rb'))
##    parser.add_argument('-o', '--output', default='combined_rss.fits', metavar='OUTPUT RSS', help='Output RSS', type=argparse.FileType('w'))
##    parser.add_argument('-t', '--stattype', default=0, choices=[0,1], metavar='COMBINATION TYPE', help='Type of combination (mean=0, median=1)', type=int)
##    args = parser.parse_args(args=args)
#
#    refima = fits.open(list_cubes[0])
#    nx = refima[0].header['NAXIS1']
#    ny = refima[0].header['NAXIS2'] 
#    nz = refima[0].header['NAXIS3'] 
#    
#    alldata = np.zeros((nz,ny,nx,len(list_cubes)), float)
#    
#    for ifile in list_cubes:
#        hdu = fits.open(ifile)
#        alldata[:,:,:,list_cubes.index(ifile)] = hdu[0].data
#       
#    if (stattype) is 0:
#       avgdata = np.mean(alldata, axis=3)
#    elif (stattype) is 1:
#       avgdata = np.median(alldata, axis=3)
#    else:
#       avgdata = np.sum(alldata, axis=3)
#
#    refima[0].data = avgdata
#    return refima
#    refima.writeto(args.output.name, overwrite = True)

def trim_cubes(cube_file, trim_numbers):
    hdu = fits.open(cube_file)
    data = hdu[0].data
    for i in range(trim_numbers[0]):
        data = np.delete(data,0,1)   # Corto líneas de píxeles de abajo.
    for i in range(trim_numbers[1]):
        data = np.delete(data,-1,1)  # Corto líneas de píxeles de arriba.
    for i in range(trim_numbers[2]):
        data = np.delete(data,0,2)   # Corto líneas de píxeles de la izquierda.
    for i in range(trim_numbers[3]):
        data = np.delete(data,-1,2)  # Corto líneas de píxeles de la derecha.
    hdu[0].header['NAXIS1'] = hdu[0].header['NAXIS1'] - 2
    hdu[0].header['NAXIS2'] = hdu[0].header['NAXIS2'] - 3
    hdu[0].header['CRPIX1'] = hdu[0].header['CRPIX1'] - 1
    hdu[0].header['CRPIX2'] = hdu[0].header['CRPIX2'] - 1
    hdu[0].data = data
    hdu.writeto('trimmed_' + cube_file, overwrite=True)

def helio_corr(ifile):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u
    hdu = fits.open(ifile)
    h1_0 = hdu[0].header

    obs_date = h1_0['DATE-OBS'].split('T')[0]

#    roque = EarthLocation.of_site('Roque de los Muchachos')
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
    pixel_size_y = target_scale # arcsec sin tener en cuenta posición del telescopio
   
    ref_data = fits.open(list_cubes[0])[0].data
    data_size_x = ref_data.shape[2]
    data_size_y = ref_data.shape[1]  
    data_size_z = ref_data.shape[0]  # Dirección espectral

### Con las dos líneas siguientes calculo el tamaño del hipercubo ###
    
    hypercube_size_x = int(round((abs(max(RA_list)-min(RA_list))*3600/pixel_size_x) + data_size_x)) # De centro a centro + otro apuntado entero 
    hypercube_size_y = int(round((abs(max(DEC_list)-min(DEC_list))*3600/pixel_size_y) + data_size_y)) # De centro a centro + otro apuntado entero 
    hypercube_size_z = data_size_z
    # Devuelve 7 cosas #
    return hypercube_size_x, hypercube_size_y, hypercube_size_z, RA_list, DEC_list, pixel_size_x, pixel_size_y, data_size_x, data_size_y


## Con esto creo la máscara ##
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
    hdu.writeto(str(outfile + '_hypercube_mask.fits'), overwrite=True)
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
### Función de rebineado ###

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
#    from sklearn.neighbors import NearestNeighbors
    
    # list_cubes = ['cube_reduced_rss_OB0001_B.fits','cube_reduced_rss_OB0002_B.fits','cube_reduced_rss_OB0003_B.fits','cube_reduced_rss_OB0004_B.fits']
    # offsets_in_list = np.loadtxt('lista_LR-B', usecols=(1,2,3,4))
    # xoff = offsets_in_list[:,0]
    # yoff = offsets_in_list[:,1]  # Estas lineas solo son necesarias si no lo corro desde terminal.
    # scale_a = offsets_in_list[:,2]
    # scale_m = offsets_in_list[:,3]    
    # target_scale= 0.45
    # outfile= 'dummy'

    hypercube_size_x, hypercube_size_y, hypercube_size_z, RA_list, DEC_list, pixel_size_x, pixel_size_y, data_size_x, data_size_y = hypercube_dimensions(list_cubes, xoff, yoff, target_scale)


    hypercube = np.zeros([hypercube_size_z,hypercube_size_y,hypercube_size_x])
    alldata = np.zeros((hypercube_size_z,hypercube_size_y,hypercube_size_x,len(list_cubes)), float)

    if helio_corr_apply:
    
        for i, ifile in enumerate(list_cubes):
            offset_pixels_x = int(round((abs((RA_list[i]-max(RA_list))*3600/pixel_size_x))))
            offset_pixels_y = int(round((abs((DEC_list[i]-min(DEC_list))*3600/pixel_size_y))))        
            hdu = fits.open(ifile)
            data_ifile = hdu[0].data
    ##################################################################################################################
            
            h1_0 = hdu[0].header
            lambda_obs = h1_0['CRVAL3'] + h1_0['CDELT3']*np.arange(h1_0['NAXIS3'])
            lambda_em = np.zeros(data_ifile.shape)
            step = h1_0['CDELT3']
            
            
            helio_corr_velocity = helio_corr(ifile)
            print('Applying heliocentric correction to', ifile, '\n')
            print('Velocity correction:', helio_corr_velocity)
            ### A partir de aquí aplico la corrección de velocidad heliocéntrica a los espectros. ###
            
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

##################################################################################################################
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
    #                        break
    #                    break
        while_loop_count+=1
        
    for i, ifile in enumerate(range(len(list_cubes))[::-1]):
        # print(ifile)
        for j in range(hypercube_size_y):
            for k in range(hypercube_size_x):
                if np.mean(alldata[:,j,k,ifile]) > 0:
                    hypercube[:,j,k] = alldata[:,j,k,ifile]

    hdu[0].data = hypercube
    hdu.writeto(str(outfile + '_hypercube.fits'), overwrite=True)
    


 ############################################################################################################


def main(args=None):
    import argparse
    import os
    import astropy.io.fits as fits

    # parse command-line options
    parser = argparse.ArgumentParser(prog='convert_rss_cube')
    # positional parameters

    parser.add_argument("rss",
                        help="RSS file / List of RSS files",
                        type=argparse.FileType('rb'))
    parser.add_argument('-l', '--is-a-list', default=False, action="store_true", help='Use for -s being a list of FITS spectra')
    parser.add_argument('-c', '--is-a-cube', default=False, action="store_true", help='Use for -s being a list of cubes (not rss) spectra')
    parser.add_argument('-p', '--pixel-size', type=float, default=0.4,
                        metavar='PIXEL_SIZE',
                        help="Pixel size in arc seconds (default = 0.4)")
#    parser.add_argument('-t', '--stattype', default=0, choices=[0,1,2], metavar='COMBINATION TYPE', help='Type of combination (mean=0, median=1, sum=2)', type=int)
    parser.add_argument('-o', '--outfile', default='test',
                        help="Name of the output cube file (default = test")
    parser.add_argument('-d', '--disable-scaling', action='store_true',
                        help="Disable flux conservation")
    parser.add_argument('--wcs-pa-from-header', action='store_true',
                        help="Use PA angle from header", dest='pa_from_header')
    parser.add_argument('-trim', '--trimming', default=False, action="store_true", help='Use for trimming the cubes')
    # parser.add_argument('-comb', '--combine', default=False, action="store_true", help='Use for -s being a list of FITS spectra')
    parser.add_argument('-hyp', '--hyper', default=False, action="store_true", help='Use for creating the hypercube')
    parser.add_argument('-helio', '--helio', default=False, action="store_true", help='Use for applying heliocentric velocity correction')
    parser.add_argument('-trimn', '--trimming-numbers', nargs='*', default= [1,2,1,1], help='Use for declare the number of rows and columns you want to trim. [Bottom rows, top rows, left column, right column] (default= 1,2,1,1)')

    args = parser.parse_args(args=args)

    
    conserve_flux = not args.disable_scaling
    
    xoff = []
    yoff = []

    if args.is_a_list == True:
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
    if args.is_a_cube == False:
        print('\n', 'Creating cube from rss', '\n')
        print('target scale is', target_scale, 'arcsec')
        list_cubes = []
        for i in list_files:
            cout = 'cube_' + i
            list_cubes.append(cout)
            rss = fits.open(i)
            cube = create_cube_from_rss(rss, xoff[list_files.index(i)], yoff[list_files.index(i)], target_scale, conserve_flux=conserve_flux)
            if not args.pa_from_header:
                print('recompute WCS from IPA')
                cube[0].header = recompute_wcs(cube[0].header)
            cube.writeto(cout,overwrite=True)

    else:
        list_cubes = list_files
        
    # if args.combine == True:     
    #     print ('\n', 'Calling combine_cube...', '\n',)
#        combine_cube(list_cubes,args.stattype).writeto(str(args.outfile + '.fits'), overwrite=True)
    
    if args.trimming == True:
        print ('\n', 'Trimming cubes...', '\n')
        list_trimmed_cubes = []
        trim_numbers = args.trimming_numbers
        trim_numbers = [ int(x) for x in trim_numbers ]
        print(trim_numbers)
        for i, ifile in enumerate(list_cubes):
            list_trimmed_cubes.append('trimmed_' + ifile)
            trim_cubes(ifile, trim_numbers)
    
    if args.helio == True:
        helio_corr_apply = True
    else:
        helio_corr_apply = False
        
    if args.hyper == True:
        outfile = args.outfile
        print ('\n', 'Creating hypercube...', '\n')
        if args.trimming == True:
            grid_combined_cube(list_trimmed_cubes, helio_corr_apply, outfile, xoff, yoff, scale_a, scale_m, target_scale)
        else:
            grid_combined_cube(list_cubes, helio_corr_apply, outfile, xoff, yoff, scale_a, scale_m, target_scale)

if __name__ == '__main__':
#    mypars = ['final_rss.list','-l','-t','2','-p','0.4']
#    main(mypars)
    main()
