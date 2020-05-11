#
# Copyright 2019-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


def main(args=None):
    import shapely.geometry as sg
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import numpy as np
    import shapely.affinity
    from astropy.io import fits
    import argparse
    import sys
    
# Parser
    parser = argparse.ArgumentParser(description='Extract spectra based on elliptical rings',prog='extract_elliptical_rings_spectrum')
    parser.add_argument('-r', '--rss', metavar='RSS-SPECTRUM', help='RSS FITS spectrum', type=argparse.FileType('rb'))
    parser.add_argument('-a', '--accumulate', default=False, action="store_true")
    parser.add_argument('-b', '--surface_brightness', default=False, action="store_true")
    parser.add_argument('-c', '--central-fiber', metavar='CENTRAL-FIBER', default=310, help='Central fiber', type=int) 
    parser.add_argument('-n', '--number-rings', metavar='NUMBER-RINGS', help='Number of rings', type=int) 
    parser.add_argument('-w', '--width', metavar='RINGS WIDTH', help='Elliptical rings width (arcsec)',type=float)
    parser.add_argument('-s', '--saved-rss', metavar='SAVED-RSS', help='Output RSS file', type=argparse.FileType('wb'))
    parser.add_argument('-e', '--ellipticity', metavar='ELLIPTICITY', help='Elliptical rings ellipticity',type=float)
    parser.add_argument('-pa', '--position-angle', metavar='POSITION ANGLE', help='Elliptical rings position angle (N->E)',type=float)
    parser.add_argument('-v', '--verbose', default=False, action="store_true")
    args = parser.parse_args(args=args)
  
    meg_spectra = args.rss
    fibra_central = args.central_fiber
    rings_number = args.number_rings

    plt.rcParams.update({'figure.max_open_warning': 0})

    hdu = fits.open(meg_spectra)
    gal_lin = hdu[0].data   ### Flux en Jy ###
    h1_0 = hdu[0].header
    locs_mm = hdu[1].header  ## en mm ##
    lam_gal = h1_0['CRVAL1'] + h1_0['CDELT1']*np.arange(h1_0['NAXIS1'])

    n=0
    b = np.zeros([gal_lin.shape[0],2])
    for i in range(gal_lin.shape[0]):
        if i < 9:
            b[n,0] = locs_mm[str('FIB00' + str(i+1) + '_X')]*1.212  ## El factor 1.212 es para pasar de mm a arcsec ##
            b[n,1] = locs_mm[str('FIB00' + str(i+1) + '_Y')]*1.212
        if 9 <= i < 99:
            b[n,0] = locs_mm[str('FIB0' + str(i+1) + '_X')]*1.212
            b[n,1] = locs_mm[str('FIB0' + str(i+1) + '_Y')]*1.212
        if 99 <= i < 999:
            b[n,0] = locs_mm[str('FIB' + str(i+1) + '_X')]*1.212
            b[n,1] = locs_mm[str('FIB' + str(i+1) + '_Y')]*1.212
        n+=1
    ell_x_offset = b[fibra_central-1][0]
    ell_y_offset = b[fibra_central-1][1]

    flux = 0
    if (args.surface_brightness):
        area_megara_spaxel = 3*np.sqrt(3)*(0.62/2)
    else:
        area_megara_spaxel = 1

    rings_data = np.zeros((rings_number,gal_lin.shape[1]))
    stacked_spectrum = np.zeros((gal_lin.shape[1]))
    total_area = 0
    
    amp_int=[]
    amp_ext=[]
    for i in range(rings_number):
        amp_int.append(float(args.width)*i)
        amp_ext.append(args.width+(args.width*i))
        
    fig = plt.figure(figsize=(8,8))
    for i in range(rings_number):
        for ind, val in enumerate(b):
            area = intersection(x_offset=val[0], y_offset=val[1],a_int=amp_int[i],a_ext=amp_ext[i],ellipticity=args.ellipticity,angle=90+args.position_angle,ell_x_offset=ell_x_offset,ell_y_offset=ell_y_offset)
            stacked_spectrum += gal_lin[ind]*area
            total_area += area
        rings_data[i] = stacked_spectrum/(total_area*area_megara_spaxel)
        if (args.verbose):
            print('Ring #', i+1, ': ', rings_data[i][2150], ' Jy/[asec/spx]^2 (@CWL) - area/rad: ',total_area*area_megara_spaxel,'/',(amp_ext[i]+amp_int[i])/2.,' [asec/spx]^2/asec)',sep="")
        if (args.accumulate is False):
            stacked_spectrum = np.zeros((gal_lin.shape[1]))
            total_area = 0
    hdu[0].data = rings_data
    hdu.writeto(args.saved_rss, overwrite = True)


def intersection(x_offset,y_offset,ell_x_offset,ell_y_offset,angle,a_ext,a_int,ellipticity):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import numpy as np
    from astropy.io import fits
    import descartes #it has to be installed
    import shapely.geometry as sg #it has to be installed
    import shapely.affinity #it has to be installed

    lado = 0.62/2
    vertices = [[x_offset+lado* 1/2,y_offset+lado*(np.sqrt(3)/2)],[x_offset+lado*1,y_offset+0],[x_offset+lado*1/2,y_offset-lado*(np.sqrt(3)/2)],
                           [x_offset-lado*1/2,y_offset-lado*(np.sqrt(3)/2)],[x_offset-lado*1,y_offset+0],[x_offset-lado*1/2,y_offset+(lado*np.sqrt(3)/2)]]
    hexagon = sg.Polygon(vertices)
    
    #### CENTRO DE LA ELIPSE ####
    centro = (ell_x_offset,ell_y_offset)
    
    ### ELIPSE exterior ###
    b_ext = a_ext*(1-ellipticity)
    ellipse_ext = (centro,(a_ext, b_ext),angle)
    circ_ext = sg.Point(ellipse_ext[0]).buffer(1)
    ell_ext  = shapely.affinity.scale(circ_ext, float(ellipse_ext[1][0]), float(ellipse_ext[1][1]))
    ellr_ext = shapely.affinity.rotate(ell_ext,ellipse_ext[2])
    interseccion_ext = hexagon.intersection(ellr_ext)
    inter_area_ext = interseccion_ext.area
    inter_area_ext_norm = inter_area_ext/hexagon.area
    
    ### ELIPSE interior ###
    b_int = a_int*(1-ellipticity)  #### ELIPTICIDAD CONSTANTE ####
    ellipse_int = (centro,(a_int, b_int),angle)
    circ_int = sg.Point(ellipse_int[0]).buffer(1)
    ell_int  = shapely.affinity.scale(circ_int, float(ellipse_int[1][0]), float(ellipse_int[1][1]))
    ellr_int = shapely.affinity.rotate(ell_int,ellipse_int[2])
    interseccion_int = hexagon.intersection(ellr_int)
    inter_area_int = interseccion_int.area
    inter_area_int_norm = inter_area_int/hexagon.area
    
    # use descartes to create the matplotlib patches
    ax = plt.gca()
    ax.add_patch(descartes.PolygonPatch(hexagon, fc='b', ec='w', alpha=0.2))
    ax.add_patch(descartes.PolygonPatch(ellr_ext.difference(ellr_int), fc='None', ec='k', alpha=0.05, zorder = 100))
    ax.add_patch(descartes.PolygonPatch(ellr_ext.difference(ellr_int), fc='r', ec='k', alpha=0.005, zorder = 100))    
    x_min = -6.2
    x_max = 6.2
    y_min = -6.2
    y_max = 6.2
    ax.set_xlim(x_min - abs(0.1*x_min), x_max + abs(0.1*x_max))
    ax.set_ylim(y_min - abs(0.1*y_min), y_max + abs(0.1*y_max))    
    ax.xaxis.set_tick_params(length = 5, width=1,labelsize=12)
    ax.yaxis.set_tick_params(length = 5, width=1,labelsize=12)
    plt.rc('xtick', direction = 'in')             
    plt.xlabel('[arcsec]', fontsize = 12)
    plt.ylabel('[arcsec]', fontsize = 12)  
    plt.setp(ax.spines.values(), linewidth=1,zorder=100)
    plt.subplots_adjust(left =0.1, bottom =0.2, right =0.9, top =0.99)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_aspect('equal')    
    return inter_area_ext_norm-inter_area_int_norm

if __name__ == '__main__':
    main()

