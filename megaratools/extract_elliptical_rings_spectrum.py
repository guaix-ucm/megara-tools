#
# Copyright 2019-2024 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
import shapely.geometry as sg
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import numpy as np
import shapely.affinity
from astropy.io import fits
import argparse


def main(args=None):

    # Parser
    parser = argparse.ArgumentParser(
        description="Extract spectra based on elliptical rings",
        prog="extract_elliptical_rings_spectrum",
    )
    parser.add_argument(
        "-r",
        "--rss",
        metavar="RSS-SPECTRUM",
        help="RSS FITS spectrum",
        type=argparse.FileType("rb"),
    )
    parser.add_argument("-a", "--accumulate", default=False, action="store_true")
    parser.add_argument(
        "-b", "--surface_brightness", default=False, action="store_true"
    )
    parser.add_argument(
        "-c",
        "--central-fiber",
        metavar="CENTRAL-FIBER",
        default=310,
        help="Central fiber",
        type=int,
    )
    parser.add_argument(
        "-n", "--number-rings", metavar="NUMBER-RINGS", help="Number of rings", type=int
    )
    parser.add_argument(
        "-w",
        "--width",
        metavar="RINGS WIDTH",
        help="Elliptical rings width (arcsec)",
        type=float,
    )
    parser.add_argument(
        "-s",
        "--saved-rss",
        metavar="SAVED-RSS",
        help="Output RSS file",
        type=argparse.FileType("wb"),
    )
    parser.add_argument(
        "-e",
        "--ellipticity",
        metavar="ELLIPTICITY",
        help="Elliptical rings ellipticity",
        type=float,
    )
    parser.add_argument(
        "-pa",
        "--position-angle",
        metavar="POSITION ANGLE",
        help="Elliptical rings position angle (N->E)",
        type=float,
    )
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("-sp", "--save-plots", default=False, action="store_true")
    args = parser.parse_args(args=args)

    meg_spectra = args.rss
    fibra_central = args.central_fiber
    rings_number = args.number_rings

    plt.rcParams.update({"figure.max_open_warning": 0})

    hdu = fits.open(meg_spectra)
    gal_lin = hdu[0].data  # Flux en Jy
    h1_0 = hdu[0].header
    locs_mm = hdu[1].header  # en mm
    lam_gal = h1_0["CRVAL1"] + h1_0["CDELT1"] * np.arange(h1_0["NAXIS1"])

    n = 0
    b = np.zeros([gal_lin.shape[0], 2])
    for i in range(gal_lin.shape[0]):
        if i < 9:
            b[n, 0] = (
                locs_mm[str("FIB00" + str(i + 1) + "_X")] * 1.212
            )  # El factor 1.212 es para pasar de mm a arcsec ##
            b[n, 1] = locs_mm[str("FIB00" + str(i + 1) + "_Y")] * 1.212
        if 9 <= i < 99:
            b[n, 0] = locs_mm[str("FIB0" + str(i + 1) + "_X")] * 1.212
            b[n, 1] = locs_mm[str("FIB0" + str(i + 1) + "_Y")] * 1.212
        if 99 <= i < 999:
            b[n, 0] = locs_mm[str("FIB" + str(i + 1) + "_X")] * 1.212
            b[n, 1] = locs_mm[str("FIB" + str(i + 1) + "_Y")] * 1.212
        n += 1
    ell_x_offset = b[fibra_central - 1][0]
    ell_y_offset = b[fibra_central - 1][1]

    flux = 0
    if args.surface_brightness:
        area_megara_spaxel = 3 * np.sqrt(3) * (0.62 / 2)
    else:
        area_megara_spaxel = 1

    rings_data = np.zeros((rings_number, gal_lin.shape[1]))
    stacked_spectrum = np.zeros((gal_lin.shape[1]))
    total_area = 0

    amp_int = []
    amp_ext = []
    for i in range(rings_number):
        amp_int.append(float(args.width) * i)
        amp_ext.append(args.width + (args.width * i))

    for i in range(rings_number):
        for ind, val in enumerate(b):
            area = intersection(
                x_offset=val[0],
                y_offset=val[1],
                a_int=amp_int[i],
                a_ext=amp_ext[i],
                ellipticity=args.ellipticity,
                angle=90 + args.position_angle,
                ell_x_offset=ell_x_offset,
                ell_y_offset=ell_y_offset,
            )
            stacked_spectrum += gal_lin[ind] * area
            total_area += area
        rings_data[i] = stacked_spectrum / (total_area * area_megara_spaxel)
        if args.verbose:
            print(
                "Ring #",
                i + 1,
                ": ",
                rings_data[i][2150],
                " Jy/[asec/spx]^2 (@CWL) - area/rad: ",
                total_area * area_megara_spaxel,
                "/",
                (amp_ext[i] + amp_int[i]) / 2.0,
                " [asec/spx]^2/asec)",
                sep="",
            )
        if args.accumulate is False:
            stacked_spectrum = np.zeros((gal_lin.shape[1]))
            total_area = 0
        # Llamada a la función para generar y guardar la gráfica del anillo
        if args.save_plots:
            plot_ring(
                b,
                ell_x_offset,
                ell_y_offset,
                90 + args.position_angle,
                amp_ext[i],
                amp_int[i],
                args.ellipticity,
            )
    hdu[0].data = rings_data
    hdu.writeto(args.saved_rss, overwrite=True)


def intersection(
    x_offset, y_offset, ell_x_offset, ell_y_offset, angle, a_ext, a_int, ellipticity
):
    lado = 0.62 / 2
    vertices = [
        [x_offset + lado * 1 / 2, y_offset + lado * (np.sqrt(3) / 2)],
        [x_offset + lado * 1, y_offset + 0],
        [x_offset + lado * 1 / 2, y_offset - lado * (np.sqrt(3) / 2)],
        [x_offset - lado * 1 / 2, y_offset - lado * (np.sqrt(3) / 2)],
        [x_offset - lado * 1, y_offset + 0],
        [x_offset - lado * 1 / 2, y_offset + (lado * np.sqrt(3) / 2)],
    ]
    hexagon = sg.Polygon(vertices)

    # CENTRO DE LA ELIPSE ####
    centro = (ell_x_offset, ell_y_offset)

    # ELIPSE exterior ###
    b_ext = a_ext * (1 - ellipticity)
    ellipse_ext = (centro, (a_ext, b_ext), angle)
    circ_ext = sg.Point(ellipse_ext[0]).buffer(1)
    ell_ext = shapely.affinity.scale(
        circ_ext, float(ellipse_ext[1][0]), float(ellipse_ext[1][1])
    )
    ellr_ext = shapely.affinity.rotate(ell_ext, ellipse_ext[2])
    interseccion_ext = hexagon.intersection(ellr_ext)
    inter_area_ext = interseccion_ext.area
    inter_area_ext_norm = inter_area_ext / hexagon.area

    # ELIPSE interior
    b_int = a_int * (1 - ellipticity)  # ELIPTICIDAD CONSTANTE
    ellipse_int = (centro, (a_int, b_int), angle)
    circ_int = sg.Point(ellipse_int[0]).buffer(1)
    ell_int = shapely.affinity.scale(
        circ_int, float(ellipse_int[1][0]), float(ellipse_int[1][1])
    )
    ellr_int = shapely.affinity.rotate(ell_int, ellipse_int[2])
    interseccion_int = hexagon.intersection(ellr_int)
    inter_area_int = interseccion_int.area
    inter_area_int_norm = inter_area_int / hexagon.area

    return inter_area_ext_norm - inter_area_int_norm


def plot_ring(b, ell_x_offset, ell_y_offset, angle, a_ext, a_int, ellipticity):
    fig, ax = plt.subplots()

    # Dibujar cada hexágono en la posición correspondiente
    for ind, val in enumerate(b):
        x_offset, y_offset = val
        lado = 0.62 / 2
        vertices = [
            [x_offset + lado * 1 / 2, y_offset + lado * (np.sqrt(3) / 2)],
            [x_offset + lado * 1, y_offset + 0],
            [x_offset + lado * 1 / 2, y_offset - lado * (np.sqrt(3) / 2)],
            [x_offset - lado * 1 / 2, y_offset - lado * (np.sqrt(3) / 2)],
            [x_offset - lado * 1, y_offset + 0],
            [x_offset - lado * 1 / 2, y_offset + (lado * np.sqrt(3) / 2)],
        ]
        hex_patch = Polygon(
            vertices, closed=True, fill=True, edgecolor="w", facecolor="b", alpha=0.2
        )
        ax.add_patch(hex_patch)
        # Incluir el número de ind dentro de cada hexágono
        ax.text(x_offset, y_offset, str(ind + 1), ha="center", va="center", fontsize=3)

    # Crear la elipse exterior
    b_ext = a_ext * (1 - ellipticity)
    ellipse_ext = sg.Point((ell_x_offset, ell_y_offset)).buffer(1)
    ellipse_ext = shapely.affinity.scale(ellipse_ext, a_ext, b_ext)
    ellipse_ext = shapely.affinity.rotate(ellipse_ext, angle)

    # Crear la elipse interior
    b_int = a_int * (1 - ellipticity)
    ellipse_int = sg.Point((ell_x_offset, ell_y_offset)).buffer(1)
    ellipse_int = shapely.affinity.scale(ellipse_int, a_int, b_int)
    ellipse_int = shapely.affinity.rotate(ellipse_int, angle)

    # Calcular la diferencia entre las elipses correctamente
    ring_area = ellipse_ext.difference(ellipse_int)

    # Dibujar la diferencia entre la elipse exterior e interior con color
    if not ring_area.is_empty:
        ring_patch = PathPatch(
            Path.make_compound_path(
                Path(ring_area.exterior.coords),
                *[Path(interior.coords) for interior in ring_area.interiors],
            ),
            facecolor="r",
            edgecolor="k",
            alpha=0.5,
        )
        ax.add_patch(ring_patch)

    # Dibujar la elipse interior con transparencia
    if not ellipse_int.is_empty:
        int_patch = PathPatch(
            Path(ellipse_int.exterior.coords),
            facecolor="none",
            edgecolor="g",
            alpha=0.5,
        )
        ax.add_patch(int_patch)

    # Configurar los límites del gráfico
    x_min, x_max = -6.2, 6.2
    y_min, y_max = -6.2, 6.2
    ax.set_xlim(x_min - abs(0.1 * x_min), x_max + abs(0.1 * x_max))
    ax.set_ylim(y_min - abs(0.1 * y_min), y_max + abs(0.1 * y_max))

    # Configurar los parámetros de los ejes
    ax.xaxis.set_tick_params(length=5, width=1, labelsize=12)
    ax.yaxis.set_tick_params(length=5, width=1, labelsize=12)

    # Etiquetas de los ejes
    plt.xlabel("[arcsec]", fontsize=12)
    plt.ylabel("[arcsec]", fontsize=12)

    # Configurar el aspecto del gráfico
    plt.setp(ax.spines.values(), linewidth=1, zorder=100)
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.99)

    # Configurar los localizadores menores de los ejes
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Mantener la relación de aspecto igual
    ax.set_aspect("equal")

    # Guardar la figura
    plt.savefig(f"ring_{a_ext}_{a_int}.png", dpi=600)


if __name__ == "__main__":
    main()
