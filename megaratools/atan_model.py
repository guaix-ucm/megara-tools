import math
from packaging.version import parse, Version
from operator import attrgetter

import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.transforms as mtrans
import matplotlib.transforms as mtransforms
import numpy as np
import warnings
import matplotlib.cbook
from megaradrp.processing.cube import main as megaracube

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def populate_array(rows, columns):
    array_dic = {}
    for row in range(0, rows):
        array_dic[row] = []
        for col in range(0, columns):
            array_dic[row].append("None")  # initialize to 'None'
    return array_dic


def residuals(paramsin, x, y, z, ez, errors):
    import numpy as np

    vsys = paramsin["vsys"].value
    pa = paramsin["pa"].value
    incldg = paramsin["incldg"].value
    a = paramsin["a"].value
    b = paramsin["b"].value
    xcenter = paramsin["xcenter"].value
    ycenter = paramsin["ycenter"].value
    inclrad = incldg * np.pi / 180.0
    theta0 = (pa - 360) * (np.pi / 180.0)
    theta0 = (pa - 360) * (np.pi / 180.0)
    r, theta = cart2polar(x, y, xcenter, ycenter)
    theta = theta + np.pi - theta0
    theta = np.where(theta < 2 * np.pi, theta, theta - 2 * np.pi)
    thetag = np.arctan(np.tan(theta) / np.cos(inclrad))
    thetag = np.where(
        (np.abs(theta) > np.pi / 2.0) & ((np.abs(theta) < 3.0 * np.pi / 2.0)),
        thetag + np.pi,
        thetag,
    )
    thetag = np.where((np.abs(theta) > 3.0 * np.pi / 2.0), thetag + 2.0 * np.pi, thetag)
    rr = r * (np.cos(theta) / np.cos(thetag))
    vr = a * np.arctan(b * rr)
    zmodel = (
        vsys
        + vr
        * np.sin(inclrad)
        * np.cos(theta)
        * np.sqrt(
            (np.cos(inclrad) ** 2) / (1 - np.sin(inclrad) ** 2 * np.cos(theta) ** 2)
        )
        - z
    )
    if errors:
        weighted = np.lib.scimath.sqrt(zmodel**2 / ez**2)
    else:
        weighted = zmodel
    return weighted


def vfunc(paramsin, x, y, z):
    import numpy as np
    import matplotlib.pyplot as plt

    vsys = paramsin["vsys"].value
    pa = paramsin["pa"].value
    incldg = paramsin["incldg"].value
    a = paramsin["a"].value
    b = paramsin["b"].value
    xcenter = paramsin["xcenter"].value
    ycenter = paramsin["ycenter"].value
    inclrad = incldg * np.pi / 180.0
    theta0 = (pa - 360) * (np.pi / 180.0)
    r, theta = cart2polar(x, y, xcenter, ycenter)
    theta = theta + np.pi - theta0
    theta = np.where(theta < 2 * np.pi, theta, theta - 2 * np.pi)
    thetag = np.arctan(np.tan(theta) / np.cos(inclrad))
    thetag = np.where(
        (np.abs(theta) > np.pi / 2.0) & ((np.abs(theta) < 3.0 * np.pi / 2.0)),
        thetag + np.pi,
        thetag,
    )
    thetag = np.where((np.abs(theta) > 3.0 * np.pi / 2.0), thetag + 2.0 * np.pi, thetag)
    rr = r * (np.cos(theta) / np.cos(thetag))
    print("Maximum Galactocentric R (arcsec): ", np.max(rr))

    vr = a * np.arctan(b * rr)
    zmodel = vsys + vr * np.sin(inclrad) * np.cos(theta) * np.sqrt(
        (np.cos(inclrad) ** 2) / (1 - np.sin(inclrad) ** 2 * np.cos(theta) ** 2)
    )
    return zmodel


def create_cube_from_array2(x614, y614, x, y, rss_data, rows, target_scale_arcsec):
    HEX_SCALE = 0.536916  # Arcseconds
    target_scale = target_scale_arcsec / HEX_SCALE
    region = rss_data[rows, :]
    spos1_x = np.asarray(x)
    spos1_y = np.asarray(y)
    if x614 < -6:
        # arcsec
        ascale = 0.536916  # 0.443 * 1.212
        print("scale is arcsec")
    else:
        # mm
        ascale = 0.443
        print("scale is mm")
    refx, refy = x614 / ascale, y614 / ascale
    rpos1_x = (spos1_x - x614) / ascale
    rpos1_y = (spos1_y - y614) / ascale
    r0l = np.array([rpos1_x, rpos1_y])
    cube_data = megaracube.create_cube(r0l, region[:, :], target_scale)
    result = np.moveaxis(cube_data, 2, 0)
    result.astype("float32")
    return result


def create_cube_from_array(rss_data, fiberconf, target_scale_arcsec):
    HEX_SCALE = 0.536916  # Arcseconds
    target_scale = target_scale_arcsec / HEX_SCALE
    conected = fiberconf.conected_fibers()
    rows = [conf.fibid - 1 for conf in conected]
    region = rss_data[rows, :]

    r0l, (refx, refy) = megaracube.calc_matrix_from_fiberconf(fiberconf)
    cube_data = megaracube.create_cube(r0l, region[:, :], target_scale)
    result = np.moveaxis(cube_data, 2, 0)
    result.astype("float32")
    return result


def merge_wcs_2d(hdr_sky, hdr_spec, out=None):
    if out is None:
        hdr = hdr_spec.copy()
    else:
        hdr = out
    # Extend header for third axis
    c_crpix = "Pixel coordinate of reference point"
    c_cunit = "Units of coordinate increment and value"
    hdr.set("CUNIT1", comment=c_cunit, after="CDELT1")
    hdr.set("CUNIT2", comment=c_cunit, after="CUNIT1")
    hdr.set("CRPIX2", value=1, comment=c_crpix, after="CRPIX1")
    c_pc = "Coordinate transformation matrix element"
    hdr.set("PC1_1", value=1.0, comment=c_pc)
    hdr.set("PC1_2", value=0.0, comment=c_pc, after="PC1_1")
    hdr.set("PC2_1", value=0.0, comment=c_pc)
    hdr.set("PC2_2", value=1.0, comment=c_pc, after="PC2_1")
    # Mapping, which keyword comes from each header
    mappings = [
        ("CRPIX1", "CRPIX1", 1, 0.0),
        ("CDELT1", "CDELT1", 1, 1.0),
        ("CRVAL1", "CRVAL1", 1, 0.0),
        ("CTYPE1", "CTYPE1", 1, " "),
        ("CUNIT1", "CUNIT1", 1, " "),
        ("PC1_1", "PC1_1", 1, 1.0),
        ("PC1_2", "PC1_2", 1, 0.0),
        ("CRPIX2", "CRPIX2", 1, 0.0),
        ("CDELT2", "CDELT2", 1, 1.0),
        ("CRVAL2", "CRVAL2", 1, 0.0),
        ("CTYPE2", "CTYPE2", 1, " "),
        ("CUNIT2", "CUNIT2", 1, " "),
        ("PC2_1", "PC2_1", 1, 0.0),
        ("PC2_2", "PC2_2", 1, 1.0),
    ]
    idem_keys = [("LONPOLE", 0.0), ("RADESYS", "FK5")]
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


def recompute_wcs(hdr, ora, odec):
    """Recompute the WCS rotations from IPA"""
    # get IPA keyword
    insAngle = -163.256
    print("insAngle is:", insAngle)
    ipa = hdr["IPA"]
    pa = -insAngle + ipa
    print("IPA angle is:", ipa, "PA angle is", math.fmod(pa, 360))
    x = hdr["PC1_1"]
    y = hdr["PC1_2"]
    print("PA from header is:", np.rad2deg(math.atan2(y, x)))
    pa_rad = np.deg2rad(pa)
    cos_pa = math.cos(pa_rad)
    sin_pa = math.sin(pa_rad)
    # Update PC_ keywords
    print("PC1_1 was {}, is {}".format(hdr["PC1_1"], cos_pa))
    hdr["PC1_1"] = cos_pa
    print("PC2_2 was {}, is {}".format(hdr["PC2_2"], cos_pa))
    hdr["PC2_2"] = cos_pa
    print("PC1_2 was {}, is {}".format(hdr["PC1_2"], sin_pa))
    hdr["PC1_2"] = sin_pa
    print("PC2_1 was {}, is {}".format(hdr["PC2_1"], -sin_pa))
    hdr["PC2_1"] = -sin_pa
    x = hdr["CRVAL1"]
    y = hdr["CRVAL2"]
    hdr["CRVAL1"] = x + (ora / 3600.0) / math.cos(y * np.pi / 180.0)
    hdr["CRVAL2"] = y + (odec / 3600.0)
    return hdr


def hexplot(
    axis,
    x,
    y,
    z,
    scale=1.0,
    extent=None,
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    alpha=None,
    linewidths=None,
    edgecolors="none",
    **kwargs
):
    """
    Make a hexagonal grid plot.
    Returns
    =======
    object: matplotlib.collections.PolyCollection
    """

    # I have to add these due to changes in private and protected interfaces
    mpl_version = parse(matplotlib.__version__)
    if mpl_version >= Version("3.4"):
        axis._process_unit_info([("x", x), ("y", y)], kwargs, convert=False)
    else:
        axis._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)

    x, y, z = cbook.delete_masked_points(x, y, z)

    x = np.array(x, float)
    y = np.array(y, float)

    M_SQRT3 = math.sqrt(3)
    M_1_SQRT3 = 1 / M_SQRT3

    sx = 2 * M_1_SQRT3 * scale * 0.99
    sy = scale * 0.99

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
    else:
        xmin, xmax = (np.amin(x - sx), np.amax(x + sx)) if len(x) else (0, 1)
        ymin, ymax = (np.amin(y - sy), np.amax(y + sy)) if len(y) else (0, 1)

        # to avoid issues with singular data, expand the min/max pairs
        xmin, xmax = mtrans.nonsingular(xmin, xmax, expander=0.1)
        ymin, ymax = mtrans.nonsingular(ymin, ymax, expander=0.1)

    padding = 1.0e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding

    n = len(x)
    polygon = np.zeros((6, 2), float)

    mx = my = 0.99 * scale
    polygon[:, 0] = mx * np.array(
        [
            -0.5 * M_1_SQRT3,
            0.5 * M_1_SQRT3,
            1.0 * M_1_SQRT3,
            0.5 * M_1_SQRT3,
            -0.5 * M_1_SQRT3,
            -1.0 * M_1_SQRT3,
        ]
    )
    polygon[:, 1] = my * np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0])

    offsets = np.zeros((n, 2), float)
    offsets[:, 0] = x
    offsets[:, 1] = y

    # I have to add these due to changes in private and protected interfaces
    if mpl_version >= Version("3.3"):
        collection = mcoll.PolyCollection(
            [polygon],
            edgecolors=edgecolors,
            linewidths=linewidths,
            offsets=offsets,
            transOffset=mtransforms.AffineDeltaTransform(axis.transData),
        )
    else:
        collection = mcoll.PolyCollection(
            [polygon],
            edgecolors=edgecolors,
            linewidths=linewidths,
            offsets=offsets,
            transOffset=mtransforms.IdentityTransform(),
            offset_position="data",
        )

    if isinstance(norm, mcolors.LogNorm):
        if (z == 0).any():
            # make sure we have not zeros
            z += 1

    if norm is not None:
        if norm.vmin is None and norm.vmax is None:
            norm.autoscale(z)

    if norm is not None and not isinstance(norm, mcolors.Normalize):
        msg = "'norm' must be an instance of 'mcolors.Normalize'"
        raise ValueError(msg)

    collection.set_array(z)
    collection.set_cmap(cmap)
    collection.set_norm(norm)
    collection.set_alpha(alpha)
    collection.update(kwargs)

    if vmin is not None or vmax is not None:
        collection.set_clim(vmin, vmax)
    else:
        collection.autoscale_None()

    corners = ((xmin, ymin), (xmax, ymax))
    axis.update_datalim(corners)
    axis.autoscale_view(tight=True)

    # add the collection last
    axis.add_collection(collection, autolim=False)
    return collection


def index_coords(data, origin=None):
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_y, origin_x = origin
        if origin_y < 0:
            origin_y += ny
        if origin_x < 0:
            origin_x += nx
    x, y = np.meshgrid(np.arange(float(nx)) - origin_x, origin_y - np.arange(float(ny)))
    print(x.shape, y.shape)
    return x, y


def cart2polar(x, y, xcenter, ycenter):
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)
    theta = np.arctan2(x - xcenter, -y + ycenter)  # # referenced to vertical
    return r, theta


def polar2cart(r, theta):
    y = r * np.cos(theta)  # # referenced to vertical
    x = r * np.sin(theta)
    return x, y


def main(args=None):
    # if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import astropy.io.fits as fits
    from astropy.wcs import WCS
    import lmfit
    from astropy.time import Time
    import astropy.units as u
    from astropy.coordinates import SkyCoord, EarthLocation
    from pylab import figure
    import megaradrp.datamodel as dm
    import matplotlib.backends.backend_pdf
    import argparse

    plt.rcParams.update({"font.size": 14})
    plt.rcParams["contour.negative_linestyle"] = "solid"

    parser = argparse.ArgumentParser(
        description="Input RSS", prog="Kinematic model for MEGARA RSS data"
    )
    parser.add_argument(
        "-s",
        "--spectrum",
        metavar="RSS SPECTRUM/PRODUCT",
        help="FITS RSS data or product",
        type=argparse.FileType("rb"),
    )
    parser.add_argument(
        "-c",
        "--channel",
        metavar="RSS VELOCITY CHANNEL",
        help="Channel of RSS product for velocity",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-e",
        "--echannel",
        metavar="RSS VELOCITY ERROR CHANNEL",
        help="Channel of RSS product for velocity error",
        type=int,
        default=28,
    )
    parser.add_argument(
        "-Z1",
        "--zcut1",
        metavar="LOWER CUT LIMIT (FOR RSS OR DIRECT IMAGE)",
        help="Lower cut limit for plot",
        type=float,
    )
    parser.add_argument(
        "-Z2",
        "--zcut2",
        metavar="UPPER CUT LIMIT (FOR RSS OR DIRECT IMAGE)",
        help="Upper cut limit for plot",
        type=float,
    )
    parser.add_argument(
        "-p",
        "--palette",
        metavar="PALETTE",
        default="jet",
        help="Matplotlib palette of the plot",
    )
    parser.add_argument(
        "-u",
        "--units",
        metavar="UNITS LABEL",
        default="km s$^{-1}$",
        help="Label of the plot",
    )
    parser.add_argument(
        "-CO",
        "--commissioning",
        default=False,
        action="store_true",
        help="Commissioning image?",
    )
    parser.add_argument(
        "-N",
        "--signal-to-noise",
        metavar="MINIMUM SIGNAL TO NOISE RATIO",
        help="Lower limit in Signal-to-Noise",
        type=float,
    )
    parser.add_argument(
        "-vmin",
        "--vmin",
        metavar="MINIMUM VELOCITY (km/s)",
        help="Minimum velocity to be considered in fit",
        type=float,
    )
    parser.add_argument(
        "-vmax",
        "--vmax",
        metavar="MAXIMUM VELOCITY (km/s)",
        help="Maximum velocity to be considered in fit",
        type=float,
    )
    parser.add_argument(
        "-smin",
        "--minimum-sigma",
        metavar="MINIMUM SIGMA (km/s)",
        help="Minimum sigma (corrected for instrumental effects) to be considered in fit",
        type=float,
    )
    parser.add_argument(
        "-P",
        "--peak",
        default=False,
        action="store_true",
        help="Fix the center to peak in continuum emission?",
    )
    parser.add_argument(
        "-I",
        "--fix-incl",
        default=False,
        action="store_true",
        help="Fix the inclination to the initial value?",
    )
    parser.add_argument(
        "-r",
        "--rescale",
        metavar="RESCALE",
        default=1.0,
        help="Rescale for hexagon sizes",
        type=float,
    )
    parser.add_argument(
        "-ox",
        "--offset-ra",
        metavar="OFFSET RA",
        default=0.0,
        help="Astrometry correction to MEGARA image (RA; arcsec)",
        type=float,
    )
    parser.add_argument(
        "-oy",
        "--offset-dec",
        metavar="OFFSET DEC",
        default=0.0,
        help="Astrometry correction to MEGARA image (Dec; arcsec)",
        type=float,
    )
    parser.add_argument(
        "-g",
        "--title-color",
        metavar="TITLE/GRID COLOR",
        default="k",
        help="Title & grid color (default=k=key)",
    )
    parser.add_argument(
        "-A",
        "--alpha",
        metavar="ALPHA",
        default=0.0,
        help="Alpha value for grid (def=0=transp; 1=solid black)",
        type=float,
    )
    parser.add_argument(
        "-O", "--output-plot", metavar="OUTPUT PLOT", help="Plot output file (PDF)"
    )
    parser.add_argument(
        "-a", "--a-par", metavar="PARAMETER A", help="Parameter a (km/s)", type=float
    )
    parser.add_argument(
        "-b",
        "--b-par",
        metavar="PARAMETER B",
        help="Parameter b (arcsec^-1)",
        type=float,
    )
    parser.add_argument(
        "-pa", "--pa-par", metavar="PARAMETER PA", help="Parameter PA (dg)", type=float
    )
    parser.add_argument(
        "-z", "--z-par", metavar="PARAMETER Z", help="Parameter Redshift", type=float
    )
    parser.add_argument(
        "-i",
        "--i-par",
        metavar="PARAMETER I",
        default=45.0,
        help="Parameter inclination (dg)",
        type=float,
    )
    parser.add_argument(
        "-x",
        "--x-par",
        metavar="PARAMETER X",
        default=0.0,
        help="Parameter xcenter (km/s)",
        type=float,
    )
    parser.add_argument(
        "-y",
        "--y-par",
        metavar="PARAMETER Y",
        default=0.0,
        help="Parameter ycenter (km/s)",
        type=float,
    )
    parser.add_argument(
        "-E",
        "--errors",
        default=False,
        action="store_true",
        help="Use the errors for the fit?",
    )
    parser.add_argument(
        "-C",
        "--confidence-intervals",
        default=False,
        action="store_true",
        help="Use the errors for the fit?",
    )

    args = parser.parse_args()

    PLATESCALE = 1.2120  # arcsec / mm
    SCALE = 0.443  # mm from center to center, upwards
    SIZE = SCALE * PLATESCALE

    x = []
    y = []
    z = []
    z_tmp = []
    c_tmp = []
    cont = []
    ez = []
    ez_tmp = []
    connected_ids = []

    if args.spectrum is not None:
        fname = args.spectrum
        img = fits.open(fname)
        datamodel = dm.MegaraDataModel()
        fiberconf = datamodel.get_fiberconf(img)
        for fiber in sorted(fiberconf.connected_fibers(), key=attrgetter("fibid")):
            connected_ids.append(fiber.fibid - 1)
            if args.commissioning:
                x.append(fiber.x / PLATESCALE)
                y.append(fiber.y / PLATESCALE)
            else:
                x.append(fiber.x)
                y.append(fiber.y)
        z_inter = img[0].data[:, args.channel]
        ez_inter = img[0].data[:, args.echannel]
        c_inter = img[0].data[:, 1]
        c_tmp.extend(c_inter)
        cont = [c_tmp[i] for i in connected_ids]
        index_peak = max(range(len(cont)), key=cont.__getitem__)
        xpeak = x[index_peak]
        ypeak = y[index_peak]
        print("x_peak = ", xpeak)
        print("y_peak = ", ypeak)
        if args.signal_to_noise is not None:
            cutchannel = 3
            snr = img[0].data[:, cutchannel]
            for i in range(len(z_inter)):
                if snr[i] < float(args.signal_to_noise):
                    z_inter[i] = np.sqrt(-1)
                    ez_inter[i] = np.sqrt(-1)
                else:
                    z_inter[i] = z_inter[i]
                    ez_inter[i] = ez_inter[i]
        if args.vmin is not None and args.vmax is not None:
            cutchannel = 16
            v = img[0].data[:, cutchannel]
            for i in range(len(z_inter)):
                if v[i] < float(args.vmin) or v[i] > float(args.vmax):
                    z_inter[i] = np.sqrt(-1)
                    ez_inter[i] = np.sqrt(-1)
                else:
                    z_inter[i] = z_inter[i]
                    ez_inter[i] = ez_inter[i]
        if args.minimum_sigma is not None:
            cutchannel = 18
            sigma = img[0].data[:, cutchannel]
            for i in range(len(z_inter)):
                if sigma[i] < float(args.minimum_sigma):
                    z_inter[i] = np.sqrt(-1)
                    ez_inter[i] = np.sqrt(-1)
                else:
                    z_inter[i] = z_inter[i]
                    ez_inter[i] = ez_inter[i]
        z_inter[622] = (
            z_inter[619]
            + z_inter[524]
            + z_inter[528]
            + z_inter[184]
            + z_inter[183]
            + z_inter[621]
        ) / 6
        ez_inter[622] = (
            ez_inter[619]
            + ez_inter[524]
            + ez_inter[528]
            + ez_inter[184]
            + ez_inter[183]
            + ez_inter[621]
        ) / 6
        z_tmp.extend(z_inter)
        ez_tmp.extend(ez_inter)
        z = [z_tmp[i] for i in connected_ids]
        ez = [ez_tmp[i] for i in connected_ids]

    merge_wcs_2d(img["FIBERS"].header, img[0].header, out=img[0].header)
    img[0].header = recompute_wcs(img[0].header, args.offset_ra, args.offset_dec)
    wcs2 = WCS(img[0].header, relax=False).celestial
    gtc = EarthLocation.of_site("lapalma")
    barycorr = SkyCoord(
        "01:58:00 +65:43:05", frame="fk5", unit=(u.hourangle, u.deg)
    ).radial_velocity_correction(
        "barycentric", obstime=Time("2019-09-24T02:23:22.19"), location=gtc
    )

    # Plotting the input velocity map
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs2)
    ax.tick_params(axis="both", labelsize=14)
    ax.minorticks_on()
    ax.coords["ra"].set_major_formatter("hh:mm:ss.s")
    ax.set_xlim([-6.5, 6.5])
    ax.set_ylim([-6.5, 6.5])
    col = hexplot(
        ax,
        x,
        y,
        z,
        scale=SCALE * args.rescale,
        cmap=args.palette,
        vmin=args.zcut1,
        vmax=args.zcut2,
        alpha=0.99,
        zorder=20,
    )
    cb = plt.colorbar(col)
    ax.coords.grid(color=args.title_color, alpha=args.alpha, linestyle="dashed")
    cb.set_label(args.units, fontsize=19)
    ax.coords.grid(color=args.title_color, alpha=args.alpha, linestyle="dashed")
    plt.title(
        "Observed radial velocity",
        x=0.04,
        y=0.9,
        zorder=30,
        loc="left",
        color=args.title_color,
    )
    ax.coords["dec"].set_axislabel("DEC (J2000)")
    ax.coords["ra"].set_axislabel("RA (J2000)")
    if args.output_plot is None:
        plt.show()

    # Creating the kinematic model
    maskz = np.isnan(z)
    zspx = np.delete(z, maskz)
    xspx = np.delete(np.array(x), maskz)
    yspx = np.delete(np.array(y), maskz)
    ezspx = np.delete(ez, maskz)

    # Initial solution
    print("Running model ...")
    p_v = lmfit.Parameters()
    p_v.add("vsys", value=args.z_par * 300000.0, vary=True)
    p_v.add("pa", value=args.pa_par, min=0.0, max=360.0, vary=True)
    if args.fix_incl:
        p_v.add("incldg", value=args.i_par, vary=False)
    else:
        p_v.add("incldg", value=args.i_par, min=1.0, max=90.0, vary=True)
    p_v.add("a", value=args.a_par, min=0.0, vary=True)
    p_v.add("b", value=args.b_par, min=0.0, vary=True)
    if args.peak:
        p_v.add("xcenter", value=xpeak, vary=False)
        p_v.add("ycenter", value=ypeak, vary=False)
    else:
        p_v.add("xcenter", value=args.x_par, min=-6.0, max=+6.0, vary=True)
        p_v.add("ycenter", value=args.y_par, min=-6.0, max=+6.0, vary=True)

    initial_p_v = p_v
    # Plotting the initial model
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs2)
    ax.tick_params(axis="both", labelsize=14)
    ax.minorticks_on()
    ax.coords["ra"].set_major_formatter("hh:mm:ss.s")
    ax.set_xlim([-6.5, 6.5])
    ax.set_ylim([-6.5, 6.5])
    print("Initial model")
    imodel = vfunc(initial_p_v, x, y, z)
    col = hexplot(
        ax,
        x,
        y,
        imodel,
        scale=SCALE * args.rescale,
        cmap=args.palette,
        vmin=args.zcut1,
        vmax=args.zcut2,
        alpha=0.99,
        zorder=20,
    )
    cb = plt.colorbar(col)
    cb.set_label(args.units, fontsize=19)
    ax.coords.grid(color=args.title_color, alpha=args.alpha, linestyle="dashed")
    plt.title(
        "Radial velocity initial model",
        x=0.04,
        y=0.9,
        zorder=30,
        loc="left",
        color=args.title_color,
    )
    ax.coords["dec"].set_axislabel("DEC (J2000)")
    ax.coords["ra"].set_axislabel("RA (J2000)")
    if args.output_plot is None:
        plt.show()

    # Fitting ...
    verr = lambda p, x, y, z, ez: residuals(p, x, y, z, ez, args.errors)  # noqa
    mini = lmfit.Minimizer(
        verr, p_v, fcn_args=(xspx, yspx, zspx, ezspx), nan_policy="omit"
    )
    fitout_v = mini.minimize(method="leastsq")
    fitted_p_v = fitout_v.params
    lmfit.report_fit(fitout_v)
    ci, trace = lmfit.conf_interval(mini, fitout_v, sigmas=[1, 2], trace=True)
    lmfit.printfuncs.report_ci(ci)

    # Plot confidence intervals ("a vs incldg" and "a vs b")
    if args.confidence_intervals:
        rows = 6
        columns = 6
        incl = "incldg"
        xcnt = "xcenter"
        ycnt = "ycenter"
        vsys = "vsys"

        vsysl = "v (km/s)"
        xcntl = "x-off (arcsec)"
        ycntl = "y-off (arcsec)"
        incll = "incl. (dg)"

        propertyx = {
            "(0, 0)": "a",
            "(0, 1)": "b",
            "(0, 2)": "pa",
            "(0, 3)": vsys,
            "(0, 4)": xcnt,
            "(0, 5)": ycnt,
            "(0, 6)": incl,
            "(1, 0)": "a",
            "(1, 1)": "b",
            "(1, 2)": "pa",
            "(1, 3)": vsys,
            "(1, 4)": xcnt,
            "(1, 5)": ycnt,
            "(1, 6)": incl,
            "(2, 0)": "a",
            "(2, 1)": "b",
            "(2, 2)": "pa",
            "(2, 3)": vsys,
            "(2, 4)": xcnt,
            "(2, 5)": ycnt,
            "(2, 6)": incl,
            "(3, 0)": "a",
            "(3, 1)": "b",
            "(3, 2)": "pa",
            "(3, 3)": vsys,
            "(3, 4)": xcnt,
            "(3, 5)": ycnt,
            "(3, 6)": incl,
            "(4, 0)": "a",
            "(4, 1)": "b",
            "(4, 2)": "pa",
            "(4, 3)": vsys,
            "(4, 4)": xcnt,
            "(4, 5)": ycnt,
            "(4, 6)": incl,
            "(5, 0)": "a",
            "(5, 1)": "b",
            "(5, 2)": "pa",
            "(5, 3)": vsys,
            "(5, 4)": xcnt,
            "(5, 5)": ycnt,
            "(5, 6)": incl,
            "(6, 0)": "a",
            "(6, 1)": "b",
            "(6, 2)": "pa",
            "(6, 3)": vsys,
            "(6, 4)": xcnt,
            "(6, 5)": ycnt,
            "(6, 6)": incl,
        }

        labelsx = {
            "(0, 0)": "a (km/s)",
            "(0, 1)": "b (arcsec^-1)",
            "(0, 2)": "pa (dg)",
            "(0, 3)": vsysl,
            "(0, 4)": xcntl,
            "(0, 5)": ycntl,
            "(0, 6)": incll,
            "(1, 0)": "a (km/s)",
            "(1, 1)": "b (arcsec^-1)",
            "(1, 2)": "pa (dg)",
            "(1, 3)": vsysl,
            "(1, 4)": xcntl,
            "(1, 5)": ycntl,
            "(1, 6)": incll,
            "(2, 0)": "a (km/s)",
            "(2, 1)": "b (arcsec^-1)",
            "(2, 2)": "pa (dg)",
            "(2, 3)": vsysl,
            "(2, 4)": xcntl,
            "(2, 5)": ycntl,
            "(2, 6)": incll,
            "(3, 0)": "a (km/s)",
            "(3, 1)": "b (arcsec^-1)",
            "(3, 2)": "pa (dg)",
            "(3, 3)": vsysl,
            "(3, 4)": xcntl,
            "(3, 5)": ycntl,
            "(3, 6)": incll,
            "(4, 0)": "a (km/s)",
            "(4, 1)": "b (arcsec^-1)",
            "(4, 2)": "pa (dg)",
            "(4, 3)": vsysl,
            "(4, 4)": xcntl,
            "(4, 5)": ycntl,
            "(4, 6)": incll,
            "(5, 0)": "a (km/s)",
            "(5, 1)": "b (arcsec^-1)",
            "(5, 2)": "pa (dg)",
            "(5, 3)": vsysl,
            "(5, 4)": xcntl,
            "(5, 5)": ycntl,
            "(5, 6)": incll,
            "(6, 0)": "a (km/s)",
            "(6, 1)": "b (arcsec^-1)",
            "(6, 2)": "pa (dg)",
            "(6, 3)": vsysl,
            "(6, 4)": xcntl,
            "(6, 5)": ycntl,
            "(6, 6)": incll,
        }

        propertyy = {
            "(0, 0)": "a",
            "(0, 1)": "a",
            "(0, 2)": "a",
            "(0, 3)": "a",
            "(0, 4)": "a",
            "(0, 5)": "a",
            "(0, 6)": "a",
            "(1, 0)": "b",
            "(1, 1)": "b",
            "(1, 2)": "b",
            "(1, 3)": "b",
            "(1, 4)": "b",
            "(1, 5)": "b",
            "(1, 6)": "b",
            "(2, 0)": "pa",
            "(2, 1)": "pa",
            "(2, 2)": "pa",
            "(2, 3)": "pa",
            "(2, 4)": "pa",
            "(2, 5)": "pa",
            "(2, 6)": "pa",
            "(3, 0)": vsys,
            "(3, 1)": vsys,
            "(3, 2)": vsys,
            "(3, 3)": vsys,
            "(3, 4)": vsys,
            "(3, 5)": vsys,
            "(3, 6)": vsys,
            "(4, 0)": xcnt,
            "(4, 1)": xcnt,
            "(4, 2)": xcnt,
            "(4, 3)": xcnt,
            "(4, 4)": xcnt,
            "(4, 5)": xcnt,
            "(4, 6)": xcnt,
            "(5, 0)": ycnt,
            "(5, 1)": ycnt,
            "(5, 2)": ycnt,
            "(5, 3)": ycnt,
            "(5, 4)": ycnt,
            "(5, 5)": ycnt,
            "(5, 6)": ycnt,
            "(6, 0)": incl,
            "(6, 1)": incl,
            "(6, 2)": incl,
            "(6, 3)": incl,
            "(6, 4)": incl,
            "(6, 5)": incl,
            "(6, 6)": incl,
        }

        labelsy = {
            "(0, 0)": "a (km/s)",
            "(0, 1)": "a (km/s)",
            "(0, 2)": "a (km/s)",
            "(0, 3)": "a (km/s)",
            "(0, 4)": "a (km/s)",
            "(0, 5)": "a (km/s)",
            "(0, 6)": "a (km/s)",
            "(1, 0)": "b (arcsec^-1)",
            "(1, 1)": "b (arcsec^-1)",
            "(1, 2)": "b (arcsec^-1)",
            "(1, 3)": "b (arcsec^-1)",
            "(1, 4)": "b (arcsec^-1)",
            "(1, 5)": "b (arcsec^-1)",
            "(1, 6)": "b (arcsec^-1)",
            "(2, 0)": "pa (dg)",
            "(2, 1)": "pa (dg)",
            "(2, 2)": "pa (dg)",
            "(2, 3)": "pa (dg)",
            "(2, 4)": "pa (dg)",
            "(2, 5)": "pa (dg)",
            "(2, 6)": "pa (dg)",
            "(3, 0)": vsysl,
            "(3, 1)": vsysl,
            "(3, 2)": vsysl,
            "(3, 3)": vsysl,
            "(3, 4)": vsysl,
            "(3, 5)": vsysl,
            "(3, 6)": vsysl,
            "(4, 0)": xcntl,
            "(4, 1)": xcntl,
            "(4, 2)": xcntl,
            "(4, 3)": xcntl,
            "(4, 4)": xcntl,
            "(4, 5)": xcntl,
            "(4, 6)": xcntl,
            "(5, 0)": ycntl,
            "(5, 1)": ycntl,
            "(5, 2)": ycntl,
            "(5, 3)": ycntl,
            "(5, 4)": ycntl,
            "(5, 5)": ycntl,
            "(5, 6)": ycntl,
            "(6, 0)": incll,
            "(6, 1)": incll,
            "(6, 2)": incll,
            "(6, 3)": incll,
            "(6, 4)": incll,
            "(6, 5)": incll,
            "(6, 6)": incll,
        }

        fig, axes = plt.subplots(rows + 1, columns + 1, figsize=(17, 13))
        for row in range(0, rows + 1):
            for col in range(0, row):
                key = "(" + str(row) + ", " + str(col) + ")"
                print(key, propertyx[key], propertyy[key])
                cx, cy, grid = lmfit.conf_interval2d(
                    mini, fitout_v, propertyx[key], propertyy[key], 30, 30
                )
                ctp = axes[row, col].contourf(cx, cy, grid, np.linspace(0, 1, 11))
                fig.colorbar(ctp, ax=axes[row, col])
                axes[row, col].minorticks_on()
                if row == rows:
                    axes[row, col].set_xlabel(labelsx[key])
                    axes[row, col].tick_params(axis="x", labelrotation=45)
                else:
                    axes[row, col].set_xticklabels([])
                if col > 0:
                    axes[row, col].set_yticklabels([])
                else:
                    axes[row, col].set_ylabel(labelsy[key])
                    axes[row, col].tick_params(axis="y", labelrotation=45)
            for col in range(row, columns + 1):
                axes[row, col].axis("off")

    # Plot velocity model only in valid pixels/spaxels
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs2)
    ax.tick_params(axis="both", labelsize=14)
    ax.minorticks_on()
    ax.coords["ra"].set_major_formatter("hh:mm:ss.s")
    ax.set_xlim([-6.5, 6.5])
    ax.set_ylim([-6.5, 6.5])
    print("Best model only in valid pixels/spaxels")
    bzmodel = vfunc(fitted_p_v, xspx, yspx, zspx)
    bnnres = zspx - bzmodel
    bzres = (zspx - bzmodel) / ezspx
    col = hexplot(
        ax,
        xspx,
        yspx,
        bzmodel,
        scale=SCALE * args.rescale,
        cmap=args.palette,
        vmin=args.zcut1,
        vmax=args.zcut2,
        alpha=0.99,
        zorder=20,
    )
    cb = plt.colorbar(col)
    cb.set_label(args.units, fontsize=19)
    ax.coords.grid(color=args.title_color, alpha=args.alpha, linestyle="dashed")
    plt.title(
        "Radial velocity model",
        x=0.04,
        y=0.9,
        zorder=30,
        loc="left",
        color=args.title_color,
    )
    ax.coords["dec"].set_axislabel("DEC (J2000)")
    ax.coords["ra"].set_axislabel("RA (J2000)")
    if args.output_plot is None:
        plt.show()

    # Plot velocity residuals only in valid pixels/spaxels
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs2)
    ax.tick_params(axis="both", labelsize=14)
    ax.minorticks_on()
    ax.coords["ra"].set_major_formatter("hh:mm:ss.s")
    ax.set_xlim([-6.5, 6.5])
    ax.set_ylim([-6.5, 6.5])
    col = hexplot(
        ax,
        xspx,
        yspx,
        bnnres,
        scale=SCALE * args.rescale,
        cmap=args.palette,
        vmin=-30.0,
        vmax=30.0,
        alpha=0.99,
        zorder=20,
    )
    cb = plt.colorbar(col)
    cb.set_label(args.units, fontsize=19)
    ax.coords.grid(color=args.title_color, alpha=args.alpha, linestyle="dashed")
    plt.title(
        "Radial velocity residuals",
        x=0.04,
        y=0.9,
        zorder=30,
        loc="left",
        color=args.title_color,
    )
    ax.coords["dec"].set_axislabel("DEC (J2000)")
    ax.coords["ra"].set_axislabel("RA (J2000)")
    if args.output_plot is None:
        plt.show()

    # Plot entire velocity model
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs2)
    ax.tick_params(axis="both", labelsize=14)
    ax.minorticks_on()
    ax.coords["ra"].set_major_formatter("hh:mm:ss.s")
    ax.set_xlim([-6.5, 6.5])
    ax.set_ylim([-6.5, 6.5])
    print("Best model in all spaxels")
    bzmodel = vfunc(fitted_p_v, x, y, z)
    bnnres = z - bzmodel
    bzres = (z - bzmodel) / ez
    col = hexplot(
        ax,
        x,
        y,
        bzmodel,
        scale=SCALE * args.rescale,
        cmap=args.palette,
        vmin=args.zcut1,
        vmax=args.zcut2,
        alpha=0.99,
        zorder=20,
    )
    cb = plt.colorbar(col)
    cb.set_label(args.units, fontsize=19)
    ax.coords.grid(color=args.title_color, alpha=args.alpha, linestyle="dashed")
    plt.title(
        "Radial velocity model",
        x=0.04,
        y=0.9,
        zorder=30,
        loc="left",
        color=args.title_color,
    )
    ax.coords["dec"].set_axislabel("DEC (J2000)")
    ax.coords["ra"].set_axislabel("RA (J2000)")

    if args.output_plot is None:
        plt.show()
    else:
        pdf = matplotlib.backends.backend_pdf.PdfPages(args.output_plot)
        for fig in range(1, figure().number):  # will open an empty extra figure :(
            pdf.savefig(fig)
        pdf.close()
