#
# Copyright 2019-2020 Universidad Complutense de Madrid
#
# This file is part of Megara Tools
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import numpy as np

import matplotlib.pyplot as plt


def axvlines(xs, **plot_kwargs):
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = plt.gca().get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot


def gaussfunc(paramsin,x):
    amp=paramsin['amp'].value
    center=paramsin['center'].value
    sigma=paramsin['sigma'].value
    g1=(x-center)/sigma
    gaustot=amp*np.exp(-.5*g1**2)
    return gaustot


def gauss2func(paramsin,x):
    amp1=paramsin['amp1'].value
    amp2=paramsin['amp2'].value
    center1=paramsin['center1'].value
    center2=paramsin['center2'].value
    sigma1=paramsin['sigma1'].value
    sigma2=paramsin['sigma2'].value
    g1=(x-center1)/sigma1
    g2=(x-center2)/sigma2
    gaus1=amp1*np.exp(-.5*g1**2)
    gaus2=amp2*np.exp(-.5*g2**2)
    gaustot_2g=(gaus1+gaus2)
    return gaustot_2g


def gaussfunc_gh(paramsin,x):
    amp=paramsin['amp'].value
    center=paramsin['center'].value
    sigma=paramsin['sigma'].value
    c1=-np.sqrt(3)
    c2=-np.sqrt(6)
    c3=2/np.sqrt(3)
    c4=np.sqrt(6)/3
    c5=np.sqrt(6)/4
    skew=paramsin['skew'].value
    kurt=paramsin['kurt'].value
    g1=(x-center)/sigma
    gaustot_gh=amp*np.exp(-.5*g1**2)*(1+skew*(c1*g1+c3*g1**3)+kurt*(c5+c2*g1**2+c4*(g1**4)))
    return gaustot_gh


def linfunc(paramsin,x):
    slope=paramsin['slope'].value
    yord=paramsin['yord'].value
    lintot=slope*np.array(x)+yord
    return lintot