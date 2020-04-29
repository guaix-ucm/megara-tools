
from setuptools import find_packages
from setuptools import setup


setup(
    name='megaratools',
    version='0.1.dev0',
    author='Armando Gil de Paz',
    author_email='agil@fis.ucm.es',
    maintainer='Sergio Pascual',
    maintainer_email='sergiopr@fis.ucm.es',
    url='https://github.com/guaix-ucm/megarad-tools',
    license='GPLv3',
    description='Analysis and visualization of data from MEGARA instrument',
    packages=find_packages(),
    install_requires=[
        'setuptools>=36.2.1',
        'numpy',
        'astropy',
        'scipy',
        'matplotlib',
        'lmfit',
        'numina',
        'megaradrp',
    ],
    entry_points={
        'console_scripts': [
          'megaratools-analyze_rss = megaratools.analyze_rss:main',
          'megaratools-analyze_spectrum = megaratools.analyze_spectrum:main',
          'megaratools-diffuse_light = megaratools.diffuse_light:main',
          'megaratools-extract_spectrum = megaratools.extract_spectrum:main',
          'megaratools-plot_spectrum = megaratools.plot_spectrum:main',
          'megaratools-rss_arith = megaratools.rss_arith:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'Development Status :: 3 - Alpha',
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    long_description=open('README.md').read()
)
