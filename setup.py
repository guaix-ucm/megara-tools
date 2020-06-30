
import pkg_resources
import setuptools

pkg_resources.require('setuptools>=39.2')
setuptools.setup(use_scm_version=True)
