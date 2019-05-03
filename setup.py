#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import path
from setuptools import setup, find_packages


# Package meta-data.
NAME = 'Genome'
DESCRIPTION = 'The Genome project.'
URL = 'https://github.com/mwong009/genome'
EMAIL = ''
AUTHOR = 'LiTrans'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = ''
REQUIRED = ['Theano>=1.0.0']
EXTRAS = {
    # 'extra feature': [],
}

# ------------------------------------------------

here = path.abspath(path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning'
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
)
