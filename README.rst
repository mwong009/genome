========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |travis| |codecov|
    * - package
      - | |commits-since|

.. |travis| image:: https://api.travis-ci.com/mwong009/genome.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/mwong009/genome

.. |codecov| image:: https://codecov.io/gh/mwong009/genome/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/mwong009/genome

.. |version| image:: https://img.shields.io/pypi/v/genome.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/genome

.. |wheel| image:: https://img.shields.io/pypi/wheel/genome.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/genome

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/genome.svg
    :alt: Supported versions
    :target: https://pypi.org/project/genome

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/genome.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/genome

.. |commits-since| image:: https://img.shields.io/github/commits-since/mwong009/genome/v0.1.1.svg
    :alt: Commits since latest release
    :target: https://github.com/mwong009/genome/compare/v0.1.1...master



.. end-badges

Deep learning library for discrete choice modelling

* Free software: MIT license

Installation
============

::

    pip install genome

You can also install the in-development version with::

    pip install https://github.com/mwong009/genome/archive/master.zip


Documentation
=============


To use the project:

.. code-block:: python

    import genome


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
