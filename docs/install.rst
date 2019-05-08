.. _install:

Installing
==========

The Genome package can be installed in any computer running Python 3 or above.
Genome does require `Theano <https://github.com/Theano/theano>`_ to be installed first.
It is recommended that users install a stable branch of Theano either through
pip, conda or directly from the Github trunk.

Users should also be familiar with using Git version control.
The latest code can always be pulled from the Github project repository.

.. tip::
    Users can install Theano directly to their computer by the ``git clone`` and ``pip3 install`` command::

        $ git clone https://github.com/Theano/Theano.git
        $ pip3 install --user Theano

Requirements
------------

* Python_ 3.5 or later
* Numpy_ 1.15 or later (base N-dimensional array package)
* Scipy_ 1.2.0 or later (library for scientific computing)

.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/


Installing using pip (recommended)
----------------------------------

To install the latest version of Genome, download the repository from the
command line::

    $ git clone https://github.com/mwong009/Genome.git

then run pip::

    $ pip3 install --user Genome

The ``--user`` option installs the library the local dist-packages.

To update to the latest version, go into the project directory and execute
``git pull origin master``.

To uninstall the package::

    $ pip3 uninstall genome -y

Test your installation
----------------------

To test if the package has been installed, run the following command from a Python environment, e.g. Jupyter_::

    >>> import genome
    >>> genome.version()

If everything is installed correctly, there will not be any error messages.
If you encounter error during the installation process, report the output of the tests under the
`issue tracker <https://github.com/mwong009/genome/issues>`_.

.. _Jupyter: https://jupyter.org/
