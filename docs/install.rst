.. _install:

Installing
==========

The Genome package can be installed in any computer running Python 3 or above.
Genome does require
`Theano Python computational libraries <https://github.com/Theano/theano>`_
to be installed first.
It is recommended that users install a stable branch of Theano either through
pip, conda or directly from the Github project page.

Users should also be familiar with using Git, though that is not a requirement.
The latest code should always be pulled from the Github project repository.

.. tip::
    Users can install Theano directly to their computer by the ``git clone`` and ``pip3 install`` command::

        git clone https://github.com/Theano/Theano.git
        pip3 install --user Theano

Requirements
------------

* Python_ 3.5 or later
* Numpy_ 1.15 or later (base N-dimensional array package)
* Scipy_ 1.2.0 or later (library for scientific computing)
* Theano_ 1.0.0 or later

.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _Theano: http://deeplearning.net/software/theano/


Installing using pip (recommended)
----------------------------------

To install the latest version of Genome, download the repository from the
command line::

    git clone https://github.com/mwong009/Genome.git

then run pip (assuming no admin privileges)::

    pip3 install --user Genome

``--user`` option installs the library the local dist-packages.

To update to the latest version, go into the project directory and execute
``git pull origin master``.

You can also install from the Github repo directly (without cloning). Use the
``--user`` flag if necessary::

    pip3 install -U --user https://github.com/mwong009/Genome/archive/master.tar.gz

To uninstall the package::

    pip3 uninstall genome -y

Virtual environments
--------------------

Use a virtual environment to manage the dependencies for your project, both in
development and in testing.

Virtual environments are independent groups of Python libraries, one for each
project. Packages installed for one project will not affect other projects or
the operating system’s packages.

.. topic:: Create an environment

    Create a project folder and a ``venv`` folder::

        mkdir myproject
        cd myproject
        python3 -m venv venv

.. topic:: Activate the environment

    Activate the corresponding enviornment from the project folder::

        . venv/bin/activate

.. topic:: Installing Genome in venv

    Within the activated virtual environment, install Genone.
    You can install it directly from the the master branch::

        pip3 install -U https://github.com/mwong009/Genome/archive/master.tar.gz

Test your installation
----------------------

To test if the package has been installed, run the following command from a Python environment, e.g. Jupyter_::

    >>> import genome
    >>> genome.version()

If everything is installed correctly, there will not be any error messages.
If you encounter error during the installation process, report the output of the tests under the
`issue tracker <https://github.com/mwong009/genome/issues>`_.

.. _Jupyter: https://jupyter.org/

Basic Usage
-----------

TODO
