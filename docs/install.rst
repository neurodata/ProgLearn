Install
=======

Below we assume you have the default Python environment already configured on
your computer and you intend to install ``ProgLearn`` inside of it.  If you want to
create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. We
also highly recommend conda. For instructions to install this, please look
at
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

First, make sure you have the latest version of ``pip`` (the Python package
manager) installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install from PyPi
-----------------
Install the current release of ``ProgLearn`` from the Terminal with ``pip``::

    $ pip install proglearn

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade proglearn

If you do not have permission to install software systemwide, you can install
into your user directory using the ``--user`` flag::

    $ pip install --user proglearn

Install from Github
-------------------
You can manually download ``ProgLearn`` by cloning the git repo master version and
running the ``setup.py`` file. That is, unzip the compressed package folder
and run the following from the top-level source directory using the Terminal::

    $ git clone https://github.com/neurodata/proglearn
    $ cd proglearn
    $ python3 setup.py install

Or, alternatively, you can use ``pip``::

    $ git clone https://github.com/neurodata/proglearn
    $ cd proglearn
    $ pip install .

The installation process takes about 4 seconds in a typical desktop computer.

If the installation is done in a new Python virtual environment, it takes about 1
minute and 35 seconds.

Python package dependencies
---------------------------
``proglearn`` requires the following packages:

- tensorflow>=1.19.0
- scikit-learn>=0.22.0
- scipy==1.4.1
- joblib>=0.14.1
- numpy==1.19.2

Hardware requirements
---------------------
``ProgLearn`` package requires only a standard computer with enough RAM to support
the in-memory operations. GPU's can speed up the networks which are powered by
tensorflow's backend.

OS Requirements
---------------
This package is supported for all major operating systems. The following
versions of operating systems was tested on CircleCI:

- **Linux**: Ubuntu 16.04
- **macOS**: Mojave (10.14.1)
- **Windows**: 10

Testing
-------
``ProgLearn`` uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.
