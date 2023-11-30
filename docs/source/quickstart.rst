Getting started
===============

Installation
------------

Before installing the package, PyTorch needs to be installed which can be done
using the instructions on `PyTorch <https://pytorch.org/get-started/locally/>`_.

The fetalbrain package will be hosted on pip, but can currently be used by cloning the 
repository locally with::

    $ git clone git@github.com:lindehesse/OMNI_ultrasound.git


User installation
^^^^^^^^^^^^^^^^^

The package can then be installed from the root of the repository with::

    $ pip install .

This will install the package and all required dependencies. After installation, the 
package can simply be used in Python with::

    import fetalbrain

Development installation 
^^^^^^^^^^^^^^^^^^^^^^^^

To install the package for development, use::

    $ pip install -e .[all]

This will install the package as editable, meaning that any changes to the package
are immediately available without having to reinstall the package. The [all] option
installs all dependencies needed for development, testing, plotting and documentation
(as specified in the setup.cfg file).



Minimal Example
---------------

The following example shows a minimal example of using the whole pipeline to align, 
and segment a fetal brain scan. 

   
.. literalinclude:: ../../src/fetalbrain/quickstart.py


