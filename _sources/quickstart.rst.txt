Getting started
===============

Installation
------------

**PyTorch**: Before installing the package, PyTorch needs to be installed which can be done
using the instructions on `PyTorch <https://pytorch.org/get-started/locally/>`_.

**Git-LFS**: It is also required to install git-lfs so that the model weights are correctly pulled from the Github. 
Pulling the repo without git-lfs installed will work, but the model weight files will then not be 
downloaded correctly (throwing pytorch errors when loading them). More information on git-lfs can
be found at `Git-lfs <https://git-lfs.com/>`_.

Git-lfs can be installed with::

    $ yay install git-lfs # for arch linux 
    $ brew install git-lfs # for mac

followed by local installation with::

    $ git lfs install

**FetalBrain package**: The fetalbrain package will be hosted on pip, but can currently be used by cloning the 
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


