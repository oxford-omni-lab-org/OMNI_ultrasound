Getting started
===============

Installation
------------

**PyTorch**: Before installing the package, PyTorch needs to be installed which can be done
using the instructions on `PyTorch <https://pytorch.org/get-started/locally/>`_.


**FetalBrain package**: The fetalbrain package will be hosted on pip, but can currently be used by cloning the 
repository locally with::

    $ git clone git@github.com:oxford-omni-lab-org/OMNI_ultrasound.git

User installation
^^^^^^^^^^^^^^^^^

The package can then be installed from the root of the repository with (This will install the package and all required dependencies)::

    $ pip install .

Then download the model weights using the provided command line downloader (this only downloads the model weights if that folder
does not exists yet)::

    $ ftlbr_download_modelweights

To overwrite any existing downloads use::
    
    $ ftlbr_download_modelweights --force


After installation and download of the model weights, the 
package can simply be used in Python with::

    import fetalbrain

Development installation 
^^^^^^^^^^^^^^^^^^^^^^^^

To install the package for development, use::

    $ pip install -e ".[all]"

This will install the package as editable, meaning that any changes to the package
are immediately available without having to reinstall the package. The [all] option
installs all dependencies needed for development, testing and documentation
(as specified in the setup.cfg file). Also rerun the downlaoder to load the model weights
in the correct location. 

To perform testing, run the downloader with the additional argument --testdata to download the test data::
    
    $ ftlbr_download_modelweights --testdata



Minimal Example
---------------

The following example shows a minimal example of using the whole pipeline to align, 
and segment a fetal brain scan. 

   
.. literalinclude:: ../../doc_scripts/quickstart.py


Advanced Example
----------------

To run the pipeline for multiple scans and with more flexibility, it is recommended to
use the individual pipeline functions rather than the wrapper functions. This ensures that
the models are not reloaded for each scan. The following example demonstrates this for a single
example. 

.. literalinclude:: ../../doc_scripts/quickstart_multiple.py










