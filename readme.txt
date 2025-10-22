This repository is still actively updated
Any feedback on usability/documentation etc., is also very welcome! 

Documentation can be found on: https://oxford-omni-lab-org.github.io/OMNI_ultrasound/

NB: this packages required Python 3.11. This can be achieved by creating a conda environement: e.g: conda create -n fetalbrain-env python=3.11

NB: All volumes must be resampled to 0.6mm3 before applying this pipeline


[Future Plans]
- Documentation
    - Include documentation for TEDS-Net
    - Ensure all docstrings are accurate (i.e. default settings etc.)

- Code
    - Ensure consistent use of torch / numpy across the modules
    - Support GPU and CPU access (this is currently only somewhat supported)
    - Include automated testing of the scripts in the doc_scripts folder (these are copied into the documentation, and it would be good that it is verified that these are correct.)

- Package
    - Publish the package on pip
    - Test with which versions of all dependencies it works and with which python versions

- Command Line Access
    - Make command line interface so that a whole folder of scans can be automatically processed without having to use any Python code.

- Interactive visualisation of results (i.e. segmentations)
    - Look into building a 3D Slicer extension 
    - Interactive GUI can also be built with Plotly



