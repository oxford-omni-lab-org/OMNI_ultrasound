Alignment Package
=====================

This package contains code for aligning the fetal brain image to a common coordinate
system. Due to differences in approach, the images can be aligned to two different reference
spaces (default is the atlas space):

* **The atlas space**. This is the default coordinate system, and matches the orientation of the 
  ultrasound fetal brain atlas `(Namburete at al. Nature 2023) <https://www.nature.com/articles/s41586-023-06630-3>`_. 
  All segmentation methods in this package expect the image to be aligned to this space without scaling. Plotting the midplanes of each
  axis of the 3D images aligned to the atlas space will look like this:

  .. image:: ../../example_images/aligned_to_atlas.png
    :alt: image aligned to the atlas coordinate system
    :width: 500px


* **The BEAN space.** This is the coordinate system used by the BEAN alignment network. This
  coordinate system is used internally by the alignment network, but it should **not** be used 
  for any downstream analysis with the segmentation networks. Plotting the midplanes of each
  axis of the 3D image aligned to the BEAN space will look like this:

  .. image:: ../../example_images/aligned_to_BEAN.png
    :alt: image aligned to the BEAN coordinate system
    :width: 500px

Note that the difference between the two coordinate systems is not only a permutation of the
axes, but also includes translation + rotation. The transformation matrix going from the
BEAN to the atlas space can be obtained from :func:`fetalbrain.alignment.align._get_atlastransform()`,
and is defined for the scaled+aligned versions of the images only. Access to this function is only required 
for package development or advanced use of the package. 

The above images can be created using the following code:

.. literalinclude:: ../../../../src/fetalbrain/doc_scripts/align_demo.py
  :language: python

For more information on the use of the alignment functions see :doc:`/fetalbrain/alignment/align`.

Credits 
-------

The code in the alignment package has been primarily developed by Felipe Moser. A preliminary version of the methods have been described in a NeuroImage paper:

`Felipe Moser, Ruobing Huang, Bartłomiej W Papież, Ana IL Namburete, INTERGROWTH-21st Consortium. 
BEAN: Brain Extraction and Alignment Network for 3D Fetal Neurosonography. NeuroImage (Sept 2022). 
<https://www.sciencedirect.com/science/article/pii/S1053811922004608>`_

The implementation of the rotation parameters has since been updated to use quaternions instead of Euler angles. 
A detailed description of this implementation can be found in the PhD thesis of Felipe Moser:

<add thesis link here>



Submodules
----------

.. toctree::
   :maxdepth: 4

   align
   fBAN_v1
   kelluwen_transforms

Module contents
---------------

.. automodule:: fetalbrain.alignment
   :members:
   :undoc-members:
   :show-inheritance:
