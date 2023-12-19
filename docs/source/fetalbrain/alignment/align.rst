Alignment
==========================

This module contains the main functions for aligning the scans.
A single scan be aligned using the :func:`align_scan` function, which is a wrapper that loads the alignment model,
prepares the scan into pytorch and computes and applies the alignment transformation. The alignment can be
applied without scaling (i.e. preserving the size of the brain) or with scaling (i.e. scaling all images to the
same reference brain size at 30GWs)::

   dummy_scan = np.random.rand(160, 160, 160)
   aligned_scan, params = align_scan(dummy_scan, scale=False, to_atlas=True)
   aligned_scan_scaled, params = align_scan(dummy_scan, scale=True)

For aligning a large number of scans, it is recommended to access the functions :func:`load_alignment_model`, :func:`prepare_scan` and the
:func:`align_to_atlas` functions directly so that the alignment model is not reloaded for the alignment of each scan.
For example as follows::

   model = load_alignment_model()
   dummy_scan = np.random.rand(160, 160, 160)
   torch_scan = prepare_scan(dummy_scan)
   aligned_scan, params = align_to_atlas(torch_scan, model, scale = False)

The :func:`align_to_atlas` function can also process batches of data (i.e. multiple scans at once), which can be useful
to speed up analysis. More advanced examples can be found in the Example Gallery.

Module functions
----------------

.. automodule:: fetalbrain.alignment.align
   :members:
   :undoc-members:
   :show-inheritance:
