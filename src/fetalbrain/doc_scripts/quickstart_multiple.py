from pathlib import Path
import torch
from fetalbrain.utils import read_image, write_image
from fetalbrain.alignment.align import load_alignment_model, align_to_atlas, prepare_scan
from fetalbrain.structural_segmentation.subcortical_segm import load_segmentation_model, segment_subcortical
from fetalbrain.tedsnet_multi.teds_multi_segm import (
    load_tedsmulti_model,
    segment_tedsall,
    load_sidedetector_model,
    detect_side,
)
from fetalbrain.brain_extraction.extract import extract_brain, load_brainextraction_model
from fetalbrain.alignment.kelluwen_transforms import apply_affine
from fetalbrain.model_paths import EXAMPLE_IMAGE_PATH


# Load the models once
align_model = load_alignment_model()
subc_segmmodel = load_segmentation_model()
teds_multimodel = load_tedsmulti_model()
side_detectormodel = load_sidedetector_model()
brainextraction_model = load_brainextraction_model()

# whether to do connected component analysis for subcortical segm
connected_component = True

# Loop over all scans (just one here as example)
example_scan, _ = read_image(EXAMPLE_IMAGE_PATH)
torch_scan = prepare_scan(example_scan)

# Start with alignment to atlas space
aligned_scan, params = align_to_atlas(torch_scan, align_model, scale=False)

# Perform subcortical segmentation
subc_segm, subc_keys = segment_subcortical(aligned_scan, subc_segmmodel, connected_component=True)

# Perform segmentation with multi structure tedsnet
side, prob_side = detect_side(aligned_scan, side_detectormodel)
allstructure_segm, multi_keys = segment_tedsall(aligned_scan, teds_multimodel, side=side)

# perform whole brian extraction (i.e. brain masking)
brain_mask, brain_key = extract_brain(aligned_scan, brainextraction_model)

# Write out the results in the aligned orientation
savefolder = Path("results")
savefolder.mkdir(exist_ok=True)
write_image(savefolder / "aligned_scan.nii.gz", aligned_scan.squeeze().numpy())
write_image(savefolder / "subcortical_segm.nii.gz", subc_segm.squeeze(), segm=True)
write_image(savefolder / "allstructure_segm.nii.gz", allstructure_segm.squeeze(), segm=True)
write_image(savefolder / "brain_mask.nii.gz", brain_mask.squeeze().numpy(), segm=True)


# to create segmentations in the original orientation
aligned_scan, params, affine = align_to_atlas(torch_scan, align_model, scale=False, return_affine=True)
subc_segm_original = apply_affine(torch.from_numpy(subc_segm).unsqueeze(0), affine.inverse(), type_resampling="nearest")

# Write out the results in the original orientation
write_image(savefolder / "original_scan.nii.gz", example_scan)
write_image(savefolder / "subcortical_segm_original.nii.gz", subc_segm_original.squeeze().numpy(),  # type: ignore
            segm=True)
