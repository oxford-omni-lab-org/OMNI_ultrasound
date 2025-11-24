from pathlib import Path
import torch
from fetalbrain.utils import read_image, write_image
from fetalbrain.alignment.align import load_alignment_model, align_to_atlas, prepare_scan
from fetalbrain.tedsnet_multi.teds_multi_segm import (
    load_tedsmulti_model,
    segment_tedsall,
    load_sidedetector_model,
    detect_side,
)
from fetalbrain.brain_extraction.extract import extract_brain, load_brainextraction_model
from fetalbrain.model_paths import EXAMPLE_IMAGE_PATH


# Load the models once
align_model = load_alignment_model()
teds_multimodel = load_tedsmulti_model()
side_detectormodel = load_sidedetector_model()
brainextraction_model = load_brainextraction_model()

# ------------------------------------------------
# Loop over all scans (just one here as example)
# Here you can change EXAMPLE_IMAGE_PATH to your 
# image path and/or loop through your directory 
# for volumes!
example_scan, _ = read_image(EXAMPLE_IMAGE_PATH) 
# ------------------------------------------------

torch_scan = prepare_scan(example_scan)

# Start with alignment to atlas space
aligned_scan, params = align_to_atlas(torch_scan, align_model, scale=False)

# Perform segmentation with multi structure tedsnet
side, prob_side = detect_side(aligned_scan, side_detectormodel)
allstructure_segm, multi_keys = segment_tedsall(aligned_scan, teds_multimodel, side=side)

# perform whole brian extraction (i.e. brain masking)
brain_mask, brain_key = extract_brain(aligned_scan, brainextraction_model)

# ------------------------------------------------
# Write out the results in the aligned orientation
# you can change this to where you want your results saved!
savefolder = Path("results")
# ------------------------------------------------

savefolder.mkdir(exist_ok=True)
write_image(savefolder / "aligned_scan.nii.gz", aligned_scan.squeeze().numpy())
write_image(savefolder / "allstructure_segm.nii.gz", allstructure_segm.squeeze(), segm=True)
write_image(savefolder / "brain_mask.nii.gz", brain_mask.squeeze().numpy(), segm=True)


