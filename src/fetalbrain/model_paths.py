from pathlib import Path

MODEL_WEIGHTS_FOLDER = Path("src/model_weights")
BAN_MODEL_PATH = MODEL_WEIGHTS_FOLDER / "alignment" / "fBAN_modelweights.pt"
SEGM_MODEL_PATH = MODEL_WEIGHTS_FOLDER / "subc_segmentation" / "subc_segm.tar"
TEDS_MULTI_MODEL_PATH = MODEL_WEIGHTS_FOLDER / 'teds_segmentation' / 'finalmodel_multistructure.pt'
SIDE_DETECTOR_MODEL_PATH = MODEL_WEIGHTS_FOLDER / 'teds_segmentation' / 'FinalModel_sidedetection.pt'
PRIOR_SHAPE_PATH = MODEL_WEIGHTS_FOLDER / 'teds_segmentation' / '26wks_AllLabels.mha'
