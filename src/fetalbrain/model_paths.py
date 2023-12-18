from pathlib import Path

MODEL_WEIGHTS_FOLDER = Path(__file__).parent.parent / Path("model_weights")
BAN_MODEL_PATH = MODEL_WEIGHTS_FOLDER / "alignment" / "fBAN_modelweights.pt"
SEGM_MODEL_PATH = MODEL_WEIGHTS_FOLDER / "subc_segmentation" / "subc_segm.tar"
TEDS_MULTI_MODEL_PATH = MODEL_WEIGHTS_FOLDER / 'teds_segmentation' / 'finalmodel_multistructure.pt'
SIDE_DETECTOR_MODEL_PATH = MODEL_WEIGHTS_FOLDER / 'teds_segmentation' / 'FinalModel_sidedetection.pt'
PRIOR_SHAPE_PATH = MODEL_WEIGHTS_FOLDER / 'teds_segmentation' / '26wks_AllLabels.mha'
BRAIN_EXTRACTION_MODEL_PATH = MODEL_WEIGHTS_FOLDER / 'brain_extraction' / 'finalmodel.pt'
