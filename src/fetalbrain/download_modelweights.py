""" This scripts donwloads the model weights into the correct folder. 
After installation of the package it can be run using from command line with:
    >>> ftlbr_download_modelweights
"""
from urllib.request import urlretrieve
from fetalbrain.model_paths import MODEL_WEIGHTS_FOLDER
import zipfile
import warnings
import os


def download_modelweights(force: bool = False) -> None:
    download_folder = MODEL_WEIGHTS_FOLDER.parent / "model_weights"

    if download_folder.exists() and not force:
        warnings.warn("Model weights already downloaded, skipping download")
        return

    else:
        url = "https://github.com/oxford-omni-lab-org/OMNI_ultrasound/releases/download/prerelease/model_weights.zip"
        path, headers = urlretrieve(url, MODEL_WEIGHTS_FOLDER.parent / "model_weights.zip")

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(MODEL_WEIGHTS_FOLDER.parent)

        # remove the zip file itself (as we have extracted it)
        os.remove(path)

        print('Model weights downloaded to {}'.format(MODEL_WEIGHTS_FOLDER))

    return


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--force", action="store_true", help="Force download and overwrite existing files")
    args = parser.parse_args()

    download_modelweights(force=args.force)


if __name__ == "__main__":
    main()
