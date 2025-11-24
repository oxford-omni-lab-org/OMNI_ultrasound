
import numpy as np
from pathlib import Path
from fetalbrain.utils import read_image


def ComputeVol(allstructure_segm, brain_mask,voxel_dim=0.6,N=160):
 
    assert np.shape(allstructure_segm) ==np.shape(np.zeros((N,N,N)))
    assert np.shape(brain_mask) ==np.shape(np.zeros((N,N,N)))

    struc_vol_dict = {} 

    # compute the TBV:
    volume = np.count_nonzero(brain_mask)*(voxel_dim**3)
    struc_vol_dict["TBV"] =volume

    # Index of each of the structures
    key_maps = {
        "CoP": 1,
        "CSP": 2,
        "CB": 3,
        "ChP": 4,
        "LV": 5,
        "DGM": 6,
        "Th": 7,
        "BS": 8,
        "WM": 9,
        "FH": 10,
    }


    for keys in key_maps.keys():
        """ Loop through the structures and measure volume
        """
        struc = np.where(allstructure_segm==key_maps[keys],1,0)
        volume = np.count_nonzero(struc)*(voxel_dim**3)
        struc_vol_dict[keys+"V"] =volume
        
    print(struc_vol_dict)



if __name__ == '__main__':
    """ Compute the volume measures of the segmented structures


    """

    savefolder = Path("results")
    allstructure_segm,h = read_image(savefolder / "allstructure_segm.nii.gz")
    brain_mask,h = read_image(savefolder / "brain_mask.nii.gz")

    # Measure the volume of each structire:
    SVol = ComputeVol(allstructure_segm, brain_mask)

