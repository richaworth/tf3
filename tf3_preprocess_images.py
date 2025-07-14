import json
import logging
from joblib import Parallel, delayed
import nibabel
import numpy as np

from pathlib import Path

from tqdm import tqdm

def replace_labels(path_original_label: Path, path_output_label: Path, label_lookup: dict, overwrite: bool = False):
    """
    Load nifti format label image, replace labels according to the lookup, save image.

    Args:
        path_original_label (Path): Path to original image directory.
        path_output_label (Path): Path to original label image directory.
        label_lookup (dict): Lookup dictionary for combining/renaming labels. Any label not in the lookup will be
            set as background (0). E.g. {1: [1, 2], 2: [3, 4]} will rename labels 1+2 as 1, and 3+4 as 2.
        overwrite (bool): Overwrite existing output files?

    Returns:
        onnx file for each trained model
    """
    if path_output_label.exists() and not overwrite:
        logging.info(f"{path_output_label} exists and is not to be overwritten - returning")
        return
    
    im = nibabel.load(path_original_label)

    im_data = im.get_fdata()
    im_data_out = np.empty(im_data.shape)

    for new, old in label_lookup.items():
        for o in old:
            logging.debug(f"Setting label {o} to {new}")
            im_data_out[:] = np.where(im_data == o, new, im_data_out[:])

    im_out = nibabel.Nifti1Image(im_data_out, im.affine, im.header)
    nibabel.save(im_out, path_output_label)
    

def main(path_original_images: Path = Path("C:/data/ToothFairy3/imagesTr"),
         path_original_labels: Path = Path("C:/data/ToothFairy3/labelsTr"), 
         path_dataset_json: Path = Path("C:/data/ToothFairy3/dataset.json"), 
         path_output_dir: Path = Path("C:/data/ToothFairy3"),
         overwrite: bool = False):
    """
    Run ToothFairy3 preprocessing:
     - Extract specific labels to construct test models

    Args:
        path_original_images (Path): Path to original image directory.
        path_original_labels (Path): Path to original label image directory.
        path_output_dir (Path): Path to output directory (also contains any interim data constructed during training).
        overwrite (bool): overwrite existing output files?

    Returns:
        onnx file for each trained model
    """
    logging.basicConfig(level=logging.INFO)

    # Get labels etc. from dataset json
    with path_dataset_json.open("r") as f:
        metadata = json.load(f)

    assert path_original_images.exists(), f"Data directory {path_original_images} is missing - can not continue."
    assert path_original_labels.exists(), f"Data directory {path_original_labels} is missing - can not continue."

    # TODO: reflect images and labels, converting left/right as necessary.

    # Prototype model - segment all bony anatomy as four labels - bone vs teeth, upper and lower.
    # Teeth are numbered 11-18, 21-28 etc. and pulps are numbered as +100.
    path_labels_bt = path_output_dir / "labelsTrBT"
    path_labels_bt.mkdir(exist_ok=True, parents=True)
    
    bones_teeth_lookup = {
        1: [1], # Lower jawbone
        2: [2], # Upper jawbone
        3: [i for i in range(11, 18)] + [i for i in range(21, 28)] + [i for i in range(111, 118)] + [i for i in range(121, 128)],
        4: [i for i in range(31, 38)] + [i for i in range(41, 48)] + [i for i in range(131, 138)] + [i for i in range(141, 148)],
    }

    Parallel(n_jobs=1)(delayed(replace_labels)(path_in, path_labels_bt / path_in.name, bones_teeth_lookup, overwrite) 
                       for path_in in tqdm(list(path_original_labels.glob(f"*{metadata['file_ending']}"))))


if __name__ == "__main__":
    main()
