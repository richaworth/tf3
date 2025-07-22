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
    

def main(path_original_images: Path = Path("C:/data/tf3/imagesTr"),
         path_original_labels: Path = Path("C:/data/tf3/labelsTr"), 
         path_dataset_json: Path = Path("C:/data/tf3/dataset.json"), 
         path_output_dir: Path = Path("C:/data/tf3"),
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
    with path_dataset_json.open("r") as j:
        metadata = dict(json.load(j))

    assert path_original_images.exists(), f"Data directory {path_original_images} is missing - can not continue."
    assert path_original_labels.exists(), f"Data directory {path_original_labels} is missing - can not continue."

    # Create reflected images and labels
    path_images_mirrored =  path_output_dir / "images_mirrored" 
    path_labels_mirrored =  path_output_dir / "labels_mirrored" 


    # TODO: reflect images and labels, converting left/right as necessary.

    # Gross labels - maxilla, mandible, lower teeth, upper teeth (for localisation)
    path_labels_gross = path_output_dir / "labelsTr_gross" 
    path_labels_gross.mkdir(exist_ok=True, parents=True)

    updated_gross_labels = {}
    gross_label_lookup = {}

    for k, v in metadata["labels"].items():
        if k == "Lower Jawbone":
            updated_gross_labels[k] = 1
            gross_label_lookup[1] = [v]
        if k == "Upper Jawbone":
            updated_gross_labels[k] = 2
            gross_label_lookup[2] = [v]
        if "Upper" in k and ("Molar" in k or "Incisor" in k or "Canine" in k or "Premolar" in k):
            updated_gross_labels[k] = 3
            gross_label_lookup[3] = [v]
        if "Lower" in k and ("Molar" in k or "Incisor" in k or "Canine" in k or "Premolar" in k):
            updated_gross_labels[k] = 4
            gross_label_lookup[4] = [v]
        else:
            continue

    with (path_labels_gross / "updated_labels.json").open("w") as j:
        json.dump(updated_gross_labels, j)

    for path_in in tqdm(list(path_original_labels.glob(f"*{metadata['file_ending']}"))):
        replace_labels(path_in, path_labels_gross / path_in.name, gross_label_lookup, overwrite=overwrite) 

    # # Bony anatomy - Combine pulps and teeth into single shapes (intending to find pulps later)
    # # Setting labels as consecutive.

    # path_labels_bony = path_output_dir / "labelsTr_bony"
    # path_labels_bony.mkdir(exist_ok=True, parents=True)
    
    # updated_bony_labels = {}
    # bony_label_lookup = {}

    # i = 0

    # for k, v in metadata["labels"].items():
    #     if "Sinus" in k or "Canal" in k or "Pharynx" in k:
    #         continue

    #     if "Pulp" in k:
    #         tooth = k.removesuffix(" Pulp")
    #         bony_label_lookup[updated_bony_labels[tooth]].append(v)
    #     else:
    #         updated_bony_labels[k] = i
    #         bony_label_lookup[i] = [v]
    #         i = i + 1

    # with (path_labels_bony / "updated_labels.json").open("w") as j:
    #     json.dump(updated_bony_labels, j)

    # for  path_in in tqdm(list(path_original_labels.glob(f"*{metadata['file_ending']}"))):
    #     replace_labels(path_in, path_labels_bony / path_in.name, bony_label_lookup, overwrite=overwrite) 


if __name__ == "__main__":
    main()
