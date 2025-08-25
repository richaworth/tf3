from copy import deepcopy
import json
import logging
import shutil
from joblib import Parallel, delayed
import nibabel
import nilearn.image
import numpy as np

from pathlib import Path

from scipy.ndimage import label, binary_dilation
from tqdm import tqdm

import SimpleITK as sitk

SD_LARGE = np.empty(shape=(200, 200, 200), dtype=float)
SD_LARGE[100, 100, 100] = 1
SD_LARGE = binary_dilation(SD_LARGE, iterations=3)

itk_image = sitk.GetImageFromArray(np.astype(SD_LARGE, int))
itk_smdm_out = sitk.SignedMaurerDistanceMap(itk_image, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)
SD_LARGE = sitk.GetArrayFromImage(itk_smdm_out)


def reflect_image(path_image_in, path_image_out, overwrite: bool = False):
    """
    Reflect image along X axis.

    Args:
        path_image_in (Path): Path to the input image
        path_image_out (Path): Path to the output image
        overwrite (bool, optional): Overwrite existing files?. Defaults to False.

    """
    if path_image_out.exists() and not overwrite:
        logging.debug(f"{path_image_out} exists and is not to be overwritten - returning")
        return

    im = nibabel.load(path_image_in)

    im_data =im.get_fdata()
    im_data_out = np.flip(np.astype(im_data, im.header.get_data_dtype()), axis=0)

    im_out = nibabel.Nifti1Image(im_data_out, im.affine, im.header, dtype=im.header.get_data_dtype())
    nibabel.save(im_out, path_image_out)

def replace_labels(path_input_label: Path, path_output_label: Path, label_lookup: dict, overwrite: bool = False):
    """
    Load nifti format label image, replace labels according to the lookup, save image.

    Args:
        path_original_label (Path): Path to original image directory.
        path_output_label (Path): Path to original label image directory.
        label_lookup (dict): Lookup dictionary for combining/renaming labels (old to new). Any label not in the lookup will be
            set as background (0). E.g. {1: 2, 2: 3} will replace labels 1 as 2, and 2 as 3.
        overwrite (bool): Overwrite existing output files?

    Returns:
        onnx file for each trained model
    """
    if path_output_label.exists() and not overwrite:
        logging.debug(f"{path_output_label} exists and is not to be overwritten - returning")
        return
    
    im = nibabel.load(path_input_label)

    im_data = im.get_fdata()
    im_data_out = np.empty(shape=im_data.shape, dtype=im.header.get_data_dtype())

    if 0 in label_lookup.keys() and label_lookup[0] == 0:
        label_lookup.pop(0)
        ignore_bg = True
    else:
        ignore_bg = False

    get_above = 0 if ignore_bg else -1

    for i, x in np.ndenumerate(im_data):
        if x > get_above and x in label_lookup.keys():
            im_data_out[i] = label_lookup[int(x)]

    im_out = nibabel.Nifti1Image(im_data_out, im.affine, im.header, dtype=im.header.get_data_dtype())
    nibabel.save(im_out, path_output_label)

def resample_image_and_label(path_image_in: Path, 
                             path_label_in: Path,
                             path_image_out: Path,
                             path_label_out: Path,
                             resolution: tuple[float] = (1, 1, 1), 
                             overwrite: bool = False):
    """
    Resample nifti image to given resolution and using the given interpolation method.

    Args:
        path_image_in (Path): Path to the input image.
        path_image_out (Path): Path to the output image.
        resolution (tuple, optional): Resolution to resample to. Defaults to (1, 1, 1).
        interpolation (str, optional): Interpolation method (continuous, linear or nearest). Defaults to "continuous".
        overwrite (bool, optional): Overwrite existing files? Defaults to False.
    """
    if path_image_out.exists() and not overwrite:
        logging.debug(f"{path_image_out} exists and is not to be overwritten - returning")
        return
    
    im = nibabel.load(path_image_in)
    lab = nibabel.load(path_label_in)
    
    target_affine = np.asarray([[resolution[0], 0, 0], [0, resolution[1], 0], [0, 0, resolution[2]]])

    im_out = nilearn.image.resample_img(im, target_affine=target_affine, interpolation="continuous", copy_header=True, force_resample=True)
    lab_out = nilearn.image.resample_to_img(lab, im, "nearest", force_resample=True, copy_header=True)
    
    nibabel.save(im_out, path_image_out)
    nibabel.save(lab_out, path_label_out)

def process_upper_lower_implants_crowns_bridges(path_input_label, path_output_label, overwrite: bool = False):
    """
    Set upper jaw implants to upper jaw bone value; set lower jaw implants to lower jaw value, and set upper jaw bridges/crowns to upper 
    teeth value; set lower jaw bridges/crowns to lower teeth value.

    Args:
        path_input_label (Path): Path to the input label image
        path_output_all_labels (Path): Path to the output label image
        overwrite (bool, optional): Overwrite existing files?. Defaults to False.
    """
    if path_output_label.exists() and not overwrite:
        logging.debug(f"{path_output_label} exists and is not to be overwritten - returning")
        return
    
    im = nibabel.load(path_input_label)

    im_data = im.get_fdata()
    
    if np.max(np.astype(im_data, int)) == 4:
        logging.info(f"{path_input_label.name} does not contain implants or bridges. Copied labelled image.")
        shutil.copy(path_input_label, path_output_label)
        return

    im_data_out = np.astype(im_data, int)

    # Labels
    # Lower jaw = 1, upper jaw = 2, Implant (in jaw bone) == 5
    lower_jaw = np.where(im_data_out == 1, 1, 0)
    upper_jaw = np.where(im_data_out == 2, 1, 0)
    
    if 5 in im_data_out:
        implants = np.where(im_data_out == 5, 1, 0)
        labelled_implants, n_implants = label(implants)

        for lab in range(1, n_implants + 1):
            implant = np.where(labelled_implants == lab, 1, 0)
            implant_dil = deepcopy(implant)

            while np.max(implant_dil + lower_jaw) == 1 and np.max(implant_dil + upper_jaw) == 1:
                implant_dil = binary_dilation(implant_dil)
                if np.max(implant_dil + lower_jaw) == 2:
                    im_data_out = np.where(implant, 1, im_data_out)
                elif np.max(implant_dil + upper_jaw) == 2:
                    im_data_out = np.where(implant, 2, im_data_out)

    if 6 in im_data_out:
        replacements = np.where(im_data_out == 6, 1, 0)
        labelled_replacements, n_replacements = label(replacements)

        for lab in range(1, n_replacements + 1):
            replacement = np.where(labelled_replacements == lab, 1, 0)
            replacement_dil = deepcopy(replacement)

            while np.max(replacement_dil + lower_jaw) == 1 and np.max(replacement_dil + upper_jaw) == 1:
                replacement_dil = binary_dilation(replacement_dil)
                if np.max(replacement_dil + lower_jaw) == 2:
                    im_data_out = np.where(replacement, 3, im_data_out)
                elif np.max(replacement_dil + upper_jaw) == 2:
                    im_data_out = np.where(replacement, 4, im_data_out)
    
    im_out = nibabel.Nifti1Image(im_data_out, im.affine, im.header, dtype=im.header.get_data_dtype())
    nibabel.save(im_out, path_output_label)


def per_tooth_image_mask_sd(path_input_image: Path, 
                            path_input_label: Path, 
                            path_per_tooth_image_dir: Path, 
                            path_per_tooth_label_dir: Path, 
                            path_sd_map_dir: Path, 
                            overwrite: bool = False):
    """
    Calculate per-tooth distance maps (for landmark detection), cropped images + cropped tooth/pulp segmentations + cropped distance maps.

    Args:
        path_input_image (Path): Path to input image. (CASE_ID is input_image stem)
        path_input_label (Path): Path to input label.
        path_per_tooth_image_dir (Path): Output directory for per tooth images (as {CASE_ID}_{LABEL}.nii.gz).
        path_per_tooth_label_dir (Path): Output directory for per tooth labels (as {CASE_ID}_{LABEL}.nii.gz) (0 == background, 1 == tooth, 2 == pulp).
        path_sd_map_dir (Path): Output directory for per tooth signed distance maps (as {CASE_ID}_{LABEL}.nii.gz). 
            {CASE_ID}_{LABEL}_cropped.nii.gz will also be created in this directory.
        roi_size (tuple[int], optional): _description_. Defaults to (96, 96, 96).
        overwrite (bool, optional): _description_. Defaults to False.
    """
    output_stem = path_input_image.name.split(".nii.gz")[0]
    
    im_label = nibabel.load(path_input_label)
    im_label_data = im_label.get_fdata()
    im_ct = nibabel.load(path_input_image)


    # Teeth are labelled 11-47, and related pulp == (tooth + 100)
    # For each - get centre of tooth, dilate to a small circle of ~1mm diameter (2 iterations), calculate distance from each voxel to this circle.
    # Additionally create a review image of all tooth centres.

    lower_teeth_mask = np.where(np.logical_and(im_label_data >= 31, im_label_data <= 48))
    upper_teeth_mask = np.where(np.logical_and(im_label_data >= 11, im_label_data <= 28))

    if lower_teeth_mask[0].size != 0:
        k0_lower, k1_lower = np.min(lower_teeth_mask[2]), np.max(lower_teeth_mask[2])
    else:
        k0_lower, k1_lower = None, None

    if upper_teeth_mask[0].size != 0:
        k0_upper, k1_upper = np.min(upper_teeth_mask[2]), np.max(upper_teeth_mask[2])
    else:
        k0_upper, k1_upper = None, None
    
    
    # Additionally, create a mask of all teeth so that a not-this-one approach can be used.
    # Include fillings, crowns, implants.
    all_targets = np.where(np.logical_and(im_label_data >= 7, im_label_data <= 48), 3, 0)

    for lab in range(11, 48):
        if lab in im_label_data:
            if lab < 29:
                k0, k1 = k0_upper, k1_upper
            else:
                k0, k1 = k0_lower, k1_lower

            path_tooth_ct_out = path_per_tooth_image_dir / f"{output_stem}_{lab}.nii.gz"
            path_tooth_labels_out = path_per_tooth_label_dir / f"{output_stem}_{lab}.nii.gz"
            path_tooth_sd_cropped_out = path_sd_map_dir / f"{output_stem}_{lab}_cropped.nii.gz"

            if not overwrite:
                if path_tooth_ct_out.exists() and path_tooth_labels_out.exists() and path_tooth_sd_cropped_out.exists():
                    logging.debug(f"{output_stem}_{lab} files exist and are not to be overwritten - skipping.")
                    continue

            mask = np.where(np.logical_or(im_label_data == lab, im_label_data == lab+100), 1, 0)
            mask = np.where(np.logical_or(im_label_data == lab, im_label_data == lab+100), 1, 0)

            arr = np.where(mask != 0)
            i0, i1, j0, j1, = np.min(arr[0]), np.max(arr[0]), np.min(arr[1]), np.max(arr[1])

            # Pad somewhat
            i0 = max(i0-10, 0)
            j0 = max(j0-10, 0)
            k0 = max(k0-20, 0)
            i1 = min(i1+10, im_label_data.shape[0])
            j1 = min(j1+10, im_label_data.shape[1])
            k1 = min(k1+20, im_label_data.shape[2])

            # Crop image to bounds of tooth
            if not path_tooth_sd_cropped_out.exists() or overwrite:
                im_ct_cropped = im_ct.slicer[i0:i1, j0:j1, k0:k1]
                nibabel.save(im_ct_cropped, path_tooth_ct_out)

                mask_tooth = np.where(im_label_data == lab, 1, 0)
                mask_tooth = np.where(im_label_data == lab+100, 2, mask_tooth)
                final_mask = np.where(mask_tooth > 0, mask_tooth, all_targets)
                
                nii_mask_tooth = nibabel.Nifti1Image(final_mask, im_label.affine, im_label.header, dtype=im_label.header.get_data_dtype())

                nii_mask_tooth_cropped = nii_mask_tooth.slicer[i0:i1, j0:j1, k0:k1]
                nibabel.save(nii_mask_tooth_cropped, path_tooth_labels_out)

                # SD image is 200 x 200 x 200
                sd_centre = (100, 100, 100)
                sd_x0 = sd_centre[0] - int((i1 - i0) / 2)
                sd_x1 = sd_x0 + (i1 - i0)
                sd_y0 = sd_centre[1] - int((j1 - j0) / 2)
                sd_y1 = sd_y0 + (j1 - j0)
                sd_z0 = sd_centre[2] - int((k1 - k0) / 2)
                sd_z1 = sd_z0 + (k1 - k0)

                sd_out = SD_LARGE[sd_x0:sd_x1, sd_y0:sd_y1, sd_z0:sd_z1]
                smdm = np.zeros_like(final_mask)
                smdm[i0:i1, j0:j1, k0:k1] = sd_out

                nii_sd_out = nibabel.Nifti1Image(smdm, im_label.affine, im_label.header, dtype=im_label.header.get_data_dtype())
                nii_sd_out_cropped = nii_sd_out.slicer[i0:i1, j0:j1, k0:k1]
                nibabel.save(nii_sd_out_cropped, path_tooth_sd_cropped_out)


    # if not (path_sd_map_dir / f"{output_stem}_all_landmarks.nii.gz").exists() or overwrite:
    #     if np.max(im_all_labels) > 0:
    #         im_out = nibabel.Nifti1Image(im_all_labels, im_label.affine, im_label.header, dtype=im_label.header.get_data_dtype())
    #         nibabel.save(im_out, path_sd_map_dir / f"{output_stem}_all_landmarks.nii.gz")


def main(path_original_images: Path = Path("C:/data/tf3/images_rolm"),
         path_original_labels: Path = Path("C:/data/tf3/labels_rolm"), 
         path_dataset_json: Path = Path("C:/data/tf3/dataset.json"), 
         path_output_dir: Path = Path("C:/data/tf3"),
         overwrite: bool = False):
    """
    Run ToothFairy3 preprocessing:
     - Create Right-or-Left-Mirrored (RoLM) versions of images and labels.
     - Create low resolution localisation/intialisation images.
     - Create per-tooth centre signed distance maps


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

    # Create reflected images and labels (ROLM - right or left mirrored)
    path_images_rolm =  path_output_dir / "images_rolm" 
    path_labels_rolm =  path_output_dir / "labels_rolm" 

    path_images_rolm.mkdir(exist_ok=True, parents=True)
    path_labels_rolm.mkdir(exist_ok=True, parents=True)

    case_ids = [x.name.split(".")[0] for x in list(path_original_labels.glob(f"*{metadata['file_ending']}"))]

    # Set up left/right label reversing
    label_lookup_rolm = {}
    for k, v in metadata["labels"].items():
        if "Left" in k:
            k_right = str(k).replace("Left", "Right")
            label_lookup_rolm[v] = metadata["labels"][k_right]

        elif "Right" in k:
            k_left = str(k).replace("Right", "Left")
            label_lookup_rolm[v] = metadata["labels"][k_left]
        
        else:
            label_lookup_rolm[v] = metadata["labels"][k]

    # # Set up localiser labels and folders
    # # Create reflected images and labels (ROLM - right or left mirrored)
    # path_images_loc =  path_output_dir / "images_localiser_rolm" 
    # path_labels_loc =  path_output_dir / "labels_localiser_rolm" 

    # path_images_loc.mkdir(exist_ok=True, parents=True)
    # path_labels_loc.mkdir(exist_ok=True, parents=True)

    # label_lookup_localisation = {}
    # for k, v in metadata["labels"].items():
    #     if k == "Lower Jawbone":
    #         label_lookup_localisation[v] = 1
    #     elif k == "Upper Jawbone":
    #         label_lookup_localisation[v] = 2
    #     elif "Lower" in k:
    #         label_lookup_localisation[v] = 3
    #     elif "Upper" in k:
    #         label_lookup_localisation[v] = 4
    #     elif k == "Implant":
    #         label_lookup_localisation[v] = 5
    #     elif k == "Bridge" or k == "Crown":
    #         label_lookup_localisation[v] = 6

    # Set up for tooth centre sd and per-tooth image/label pair creation
    path_per_tooth_image_dir = path_output_dir / "images_per_tooth_cropped"
    path_per_tooth_label_dir = path_output_dir / "labels_per_tooth_cropped"
    path_tooth_centre_sd = path_output_dir / "labels_tooth_centre_sd"

    path_per_tooth_image_dir.mkdir(exist_ok=True, parents=True)
    path_per_tooth_label_dir.mkdir(exist_ok=True, parents=True)
    path_tooth_centre_sd.mkdir(exist_ok=True, parents=True)

    def _run_case(case_id):
        # if not (path_images_rolm / f"{case_id}.nii.gz").exists():
        #     shutil.copy(path_original_images / f"{case_id}_0000.nii.gz", path_images_rolm / f"{case_id}.nii.gz")
        # if not (path_labels_rolm / f"{case_id}.nii.gz").exists():
        #     shutil.copy(path_original_labels / f"{case_id}.nii.gz", path_labels_rolm / f"{case_id}.nii.gz")

        # reflect_image(path_images_rolm / f"{case_id}.nii.gz", path_images_rolm / f"{case_id}_mirrored.nii.gz")

        # if not (path_labels_rolm / f"{case_id}_mirrored.nii.gz").exists() or overwrite:
        #     reflect_image(path_labels_rolm / f"{case_id}.nii.gz", path_labels_rolm / f"{case_id}_m1.nii.gz")
        #     replace_labels(path_labels_rolm / f"{case_id}_m1.nii.gz", path_labels_rolm / f"{case_id}_mirrored.nii.gz", label_lookup_rolm, overwrite=overwrite)
        
        # for suffix in ["", "_mirrored"]:
        for suffix in [""]:
            # # Construct localisation model images - low resolution, teeth (inc bridges, crowns) + bones.
            # if not (path_labels_loc / f"{case_id}{suffix}.nii.gz").exists() or overwrite:
            #     resample_image_and_label(path_images_rolm / f"{case_id}{suffix}.nii.gz",
            #                              path_labels_rolm / f"{case_id}{suffix}.nii.gz",
            #                              path_images_loc / f"{case_id}{suffix}.nii.gz",
            #                              path_labels_loc / f"{case_id}{suffix}_res.nii.gz",
            #                              (1,1,1),
            #                              overwrite)

            #     # Manage labels, upper and lower row implants/crowns etc.
            #     replace_labels(path_labels_loc / f"{case_id}{suffix}_res.nii.gz", path_labels_loc / f"{case_id}{suffix}_lab.nii.gz", label_lookup_localisation, overwrite=overwrite)
            #     process_upper_lower_implants_crowns_bridges(path_labels_loc / f"{case_id}{suffix}_lab.nii.gz", path_labels_loc / f"{case_id}{suffix}.nii.gz", overwrite=overwrite)
            
            # Construct tooth centre signed distance maps, cropped images, masks (does internal existence checking, so no need to put here).
            per_tooth_image_mask_sd(path_images_rolm / f"{case_id}{suffix}.nii.gz", path_labels_rolm / f"{case_id}{suffix}.nii.gz", 
                                    path_per_tooth_image_dir, path_per_tooth_label_dir, path_tooth_centre_sd)
        
            # Tidy up interim files as necessary:
            # (path_images_rolm / f"{case_id}_m1.nii.gz").unlink(missing_ok=True)
            # (path_labels_rolm / f"{case_id}_m1.nii.gz").unlink(missing_ok=True)
            # (path_labels_loc / f"{case_id}{suffix}_res.nii.gz").unlink(missing_ok=True)
            # (path_images_loc / f"{case_id}{suffix}_lab.nii.gz").unlink(missing_ok=True)
            # (path_labels_loc / f"{case_id}{suffix}_lab.nii.gz").unlink(missing_ok=True)

    Parallel(n_jobs=4)(delayed(_run_case)(c) for c in tqdm(case_ids))


if __name__ == "__main__":
    main()
