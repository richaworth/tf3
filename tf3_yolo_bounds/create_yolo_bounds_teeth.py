import numpy as np
import nibabel
from pathlib import Path
from skimage.transform import resize
from joblib import Parallel, delayed
from tqdm import tqdm

import yaml
import imageio
import logging

import yolo_utils


UPPER_LABELS = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 
                21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15} 
LOWER_LABELS = {31: 0, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6, 38: 7, 
                41: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13, 47: 14, 48: 15} 

def main(path_images_in: Path = Path("C:/data/tf3/images_rolm"), 
         path_labels_in: Path = Path("C:/data/tf3/labels_rolm"), 
         path_init_labels_in: Path = Path("C:/data/tf3/labels_localiser_rolm"),
         path_case_ids_yaml: Path = Path("C:/data/tf3/case_id_lists.yaml"),
         path_yolo_out: Path = Path("C:/data/tf3/yolo_teeth_data_640"),
         target_size: int = 640):
    """
    Calculate maximum intensity projection (MIP) along the Z of the target image between the bounds for the initial label. 
    Calculate the related bounding box fractions on those MIPs.
    Output in format suitable for YOLO training.

    Args:
        path_images_in (Path): Path to original image directory. Defaults to "C:/data/tf3/images_rolm".
        path_labels_in (Path): Path to original labels directory. Defaults to "C:/data/tf3/labels_localiser_rolm" (jaws + teeth combined labels).
        path_case_ids_yaml (Path): Path to yaml file containing case IDs for train/test/val split. Defaults to "C:/data/tf3/case_id_lists.yaml".
        path_init_labels_in: Path to localiser yolo data dir.
        path_yolo_out (Path): Path to output yolo training data. Defaults to "C:/data/tf3/yolo_localiser_data".
        target_size (int): Target size - single dimension, as all outputs will be forced square for YOLO training. Defaults to 512.
    """

    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    def _run_case(split_name, case_id):
        for suffix in ["", "_mirrored"]:
            # If already processed, skip
            if (path_yolo_out / split_name / "lower_teeth" / "labels" / f"{case_id}{suffix}.txt").exists() or \
                (path_yolo_out / split_name / "upper_teeth" / "labels" / f"{case_id}{suffix}.txt").exists():
                continue

            path_ct_image = path_images_in / f"{case_id}{suffix}.nii.gz"
            path_label_image = path_labels_in / f"{case_id}{suffix}.nii.gz"
            path_init_label_image = path_init_labels_in / f"{case_id}{suffix}.nii.gz"

            # Calculate 3 MIP images (one per axis)
            image = nibabel.load(path_ct_image)
            labels = nibabel.load(path_label_image)
            init_labels = nibabel.load(path_init_label_image)

            # Clip air to -1000 + max as 3000HU
            # Set air to 0 intensity before padding.
            im_array = image.get_fdata()
            im_array = np.clip(im_array, -1000, 3000) + 1000
            lab_array = labels.get_fdata()
            init_array = init_labels.get_fdata()

            # In the case where shape[1] (Y dir) is longer than shape[0],
            # we can crop square to the front edge of the image.
            # Otherwise, pad square (when Y dir is short) from far end only 
            # (ensures the jaw/teeth are in roughly the same place in all images).
            
            if im_array.shape[1] > im_array.shape[0]:
                im_array = im_array[:, 0:im_array.shape[0], :]
                lab_array = lab_array[:, 0:lab_array.shape[0], :]
                init_array = init_array[:, 0:init_array.shape[0], :]
            elif im_array.shape[1] < im_array.shape[0]:
                pad = im_array.shape[0] - im_array.shape[1]
                im_array = np.pad(im_array, ((0, 0), (0, pad), (0, 0)))
                lab_array = np.pad(lab_array, ((0, 0), (0, pad), (0, 0)))
                init_array = np.pad(init_array, ((0, 0), (0, pad), (0, 0)))
            else:
                logging.debug(f"Image for {case_id} is already square in X and Y, no change required.")

            for init_label, init_jawbone, set_name, label_lookup in zip([3, 4], [1, 2], ("lower", "upper"), (LOWER_LABELS, UPPER_LABELS)):
                # Check if lower jaw (1)/upper jaw (2) are in init_array.
                # Also check if lower/upper (3/4) teeth are in init_array.
                if init_jawbone not in init_array or init_label not in init_array:
                    logging.debug(f"No {set_name} jaw/teeth in {case_id}.")
                    continue

                # Get bounds of this init label; create max intensity image through this
                init_label_mask = np.where(init_array == init_label, 1, 0)
                init_label_axis = np.max(np.max(init_label_mask, axis=0), axis=0)
                
                # Add slices below Z0 (lower)/above Z1 (upper) where possible, 
                # to add context clues for YOLO (i.e. mandible shape, head anatomy).
                # Remove a small number of slices in case the upper and lower teeth overlap in Z.
                context_slices = 30 
                overlap_slices = 10

                z0 = np.min(np.where(init_label_axis != 0))
                z1 = np.max(np.where(init_label_axis != 0))

                if set_name == "lower":
                    z0 = max(z0 - context_slices, 0)
                    z1 = z1 - overlap_slices
                else:
                    z1 = min(z1 + context_slices, im_array.shape[2])
                    z0 = z0 + overlap_slices
                
                # # If, after these, there's no box, skip this case.
                # if z1 <= z0: 
                #     continue

                im_array_z_crop = im_array[:, :, z0 : z1]
                lab_array_z_crop = lab_array[:, :, z0 : z1]

                # Set pulp labels as tooth labels
                lab_array_z_crop = np.where(lab_array_z_crop > 100, lab_array_z_crop - 100, lab_array_z_crop)
                
                path_image_out = path_yolo_out / split_name / f"{set_name}_teeth" / "images" / f"{case_id}{suffix}.png"
                path_yolo_txt_out = path_yolo_out / split_name / f"{set_name}_teeth" / "labels" / f"{case_id}{suffix}.txt"
                path_label_seg_dir_out = path_yolo_out / split_name / f"{set_name}_teeth" / "seg" 

                image_mip = yolo_utils.calculate_mip(im_array_z_crop, axes=2)
                # label_mip = yolo_utils.calculate_mip(lab_array_z_crop, axes=2)
                
                path_image_out.parent.mkdir(exist_ok=True, parents=True)
                path_yolo_txt_out.parent.mkdir(exist_ok=True, parents=True)
                path_label_seg_dir_out.mkdir(exist_ok=True, parents=True)

                # Remove existing yolo bounds file if it exists to prevent accidental double-writing.
                if path_yolo_txt_out.exists():
                    path_yolo_txt_out.unlink()
                
                # Process per tooth labels into seg images and bounds files
                for anatomy_label, yolo_label in label_lookup.items():
                    bb = yolo_utils.calculate_2d_bounds_as_xywh(lab_array_z_crop, anatomy_label, 2)
                    
                    if bb is None:
                        logging.debug(f"{case_id} - No bounds available for {anatomy_label}")
                    else:
                        # No reason for > 4 decimal places' accuracy here, makes output file clearer.
                        # Look up new (continuous, zero indexed) values for teeth.
                        ln = f"{yolo_label}\t{float(bb[0]):.4f}\t{float(bb[1]):.4f}\t{float(bb[2]):.4f}\t{float(bb[3]):.4f}\n" 
                        with (path_yolo_txt_out).open("a") as f:
                            f.write(ln)
                        
                        # Construct per-label segmentation masks in 2d (potentially for yolo seg)
                        label_seg_out = np.max(np.where(lab_array_z_crop == int(anatomy_label), 255, 0), axis=2)
                        label_seg_out_r = resize(label_seg_out, (target_size, target_size), order=0)

                        imageio.imwrite(path_label_seg_dir_out / f"{case_id}{suffix}_{yolo_label}.png", 
                                        label_seg_out_r.astype('uint8'))
                        
                # Save image MIP png - scale to [0, 4000] == [0, 255] for saving as png.
                # Done last, as the image is not required if there's no bounds for it.
                if path_yolo_txt_out.exists():
                    image_mip_out = resize(image_mip, (target_size, target_size))
                    image_mip_out = image_mip_out * (255 / 4000)
                    imageio.imwrite(path_image_out, image_mip_out.astype('uint8'))

    for split_name, case_ids in d_case_ids.items():
        Parallel(n_jobs=8)(delayed(_run_case)(split_name, c) for c in tqdm(case_ids))

if __name__ == "__main__":
    main()
