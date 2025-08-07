import numpy as np
import nibabel
from pathlib import Path
from skimage.transform import resize
from joblib import Parallel, delayed
from tqdm import tqdm

import yaml
import imageio
import logging

from yolo_utils import calculate_mip, calculate_2d_bounds_as_fraction

def main(path_images_in: Path = Path("C:/data/tf3/images_rolm"), 
         path_labels_in: Path = Path("C:/data/tf3/labels_localiser_rolm"), 
         path_case_ids_yaml: Path = Path("C:/data/tf3/case_id_lists.yaml"),
         path_yolo_out: Path = Path("C:/data/tf3/yolo_localiser_data_640"), 
         mip_axes: list[int] = [0, 1],
         bounds_labels: list[int] = [1, 2, 3, 4],
         target_size: int = 640):
    """
    Calculate maximum intensity projection (MIP) along the axes of a 3d image; calculate the related bounding box fractions on those MIPs.
    Output in format suitable for YOLO training.

    Args:
        path_images_in (Path): Path to original image directory. Defaults to "C:/data/tf3/images_rolm".
        path_labels_in (Path): Path to original labels directory. Defaults to "C:/data/tf3/labels_localiser_rolm" (jaws + teeth combined labels).
        path_case_ids_yaml (Path): Path to yaml file containing case IDs for train/test/val split. Defaults to "C:/data/tf3/case_id_lists.yaml".
        path_yolo_out (Path): Path to output yolo training data. Defaults to "C:/data/tf3/yolo_localiser_data".
        mip_axes (list[int]): Axes to calculate MIP over. Defaults to [0, 1, 2] (all axes).
        bounds_labels (list[int]): Labels to add to the text file. Defaults to [1, 2, 3, 4] (jaws + teeth combined labels).
        target_size (int): Target size - single dimension, as all outputs will be forced square for YOLO training. Defaults to 640.
    """
    # Label needs to be zero indexed and continuous
    output_label_old_new = {
        1: 0,
        2: 1,
        3: 2,
        4: 3
    }

    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    def _run_case(split_name, case_id):
        for suffix in ["", "_mirrored"]:
            # If already processed, skip
            if (path_yolo_out / split_name / "axis_1" / "images" / f"{case_id}{suffix}.png").exists() and \
                (path_yolo_out / split_name / "axis_1" / "label_review" / f"{case_id}{suffix}.png").exists() and \
                (path_yolo_out / split_name / "axis_1" / "labels" / f"{case_id}{suffix}.txt"):
                continue

            path_ct_image = path_images_in / f"{case_id}{suffix}.nii.gz"
            path_label_image = path_labels_in / f"{case_id}{suffix}.nii.gz"

            # Calculate 3 MIP images (one per axis)
            image = nibabel.load(path_ct_image)
            labels = nibabel.load(path_label_image)

            # Clip air to -1000 + max as 3000HU
            # Set air to 0 intensity before padding.
            im_array = image.get_fdata()
            im_array = np.clip(im_array, -1000, 3000) + 1000
            lab_array = labels.get_fdata()

            # Create letterboxed cubes from im_array, lab_array
            largest_dim = np.max(im_array.shape)
            pad_max = [largest_dim - x for x in im_array.shape]
            pad_start = [int(x / 2) for x in pad_max]
            pad_end = [x - pad_start[i] for i, x in enumerate(pad_max)]

            im_square = np.pad(im_array, ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])))
            lab_square = np.pad(lab_array, ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])))

            image_mips = calculate_mip(im_square, axes=mip_axes)
            label_mips = calculate_mip(lab_square, axes=mip_axes)  # For review  - bounds are not calculated on label MIP.

            # YOLO expected dir structure
            # DIRNAME/SPLIT/images/im0.jpg # Path to the image file
            # DIRNAME/SPLIT/labels/im0.txt # Path to the corresponding label file

            # File structure (per line):
            for a in mip_axes:
                path_image_mip_out = path_yolo_out / split_name / f"axis_{a}" / "images" / f"{case_id}{suffix}.png"
                path_label_mip_out = path_yolo_out / split_name / f"axis_{a}" / "label_review" / f"{case_id}{suffix}.png"
                path_yolo_txt_out = path_yolo_out / split_name / f"axis_{a}" / "labels" / f"{case_id}{suffix}.txt"

                path_image_mip_out.parent.mkdir(exist_ok=True, parents=True)
                path_label_mip_out.parent.mkdir(exist_ok=True, parents=True)
                path_yolo_txt_out.parent.mkdir(exist_ok=True, parents=True)

                image_mip_out = resize(image_mips[a], (target_size, target_size))
                label_mip_out = resize(label_mips[a], (target_size, target_size))

                # Save image MIP png - scale to [0, 4000] == [0, 255] for saving as png.
                image_mip_out = (image_mip_out) * (255 / 4000)
                imageio.imwrite(path_image_mip_out, image_mip_out.astype('uint8'))

                # Save mask review MIP png - scale to [0, 4] == [0, 255] for saving as png.
                label_mip_out = label_mip_out * (255 / len(bounds_labels))
                imageio.imwrite(path_label_mip_out, label_mip_out.astype('uint8'))

                # Remove existing yolo bounds file if it exists to prevent accidental double-writing.
                if path_yolo_txt_out.exists():
                    path_yolo_txt_out.unlink()

                for label in bounds_labels:
                    bb = calculate_2d_bounds_as_fraction(lab_square, label, a)

                    if bb is None:
                        logging.info(f"{path_ct_image.name} - No bounds available for {label}")
                    else:
                        # No reason for > 4 decimal places' accuracy here, makes output file clearer.
                        label_out = output_label_old_new[label]
                        
                        line = f"{label_out}\t{float(bb[0]):.4f}\t{float(bb[1]):.4f}\t{float(bb[2]):.4f}\t{float(bb[3]):.4f}\n" 
                        with (path_yolo_txt_out).open("a") as f:
                            f.write(line)

    for split_name, case_ids in d_case_ids.items():
        Parallel(n_jobs=6)(delayed(_run_case)(split_name, c) for c in tqdm(case_ids))

if __name__ == "__main__":
    main()
