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

UPPER_LABELS = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 
                21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15} 
LOWER_LABELS = {31: 0, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6, 38: 7, 
                41: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13, 47: 14, 48: 15} 

def main(path_images_in: Path = Path("C:/data/tf3/images_rolm"), 
         path_labels_in: Path = Path("C:/data/tf3/labels_rolm"), 
         path_case_ids_yaml: Path = Path("C:/data/tf3/case_id_lists.yaml"),
         path_localiser_yolo_data: Path = Path("C:/data/tf3/yolo_localiser_data_640"),
         path_data_out: Path = Path("C:/data/tf3/yolo_teeth_data_640_3mm_slices"), 
         target_size: int = 640):
    """
    Calculate maximum intensity projection (MIP) along the Z of the target image between the bounds for the initial label. 
    Calculate the related bounding box fractions on those MIPs.
    Output in format suitable for YOLO training.

    Args:
        path_images_in (Path): Path to original image directory. Defaults to "C:/data/tf3/images_rolm".
        path_labels_in (Path): Path to original labels directory. Defaults to "C:/data/tf3/labels_localiser_rolm" (jaws + teeth combined labels).
        path_case_ids_yaml (Path): Path to yaml file containing case IDs for train/test/val split. Defaults to "C:/data/tf3/case_id_lists.yaml".
        path_localiser_yolo_data: Path to localiser yolo data dir.
        path_yolo_out (Path): Path to output yolo training data. Defaults to "C:/data/tf3/yolo_localiser_data".
        target_size (int): Target size - single dimension, as all outputs will be forced square for YOLO training. Defaults to 512.
    """


    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    # Define parallel function
    def _run_case(split_name, case_id):
        for suffix in ["", "_mirrored"]:
                if (path_data_out / split_name / "lower_teeth" / "labels" / f"{case_id}{suffix}.txt").exists() or \
                    (path_data_out / split_name / "upper_teeth" / "labels" / f"{case_id}{suffix}.txt").exists():
                    continue
        
                path_ct_image = path_images_in / f"{case_id}{suffix}.nii.gz"
                path_label_image = path_labels_in / f"{case_id}{suffix}.nii.gz"

                image = nibabel.load(path_ct_image)
                labels = nibabel.load(path_label_image)

                # Load Z bounds fraction (zbf) from axis 0 and axis 1 data. 
                # Note - consider using the average of axis 0 and axis 1 predictions after running YOLO. Not needed here.
                zbf_axis_0 = {}
                zbf_axis_1 = {}

                if not (path_localiser_yolo_data / split_name / "axis_0" / "labels" / f"{case_id}{suffix}.txt").exists() or\
                    not (path_localiser_yolo_data / split_name / "axis_0" / "labels" / f"{case_id}{suffix}.txt").exists():
                    continue
                
                with (path_localiser_yolo_data / split_name / "axis_0" / "labels" / f"{case_id}{suffix}.txt").open("r") as f:
                    for ln in f:
                        zbf_axis_0[int(ln.split()[0])] = (float(ln.split()[1]), float(ln.split()[2]), float(ln.split()[3]), float(ln.split()[4]))
                with (path_localiser_yolo_data / split_name / "axis_1" / "labels" / f"{case_id}{suffix}.txt").open("r") as f:
                    for ln in f:
                        zbf_axis_1[int(ln.split()[0])] = (float(ln.split()[1]), float(ln.split()[2]), float(ln.split()[3]), float(ln.split()[4]))
                
                # Clip air to -1000 + max as 3000HU
                # Set air to 0 intensity before padding.
                im_array = image.get_fdata()
                im_array = np.clip(im_array, -1000, 3000) + 1000
                lab_array = labels.get_fdata()

                # Crop image, mask to localiser results
                # * Add together bounds for labels 2+3
                # * Include a boundary around the labels in case of localiser error
                # * Ensure cases with small boxes have a minimum box size of, for example, 8cm x 8cm in X and Y. 
                # * Note - localiser bounds include cube-shaped padding.
                
                # Reconstruct original padding
                largest_dim_orig = np.max(im_array.shape)
                pad_max_orig = [largest_dim_orig - x for x in im_array.shape]
                pad_start_orig = [int(x / 2) for x in pad_max_orig]
                pad_end_orig = [x - pad_start_orig[i] for i, x in enumerate(pad_max_orig)]

                im_square_orig = np.pad(im_array, ((pad_start_orig[0], pad_end_orig[0]), (pad_start_orig[1], pad_end_orig[1]), (pad_start_orig[2], pad_end_orig[2])))
                lab_square_orig = np.pad(lab_array, ((pad_start_orig[0], pad_end_orig[0]), (pad_start_orig[1], pad_end_orig[1]), (pad_start_orig[2], pad_end_orig[2])))

                # Construct cropping box (sf == slice fraction, cb = crop box)
                margin = 10
                min_size = 270 # 80mm * 0.3 voxel size - rounded up slightly. Note - must be even.
                assert min_size < largest_dim_orig, "Longest uncropped image dimension can not be shorter than the minimum crop box."

                if 2 in zbf_axis_0.keys() and 3 in zbf_axis_0.keys() and 2 in zbf_axis_1.keys() and 3 in zbf_axis_1.keys():
                    crop_0 = min(zbf_axis_0[2][2], zbf_axis_0[3][2]), max(zbf_axis_0[2][3], zbf_axis_0[3][3])
                    crop_1 = min(zbf_axis_1[2][2], zbf_axis_1[3][2]), max(zbf_axis_1[2][3], zbf_axis_1[3][3])
                    crop_2 = min(zbf_axis_0[2][0], zbf_axis_1[2][0], zbf_axis_0[3][0], zbf_axis_1[3][0]), \
                        max(zbf_axis_0[2][1], zbf_axis_1[2][1], zbf_axis_0[3][1], zbf_axis_1[3][1])
                elif 2 in zbf_axis_0.keys() and 2 in zbf_axis_1.keys():
                    crop_0 = zbf_axis_0[2][2], zbf_axis_0[2][3]
                    crop_1 = zbf_axis_1[2][2], zbf_axis_1[2][3]
                    crop_2 = min(zbf_axis_0[2][0], zbf_axis_1[2][0]), max(zbf_axis_0[2][1], zbf_axis_1[2][1])
                elif 3 in zbf_axis_0.keys() and 3 in zbf_axis_1.keys():
                    crop_0 = zbf_axis_0[3][2], zbf_axis_0[3][3]
                    crop_1 = zbf_axis_1[3][2], zbf_axis_1[3][3]
                    crop_2 = min(zbf_axis_0[3][0], zbf_axis_1[3][0]), max(zbf_axis_0[3][1], zbf_axis_1[3][1])
                else:
                    continue

                # Calculate cropping box in voxels, add margins.
                box = [max(int(crop_0[0] * largest_dim_orig) - margin, 0), 
                       min(int(crop_0[1] * largest_dim_orig) + margin, largest_dim_orig), 
                       max(int(crop_1[0] * largest_dim_orig) - margin, 0), 
                       min(int(crop_1[1] * largest_dim_orig) + margin, largest_dim_orig), 
                       max(int(crop_2[0] * largest_dim_orig), 0), 
                       min(int(crop_2[1] * largest_dim_orig), largest_dim_orig), 
                       ]

                # Ensure minimums in X and Y.
                for dim in [(0, 1), (2, 3)]:
                    dim_0 = box[dim[0]]
                    dim_1 = box[dim[1]]
                    length_dim = dim_1 - dim_0

                    if length_dim >= min_size:
                        continue

                    dim_0 = dim_0 - int(min_size / 2)
                    dim_1 = dim_1 + int(min_size / 2)

                    # If this leaves boxes that fall outside the image, adjust accordingly.
                    if dim_0 < 0:
                        diff = dim_0 * -1
                        dim_0 = 0
                        dim_1 = dim_1 + diff
                    elif dim_1 > largest_dim_orig:
                        diff = dim_0 - largest_dim_orig
                        dim_1 = largest_dim_orig
                        dim_0 = dim_0 - diff
                    else:
                        pass
                    
                    box[dim[0]] = dim_0
                    box[dim[1]] = dim_1

                # Crop previously padded images, re-pad to new longest.
                im_cropped = im_square_orig[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
                lab_cropped = lab_square_orig[box[0]:box[1], box[2]:box[3], box[4]:box[5]]

                largest_dim = np.max(im_cropped.shape)
                pad_max = [largest_dim - x for x in im_cropped.shape]
                pad_start = [int(x / 2) for x in pad_max]
                pad_end = [x - pad_start[i] for i, x in enumerate(pad_max)]

                im_square = np.pad(im_cropped, ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])))
                lab_square = np.pad(lab_cropped, ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])))

                # Construct MIPs
                for init_label in [2, 3]:
                    if init_label not in zbf_axis_0.keys() or init_label not in zbf_axis_1.keys():
                            print(f"Label {init_label} not in init data for {case_id}")
                            continue
                    
                    z0_slice_orig = int(min(zbf_axis_0[init_label][0], zbf_axis_1[init_label][0]) * largest_dim_orig)
                    z1_slice_orig = int(max(zbf_axis_0[init_label][1], zbf_axis_1[init_label][1]) * largest_dim_orig)

                    z0_slice = z0_slice_orig - pad_start_orig[2] + pad_start[2]
                    z1_slice = z1_slice_orig - pad_start_orig[2] + pad_start[2]

                    if init_label == 2:
                        verbose_label = "lower_teeth"
                        dict_labels = LOWER_LABELS
                    else:
                        verbose_label = "upper_teeth"
                        dict_labels = UPPER_LABELS
                    
                    i = 0
                    slice_thickness = 10
                    for z0 in range(z0_slice, z1_slice, slice_thickness):
                        z1 = z0 + slice_thickness
                        
                        # Manage partial slice - if it's the only possible one, keep it; if it's more than 50%, keep it; otherwise skip.
                        if z1 > z1_slice:
                            if i == 0 or z1_slice - z0 > slice_thickness / 2:
                                z1 = z1_slice
                            else:
                                continue
                            
                        image_mip_in = im_square[:, :, z0 : z1]
                        label_mip_in = lab_square[:, :, z0 : z1]
                        
                        # Set pulp labels as teeth labels.
                        label_mip_in = np.where(label_mip_in > 100, label_mip_in - 100, label_mip_in)

                        # Manage paths and directories
                        path_image_mip_out = path_data_out / split_name / verbose_label / "images" / f"{case_id}_{i}{suffix}.png"
                        path_label_mip_out = path_data_out / split_name / verbose_label / "label_review" / f"{case_id}_{i}{suffix}.png"
                        path_yolo_txt_out = path_data_out / split_name / verbose_label / "labels" / f"{case_id}_{i}{suffix}.txt"

                        path_image_mip_out.parent.mkdir(exist_ok=True, parents=True)
                        path_label_mip_out.parent.mkdir(exist_ok=True, parents=True)
                        path_yolo_txt_out.parent.mkdir(exist_ok=True, parents=True)

                        # Remove existing yolo bounds file if it exists to prevent accidental double-writing.
                        if path_yolo_txt_out.exists():
                            path_yolo_txt_out.unlink()

                        for label in dict_labels:
                            # Pulp is entirely contained within tooth, so can just get tooth box.
                            bb = calculate_2d_bounds_as_fraction(label_mip_in, label, 2)  

                            if bb is None:
                                logging.info(f"{path_ct_image.name} - No bounds available for {label}")
                            else:
                                # No reason for > 4 decimal places' accuracy here, makes output file clearer.
                                # Look up new (continuous, zero indexed) values for teeth.
                                label_out = dict_labels[label]
                                ln = f"{label_out}\t{float(bb[0]):.4f}\t{float(bb[1]):.4f}\t{float(bb[2]):.4f}\t{float(bb[3]):.4f}\n" 
                                with (path_yolo_txt_out).open("a") as f:
                                    f.write(ln)
                        
                        # If no yolo bounds file is created (i.e. no bounds were found), do not create the image + label review.
                        if path_yolo_txt_out.exists():
                            image_mip = calculate_mip(image_mip_in, axes=2)
                            label_mip = calculate_mip(label_mip_in, axes=2)  # Review only

                            image_mip_out = resize(image_mip, (target_size, target_size))
                            label_mip_out = resize(label_mip, (target_size, target_size))

                            # Save image MIP png - scale to [0, 4000] == [0, 255] for saving as png.
                            image_mip_out = image_mip_out * (255 / 4000)
                            imageio.imwrite(path_image_mip_out, image_mip_out.astype('uint8'))

                            # Save mask review MIP png - scale list_labels == [0, 255] for saving as png.
                            label_mip_out = np.where(label_mip_out > 100, label_mip_out - 100, label_mip_out)
                            label_mip_out = label_mip_out * (255 / max(dict_labels.keys()))
                            imageio.imwrite(path_label_mip_out, label_mip_out.astype('uint8'))

                        # Increment slice
                        i = i + 1


    for split_name, case_ids in d_case_ids.items():
        Parallel(n_jobs=1)(delayed(_run_case)(split_name, c) for c in tqdm(case_ids))

if __name__ == "__main__":
    main()
