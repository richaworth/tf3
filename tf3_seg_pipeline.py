import logging
from types import SimpleNamespace

import imageio
import nibabel
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from skimage.transform import resize
from scipy.ndimage import binary_dilation
import SimpleITK as sitk

import yaml

from tf3_yolo_bounds.yolo_utils import yolo_bounds_from_image, calculate_mip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_TEETH_LOOKUP = {
    0: "right_central_incisor",
    1: "right_lateral_incisor",
    2: "right_canine",
    3: "right_first_premolar",
    4: "right_second_premolar",
    5: "right_first_molar",
    6: "right_second_molar",
    7: "right_third_molar",
    8: "left_central_incisor",
    9: "left_lateral_incisor",
    10: "left_canine",
    11: "left_first_premolar",
    12: "left_second_premolar",
    13: "left_first_molar",
    14: "left_second_molar",
    15: "left_third_molar"
}

LOWER_TEETH_LOOKUP = {
    "right_central_incisor": 41,
    "right_lateral_incisor": 42,
    "right_canine": 43,
    "right_first_premolar": 44,
    "right_second_premolar": 45,
    "right_first_molar": 46,
    "right_second_molar": 47,
    "right_third_molar": 48,
    "left_central_incisor": 31,
    "left_lateral_incisor": 32,
    "left_canine": 33,
    "left_first_premolar": 34,
    "left_second_premolar": 35,
    "left_first_molar": 36,
    "left_second_molar": 37,
    "left_third_molar": 38
}

UPPER_TEETH_LOOKUP = {
    "right_central_incisor": 21,
    "right_lateral_incisor": 22,
    "right_canine": 23,
    "right_first_premolar": 24,
    "right_second_premolar": 25,
    "right_first_molar": 26,
    "right_second_molar": 27,
    "right_third_molar": 28,
    "left_central_incisor": 11,
    "left_lateral_incisor": 12,
    "left_canine": 13,
    "left_first_premolar": 14,
    "left_second_premolar": 15,
    "left_first_molar": 16,
    "left_second_molar": 17,
    "left_third_molar": 18
}

def segment_one_image(im_array_in: np.ndarray, case_id: str, path_interim_data: Path, models: dict) -> np.ndarray:
    """
    Note: managing upper and lower in near-identical sets rather than refactoring due to time constraints.

    Args:
        im_array_in:
        case_id:
        path_interim_data:
        models:

    Returns:

    """
    # Create output array
    label_array_out = np.zeros_like(im_array_in)

    # Calculate square, clipped 2D images - AP and LR axes.
    largest_dim = np.max(im_array_in.shape)
    pad_max = [largest_dim - x for x in im_array_in.shape]
    pad_start = [int(x / 2) for x in pad_max]
    pad_end = [x - pad_start[i] for i, x in enumerate(pad_max)]
    pad_shape = (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])

    # Clip to -1000, 3000 for air-max HU, align -1000 to zero for padding with zeros.
    im_array_clipped = np.clip(im_array_in, -1000, 3000) + 1000
    im_array_square = np.pad(im_array_clipped, pad_shape)

    # Calculate Maximum Intensity Projection through the volume in AP and LR directions
    # LR is axis 0, AP is axis 1.
    image_mips = calculate_mip(im_array_square, axes=[0, 1])

    # Run YOLO localiser (lower jaw, upper jaw, lower teeth, upper teeth) on these.
    # All YOLO models were trained on 640 x 640 square images.
    # Take results of these and calculate centre of upper teeth, lower teeth in Z (removing padding). 
    # Note: Due to the way MIP images are created, Z is axis 0 (Y) in LR images and axis 0 (X) in AP images.
    
    path_image_mip_axis_lr = path_interim_data / "yolo_1" / f"axis_lr" / f"{case_id}_axis_lr.png"
    path_image_mip_axis_ap = path_interim_data / "yolo_1" / f"axis_ap" / f"{case_id}_axis_ap.png"
    
    (path_interim_data / "yolo_1" / f"axis_lr").mkdir(exist_ok=True, parents=True)
    (path_interim_data / "yolo_1" / f"axis_ap").mkdir(exist_ok=True, parents=True)
    
    # Process LR axis MIP
    image_mip_axis_lr_out = resize(image_mips[0], (640, 640)) 
    image_mip_axis_lr_out = image_mip_axis_lr_out * (255 / 4000) # TODO: Remove this and output png once happy
    imageio.imwrite(path_image_mip_axis_lr, image_mip_axis_lr_out.astype('uint8'))
    
    lr_model_result = yolo_bounds_from_image(path_image_mip_axis_lr, YOLO(models["yolo_axis_lr"]), path_image_mip_axis_lr.parent)

    # Process AP axis MIP
    image_mip_axis_ap_out = resize(image_mips[1], (640, 640))
    image_mip_axis_ap_out = image_mip_axis_ap_out * (255 / 4000) # TODO: Remove this and output png once happy
    imageio.imwrite(path_image_mip_axis_ap, image_mip_axis_ap_out.astype('uint8'))

    ap_model_result = yolo_bounds_from_image(path_image_mip_axis_ap, YOLO(models["yolo_axis_ap"]), path_image_mip_axis_ap.parent)

    # Use results to construct upper and lower teeth images
    # * If a YOLO result doesn't have the related jaw, do not include the teeth.
    # * Get highest "lower teeth" slice and lowest "higher teeth" slice (If missing from one axis, use the other; otherwise mean of both)
    # * Adjust slightly - remove some central slices in case of tooth overlap, add extra slices above/below for additional context clues for YOLO
    context_slices = 30
    overlap_slices = 10
    
    if all(["lower_jaw" in lr_model_result.keys(), "lower_teeth" in lr_model_result.keys(),
            "lower_jaw" in ap_model_result.keys(), "lower_teeth" in ap_model_result.keys()]):
        z0_lower = np.mean([lr_model_result["lower_teeth"]["box"][0], ap_model_result["lower_teeth"]["box"][0]])
        z1_lower = np.mean([lr_model_result["lower_teeth"]["box"][2], ap_model_result["lower_teeth"]["box"][2]])
    elif "lower_jaw" in lr_model_result.keys() and "lower_teeth" in lr_model_result.keys():
        z0_lower = lr_model_result["lower_teeth"]["box"][0]
        z1_lower = lr_model_result["lower_teeth"]["box"][2]
    elif "lower_jaw" in ap_model_result.keys() and "lower_teeth" in ap_model_result.keys():
        z0_lower = ap_model_result["lower_teeth"]["box"][0]
        z1_lower = ap_model_result["lower_teeth"]["box"][2]
    else:
        z0_lower = None
        z1_lower = None

    if z0_lower is not None:
        z0_lower = int(((z0_lower / 640) * largest_dim) - pad_start[2])
        z1_lower = int(((z1_lower / 640) * largest_dim) - pad_start[2])
        z0_lower = max(z0_lower - context_slices, 0)
        z1_lower = z1_lower - overlap_slices

    if all(["upper_jaw" in lr_model_result.keys(), "upper_teeth" in lr_model_result.keys(),
            "upper_jaw" in ap_model_result.keys(), "upper_teeth" in ap_model_result.keys()]):
        z0_upper = np.mean([lr_model_result["upper_teeth"]["box"][0], ap_model_result["upper_teeth"]["box"][0]])
        z1_upper = np.mean([lr_model_result["upper_teeth"]["box"][2], ap_model_result["upper_teeth"]["box"][2]])
    elif "upper_jaw" in lr_model_result.keys() and "upper_teeth" in lr_model_result.keys():
        z0_upper = lr_model_result["upper_teeth"]["box"][0]
        z1_upper = lr_model_result["upper_teeth"]["box"][2]
    elif "upper_jaw" in ap_model_result.keys() and "upper_teeth" in ap_model_result.keys():
        z0_upper = ap_model_result["upper_teeth"]["box"][0]
        z1_upper = ap_model_result["upper_teeth"]["box"][2]
    else:
        z0_upper = None
        z1_upper = None

    if z0_upper is not None:
        z0_upper = int(((z0_upper / 640) * largest_dim) - pad_start[2])
        z1_upper = int(((z1_upper / 640) * largest_dim) - pad_start[2])
        z0_upper = z0_upper + overlap_slices
        z1_upper = min(z1_upper + context_slices, im_array_clipped.shape[2])

    # Cropping/padding for the tooth-finder (Y-axis only):    
    # In the case where shape[1] is longer than shape[0], we can crop square to the front edge of the image.
    # Otherwise, pad square from far end only (ensures the jaw/teeth are in roughly the same place in all images).
    if im_array_clipped.shape[1] > im_array_clipped.shape[0]:
        im_array_tooth_finder = im_array_clipped[:, 0:im_array_clipped.shape[0], :]
    elif im_array_clipped.shape[1] < im_array_clipped.shape[0]:
        pad = im_array_clipped.shape[0] - im_array_clipped.shape[1]
        im_array_tooth_finder = np.pad(im_array_clipped, ((0, 0), (0, pad), (0, 0)))
    else:
        im_array_tooth_finder = im_array_clipped

    # Run YOLO tooth finder on these
    yolo_lower_teeth = None
    yolo_upper_teeth = None

    if z0_lower is not None:
        path_image_mip_lower = path_interim_data / "yolo_2" / f"lower_teeth" / f"{case_id}_lower_teeth.png"
        path_image_mip_lower.parent.mkdir(exist_ok=True, parents=True)

        im_array_tooth_finder_lower = im_array_tooth_finder[:, :, z0_lower: z1_lower]
        mip_tooth_finder_lower = calculate_mip(im_array_tooth_finder_lower, axes=[2])
        
        mip_tooth_finder_lower_out = resize(mip_tooth_finder_lower, (640, 640))
        mip_tooth_finder_lower_out = mip_tooth_finder_lower_out * (255 / 4000) # TODO: Remove this and output png once happy
        imageio.imwrite(path_image_mip_lower, mip_tooth_finder_lower_out.astype('uint8'))

        yolo_lower_teeth = yolo_bounds_from_image(path_image_mip_lower, YOLO(models["yolo_lower_teeth"]),
                                                  path_image_mip_lower.parent, combine_multiple=False, min_conf=0.5)

    if z0_upper is not None:
        path_image_mip_upper = path_interim_data / "yolo_2" / f"upper_teeth" / f"{case_id}_upper_teeth.png"
        path_image_mip_upper.parent.mkdir(exist_ok=True, parents=True)

        im_array_tooth_finder_upper = im_array_tooth_finder[:, :, z0_upper: z1_upper]
        mip_tooth_finder_upper = calculate_mip(im_array_tooth_finder_upper, axes=[2])

        mip_tooth_finder_upper_out = resize(mip_tooth_finder_upper, (640, 640))
        mip_tooth_finder_upper_out = mip_tooth_finder_upper_out * (255 / 4000)  # TODO: Remove this and output png once happy
        imageio.imwrite(path_image_mip_upper, mip_tooth_finder_upper_out.astype('uint8'))

        yolo_upper_teeth = yolo_bounds_from_image(path_image_mip_upper, YOLO(models["yolo_upper_teeth"]),
                                                  path_image_mip_upper.parent, combine_multiple=False, min_conf=0.5)
        print(f"{case_id} - {len(yolo_upper_teeth.keys())} upper teeth found.")

    # TODO: Load and run large-anatomy model here
    # Perform resampling/MONAI transforms.

    # Run inference
    large_anatomy_out = torch.Tensor()

    # Move result to CPU, set values of mask to lookup values # TODO - check upper uses correct lookup.
    # TODO: tensor.where is probably quicker than np.where if label_array_out can stay in GPU. To test.
    # Perform inverse resampling/MONAI transforms.

    # Correct labels - lower, upper jawbones are correct (1=1, 2=2) (set per-label)
    # Sinuses and pharynx are off by 2 (3=5, 4=6, 5=7)
    large_anatomy_result = large_anatomy_out.detach().cpu().numpy()
    label_array_out = np.where(large_anatomy_result == 1, 1, label_array_out)
    label_array_out = np.where(large_anatomy_result == 2, 2, label_array_out)
    label_array_out = np.where(large_anatomy_result > 2, large_anatomy_result + 2, label_array_out)

    # TODO: From these, run implant/crown/bridge segmentation and canal segmentation
    # Perform resampling/MONAI transforms.
    # Run inference
    implant_crown_bridge_out = torch.Tensor()

    # Correct labels - bridge, crown, implant are off by 7 (1=8, 2=9, 3=10)
    implant_crown_bridge_result = implant_crown_bridge_out.detach().cpu().numpy()
    label_array_out = np.where(implant_crown_bridge_result > 0, implant_crown_bridge_result + 7, label_array_out)

    # TODO: Load and run canal model here
    # Perform resampling/MONAI transforms.
    # Run inference
    canals_out = torch.Tensor()

    # Move result to CPU, set values of mask to lookup values # TODO - check upper uses correct lookup.
    # Perform resampling

    # Correct labels and add to label_array.
    # L/R inferior alveolar canals are off by 2 (1=3, 2=4) (set per-label)
    # L/R incisive canals are off by 100 (3=103, 4=104, 5=105)
    canals_result = canals_out.detach().cpu().numpy()
    label_array_out = np.where(canals_result == 1, 3, label_array_out)
    label_array_out = np.where(canals_result == 2, 4, label_array_out)
    label_array_out = np.where(canals_result > 2, canals_result + 100, label_array_out)


    # For each tooth found - find centres. (TOOTH_LOOKUP is in the form label_value: label_name).
    # * Currently, just take highest confidence if there's more than one box for that label.
    # * Tooth patch model ROI == 80 x 80 x 80 voxels
    # For each tooth centre, create a signed distance map from a small sphere at this point to the rest of the patch.

    # TODO: Load tooth model here

    for k, v in YOLO_TEETH_LOOKUP.items():
        if yolo_lower_teeth is not None and v in yolo_lower_teeth.keys():
            if isinstance(yolo_lower_teeth[v]["conf"], list):
                print(f"{case_id} {v} - {len(yolo_lower_teeth[v]['conf'])} boxes found.")
                yolo_lower_teeth[v]["box"] = yolo_lower_teeth[v]["box"][np.argmax(yolo_lower_teeth[v]["conf"])]
                yolo_lower_teeth[v]["conf"] = yolo_lower_teeth[v]["conf"][np.argmax(yolo_lower_teeth[v]["conf"])]
                print(f"{case_id} {v} - Max confidence: {yolo_lower_teeth[v]['conf']}")

            # Note: Tooth finder image is indexed from the uncropped side in Y, and uncropped in X.
            bb = [(yolo_lower_teeth[v]["box"][0] / 640) * im_array_tooth_finder.shape[0],
                  (yolo_lower_teeth[v]["box"][1] / 640)  * im_array_tooth_finder.shape[1],
                  z0_lower,
                  (yolo_lower_teeth[v]["box"][2] / 640) * im_array_tooth_finder.shape[0],
                  (yolo_lower_teeth[v]["box"][3] / 640) * im_array_tooth_finder.shape[1],
                  z1_lower]

            tooth_centre = [int((bb[0] + bb[3]) / 2), int((bb[1] + bb[4]) / 2), int((bb[2] + bb[5]) / 2)]

            # Ensure that patches don't have negative or over-sized bounds.
            x0 = max(tooth_centre[0] - 40, 0)
            y0 = max(tooth_centre[1] - 40, 0)
            z0 = max(tooth_centre[2] - 40, 0)
            x1 = min(tooth_centre[0] + 40, im_array_in.shape[0])
            y1 = min(tooth_centre[1] + 40, im_array_in.shape[1])
            z1 = min(tooth_centre[2] + 40, im_array_in.shape[2])

            # Check the identified centre isn't already in an implant or bridge.
            if label_array_out[(x0+x1) / 2, (y0+y1) / 2, (z0+z1) / 2] == 8 or \
                    label_array_out[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2] == 9:
                print(f"Tooth centre for lower {v} is in mask identified as implant or bridge. Skipping.")
                continue

            # Pad tooth patch if at the edge.
            pad_x0 = 0 if x0 > 0 else -1 * (tooth_centre[0] - 40)
            pad_y0 = 0 if y0 > 0 else -1 * (tooth_centre[1] - 40)
            pad_z0 = 0 if y0 > 0 else -1 * (tooth_centre[1] - 40)
            pad_x1 = 0 if x1 < im_array_in.shape[0] else (tooth_centre[0] + 40) - im_array_in.shape[0]  # Potentially needs -1 here?
            pad_y1 = 0 if y1 < im_array_in.shape[1] else (tooth_centre[1] + 40) - im_array_in.shape[1]  # Potentially needs -1 here?
            pad_z1 = 0 if z1 < im_array_in.shape[2] else (tooth_centre[2] + 40) - im_array_in.shape[2]  # Potentially needs -1 here?

            tooth_image = im_array_in[x0:x1, y0:y1, z0:z1]
            tooth_image_pad = np.pad(tooth_image, (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1), mode='constant', constant_values=0)

            tooth_sd = np.zeros_like(tooth_image_pad)
            tooth_sd[40, 40, 40] = 1
            tooth_sd = binary_dilation(tooth_sd, iterations=2).astype(np.uint8)

            itk_tooth_sd = sitk.GetImageFromArray(tooth_sd)
            itk_smdm_out = sitk.SignedMaurerDistanceMap(itk_tooth_sd, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)

            # Clip image to 0-3000; SD to -50-0. Set these to 0-1.
            tooth_image_pad = ((np.clip(tooth_image_pad, a_min=0, a_max=3000)) / 3000)
            tooth_sd = ((np.clip(sitk.GetArrayViewFromImage(itk_smdm_out), a_min=-50, a_max=0)) / 50 ) + 1

            # Concatenate Image and SD along axis 0
            tensor_tooth_im = torch.Tensor(tooth_image_pad.astype(np.float32)).to(DEVICE)
            tensor_tooth_sd = torch.Tensor(tooth_sd.astype(np.float32)).to(DEVICE)
            tensor_inputs = torch.cat([tensor_tooth_im, tensor_tooth_sd], dim=0)

            # TODO: Apply any transforms that are required (possibly ChannelFirst, Orientation). Run Tooth search.
            # From MONAI:
            #   EnsureChannelFirstd(keys=["image", "localiser", "label"]),
            #   Orientationd(keys=["image", "localiser", "label"], axcodes="RAS"),

            tensor_output = torch.Tensor()

            # Move result to CPU, set values of mask to lookup values # TODO - check upper uses correct lookup.
            tooth_model_result = tensor_output.detach().cpu().numpy()
            tooth_model_result = np.where(tooth_model_result == 1, LOWER_TEETH_LOOKUP[v], tooth_model_result)
            tooth_model_result = np.where(tooth_model_result == 2, LOWER_TEETH_LOOKUP[v] + 100, tooth_model_result)

            # Un-pad result if necessary:
            if tooth_image.shape != tooth_image_pad.shape:
                tooth_model_result = tooth_model_result[pad_x0:80-pad_x1, pad_y0:80-pad_y1, pad_x0:80-pad_z1]

            # Back-fill output array.
            im_array_in[x0:x1, y0:y1, z0:z1] = np.where(tooth_model_result > 0, tooth_model_result, im_array_in[x0:x1, y0:y1, z0:z1])

    # TODO; duplicate for upper.

    return label_array_out

    # * If time allows, model the overall tooth centre shape and use this + plausibility detection to see if any were
    #   missed or incorrectly labelled.







    print()

def main():
    path_case_ids_yaml = Path("C:/data/tf3/case_id_lists.yaml")
    path_data_dir = Path("C:/data/tf3/images_rolm")
    path_output_data = Path("C:/data/tf3/test_pipeline_out")

    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    test_cases = d_case_ids["test"]
    list_images = ([path_data_dir / f"{x}.nii.gz" for x in test_cases] +
                   [path_data_dir / f"{x}_mirrored.nii.gz" for x in test_cases])

    model_paths = {
        "yolo_axis_lr": Path("C:/data/tf3/yolo_models/yolo_localiser_640_axis_0.pt"),
        "yolo_axis_ap": Path("C:/data/tf3/yolo_models/yolo_localiser_640_axis_1.pt"),
        "yolo_lower_teeth": Path("C:/data/tf3/yolo_models/yolo_tooth_finder_lower_jaw.pt"),
        "yolo_upper_teeth": Path("C:/data/tf3/yolo_models/yolo_tooth_finder_upper_jaw.pt")
    }

    for path_image in list_images:
        im = nibabel.load(path_image)
        im_array = im.get_fdata()

        result = segment_one_image(im_array, path_image.name.split(".nii.gz")[0], path_output_data, model_paths)

        # Save final image
        im_out = nibabel.Nifti1Image(result, im.affine, im.header, dtype=im.header.get_data_dtype())
        nibabel.save(im_out, path_output_data / path_image.name)


if __name__ == "__main__":
    main()


