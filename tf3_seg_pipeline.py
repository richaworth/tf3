import imageio
import nibabel
import numpy as np
import logging

import torch
from ultralytics import YOLO
from pathlib import Path
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import SimpleITK as sitk

from monai.data.utils import decollate_batch
from monai.networks.nets.unet import UNet
from monai.networks.nets.unetr import UNETR
from monai.inferers.utils import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirst,
    EnsureType,
    Orientation,
    LoadImage,
    SaveImage,
    Spacing,
    SpatialPad,
    ScaleIntensityRange
)

from tqdm import tqdm

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

PULP_LABEL = 50

def review_pngs(array: np.ndarray, axes: list[int], path_outdir: Path, filestem: str):
    path_outdir.mkdir(exist_ok=True, parents=True)
    mips = calculate_mip(array, axes=axes)

    for mip, a in zip(mips, axes):
        mip_out = mip * 255 / np.max(mip)
        imageio.imwrite(path_outdir / f"{filestem}_axis_{a}.png", mip_out.astype('uint8'))


def segment_one_image(path_image_in: Path, path_interim_data: Path, models: dict) -> np.ndarray:
    """
    Note: managing upper and lower in near-identical sets rather than refactoring due to time constraints.

    Args:
        im_array_in:
        path_interim_data:
        models:

    Returns:

    """
    # Load image
    im = LoadImage()(path_image_in)
    case_id = path_image_in.name.split(".")[0]

    # Create output array
    im_array_in = im.numpy()
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
    
    path_image_mip_axis_lr = path_interim_data / "yolo_1" / "axis_lr" / f"{case_id}_axis_lr.png"
    path_image_mip_axis_ap = path_interim_data / "yolo_1" / "axis_ap" / f"{case_id}_axis_ap.png"
    
    (path_interim_data / "yolo_1" / "axis_lr").mkdir(exist_ok=True, parents=True)
    (path_interim_data / "yolo_1" / "axis_ap").mkdir(exist_ok=True, parents=True)
    
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
        path_image_mip_lower = path_interim_data / "yolo_2" / "lower_teeth" / f"{case_id}_lower_teeth.png"
        path_image_mip_lower.parent.mkdir(exist_ok=True, parents=True)

        im_array_tooth_finder_lower = im_array_tooth_finder[:, :, z0_lower: z1_lower]
        mip_tooth_finder_lower = calculate_mip(im_array_tooth_finder_lower, axes=[2])
        
        mip_tooth_finder_lower_out = resize(mip_tooth_finder_lower, (640, 640))
        mip_tooth_finder_lower_out = mip_tooth_finder_lower_out * (255 / 4000) # TODO: Remove this and output png once happy
        imageio.imwrite(path_image_mip_lower, mip_tooth_finder_lower_out.astype('uint8'))

        yolo_lower_teeth = yolo_bounds_from_image(path_image_mip_lower, YOLO(models["yolo_lower_teeth"]),
                                                  path_image_mip_lower.parent, combine_multiple=False, min_conf=0.5)
        
        # YOLO output is not commensurate with 3D image frame - need to transpose X and Y for each box.
        for k, v in yolo_lower_teeth.items():
            yolo_lower_teeth[k]["box"] = np.array([v["box"][1], v["box"][0], v["box"][3], v["box"][2]])


    if z0_upper is not None:
        path_image_mip_upper = path_interim_data / "yolo_2" / "upper_teeth" / f"{case_id}_upper_teeth.png"
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

    # Transforms input image according to model training.
    la_image_in = EnsureChannelFirst()(im)
    la_image_in = Orientation(axcodes="RAS")(la_image_in)
    la_image_in = ScaleIntensityRange(a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True)(la_image_in)
    la_image_in = SpatialPad((96, 96, 96))(la_image_in)
    la_image_in = Spacing(pixdim=(0.5, 0.5, 0.5), mode="bilinear")(la_image_in)

    large_anatomy_model = UNETR(
        in_channels=1,
        out_channels=5,
        img_size=(96, 96, 96),
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    large_anatomy_model.to(torch.device(DEVICE))
    torch.load(models["seg_large_anatomy"], large_anatomy_model.state_dict(), weights_only=False)
    large_anatomy_model.eval()
    
    with torch.no_grad():
        with torch.autocast("cuda"):
            large_anatomy_out = sliding_window_inference(la_image_in, (96, 96, 96), 8, large_anatomy_model, 0.5)
    
    large_anatomy_out = AsDiscrete(argmax=True)(large_anatomy_out)
    large_anatomy_out = Spacing(pixdim=(0.3, 0.3, 0.3), mode="nearest")(large_anatomy_out)

    # Move result to CPU, set values of mask to lookup values # TODO - check upper uses correct lookup.
    # TODO: tensor.where is probably quicker than np.where if label_array_out can stay in GPU. To test.
    # Perform inverse resampling/MONAI transforms.

    # Correct labels - lower, upper jawbones are correct (1=1, 2=2) (set per-label)
    # Sinuses and pharynx are off by 2 (3=5, 4=6, 5=7)
    large_anatomy_result = large_anatomy_out.detach().cpu().numpy()

    label_array_out = np.where(large_anatomy_result == 1, 1, label_array_out)
    label_array_out = np.where(large_anatomy_result == 2, 2, label_array_out)
    label_array_out = np.where(large_anatomy_result > 2, large_anatomy_result + 2, label_array_out)

    """
    # TODO: From these, run implant/crown/bridge segmentation and canal segmentation
    # Perform MONAI transforms.
    # Run inference
    implant_crown_bridge_out = torch.Tensor()

    # Correct labels - bridge, crown, implant are off by 7 (1=8, 2=9, 3=10)
    implant_crown_bridge_result = implant_crown_bridge_out.detach().cpu().numpy()
    label_array_out = np.where(implant_crown_bridge_result > 0, implant_crown_bridge_result + 7, label_array_out)

    # TODO: Load and run canal model here
    # Perform MONAI transforms.
    # Run inference
    canals_out = torch.Tensor()

    # Move result to CPU, set values of mask to lookup values # TODO - check upper uses correct lookup.

    # Correct labels and add to label_array.
    # L/R inferior alveolar canals are off by 2 (1=3, 2=4) (set per-label)
    # L/R incisive canals are off by 100 (3=103, 4=104, 5=105)
    canals_result = canals_out.detach().cpu().numpy()
    label_array_out = np.where(canals_result == 1, 3, label_array_out)
    label_array_out = np.where(canals_result == 2, 4, label_array_out)
    label_array_out = np.where(canals_result > 2, canals_result + 100, label_array_out)
    """

    # For each tooth found - find centres. (TOOTH_LOOKUP is in the form label_value: label_name).
    # * Currently, just take highest confidence if there's more than one box for that label.
    # * Tooth patch model ROI == 80 x 80 x 80 voxels
    # For each tooth centre, create a signed distance map from a small sphere at this point to the rest of the patch.

    # Ended up having to retrain. Bad times.

    # TODO: Load tooth model here
    # tooth_model = UNet(
    #     spatial_dims=3,
    #     in_channels=2,
    #     out_channels=3,
    #     channels=(4, 8, 16, 32, 64),
    #     strides=(2, 2, 2, 2),
    # )

    # tooth_model.to(DEVICE)
    # torch.load(models["seg_tooth"], tooth_model.state_dict(), weights_only=False)
    # tooth_model.eval()

    # def _run_teeth_finder(label_value, tooth_name, row):
    #     if row == "lower":
    #         yolo_result = yolo_lower_teeth
    #         z0 = z0_lower
    #         z1 = z1_lower
    #         lookup = LOWER_TEETH_LOOKUP
    #     else:
    #         yolo_result = yolo_upper_teeth
    #         z0 = z0_upper
    #         z1 = z1_upper
    #         lookup = UPPER_TEETH_LOOKUP

    #     if yolo_result is None or tooth_name not in yolo_result.keys():
    #         return        

    #     if isinstance(yolo_result[tooth_name]["conf"], list):
    #         print(f"{case_id} {tooth_name} - {len(yolo_result[tooth_name]['conf'])} boxes found.")
    #         yolo_result[tooth_name]["box"] = yolo_result[tooth_name]["box"][np.argmax(yolo_result[tooth_name]["conf"])]
    #         yolo_result[tooth_name]["conf"] = yolo_result[tooth_name]["conf"][np.argmax(yolo_result[tooth_name]["conf"])]
    #         logging.debug(f"{case_id} {tooth_name} - Max confidence: {yolo_result[tooth_name]['conf']}")

    #     # Note: Tooth finder image is indexed from the uncropped side in Y, and uncropped in X.
    #     bb = [(yolo_result[tooth_name]["box"][0] / 640) * im_array_tooth_finder.shape[0],
    #           (yolo_result[tooth_name]["box"][1] / 640)  * im_array_tooth_finder.shape[1],
    #           z0,
    #           (yolo_result[tooth_name]["box"][2] / 640) * im_array_tooth_finder.shape[0],
    #           (yolo_result[tooth_name]["box"][3] / 640) * im_array_tooth_finder.shape[1],
    #           z1
    #         ]

    #     tooth_centre = [int((bb[0] + bb[3]) / 2), int((bb[1] + bb[4]) / 2), int((bb[2] + bb[5]) / 2)]

    #     # Ensure that patches don't have negative or over-sized bounds.
    #     x0 = max(tooth_centre[0] - 40, 0)
    #     y0 = max(tooth_centre[1] - 40, 0)
    #     z0 = max(tooth_centre[2] - 40, 0)
    #     x1 = min(tooth_centre[0] + 40, im_array_in.shape[0])
    #     y1 = min(tooth_centre[1] + 40, im_array_in.shape[1])
    #     z1 = min(tooth_centre[2] + 40, im_array_in.shape[2])

    #     # Check the identified centre isn't already in an implant or bridge.
    #     if label_array_out[int((x0+x1) / 2), int((y0+y1) / 2), int((z0+z1) / 2)] == 8 or \
    #             label_array_out[int((x0+x1) / 2), int((y0+y1) / 2), int((z0+z1) / 2)] == 9:
    #         print(f"Tooth centre for {tooth_name} is in mask identified as implant or bridge. Skipping.")
    #         return 

    #     # Pad tooth patch if at the edge.
    #     pad_x0 = 0 if x0 > 0 else -1 * (tooth_centre[0] - 40)
    #     pad_y0 = 0 if y0 > 0 else -1 * (tooth_centre[1] - 40)
    #     pad_z0 = 0 if z0 > 0 else -1 * (tooth_centre[2] - 40)
    #     pad_x1 = 0 if x1 < im_array_in.shape[0] else (tooth_centre[0] + 40) - im_array_in.shape[0]  # Potentially needs -1 here?
    #     pad_y1 = 0 if y1 < im_array_in.shape[1] else (tooth_centre[1] + 40) - im_array_in.shape[1]  # Potentially needs -1 here?
    #     pad_z1 = 0 if z1 < im_array_in.shape[2] else (tooth_centre[2] + 40) - im_array_in.shape[2]  # Potentially needs -1 here?

    #     tooth_image = im_array_in[x0:x1, y0:y1, z0:z1]
    #     padding = (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1)
    #     if not all(padding) == 0:
    #         tooth_image_pad = np.pad(tooth_image, (padding), mode='constant', constant_values=0)
    #     else:
    #         tooth_image_pad = tooth_image

    #     tooth_sd = np.zeros_like(tooth_image_pad)
    #     tooth_sd[40, 40, 40] = 1
    #     tooth_sd = binary_dilation(tooth_sd, iterations=2).astype(np.uint8)

    #     itk_tooth_sd = sitk.GetImageFromArray(tooth_sd)
    #     itk_smdm_out = sitk.SignedMaurerDistanceMap(itk_tooth_sd, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)

    #     # Clip image to 0-3000; SD to -50-0. Set these to 0-1.
    #     tooth_image_pad = ((np.clip(tooth_image_pad, a_min=0, a_max=3000)) / 3000)
    #     tooth_sd = ((np.clip(sitk.GetArrayViewFromImage(itk_smdm_out), a_min=-50, a_max=0)) / 50 ) + 1

    #     # Concatenate Image and SD along axis 0
    #     tensor_tooth_im = EnsureChannelFirst()(tooth_image_pad, meta_dict=im.meta)
    #     tensor_tooth_sd = EnsureChannelFirst()(tooth_sd, meta_dict=im.meta)

    #     tensor_tooth_im = Orientation(axcodes="RAS")(tensor_tooth_im)
    #     tensor_tooth_sd = Orientation(axcodes="RAS")(tensor_tooth_sd)

    #     tensor_inputs = torch.cat([tensor_tooth_im, tensor_tooth_sd], dim=0)
    #     tensor_inputs = EnsureType(dtype=np.float32)(tensor_inputs)

    #     # review_pngs(tensor_tooth_im.numpy()[0], [0, 1, 2], path_interim_data / "png", f"{case_id}_{tooth_name}_im_tensor")
    #     # review_pngs(tensor_tooth_sd.numpy()[0], [0, 1, 2], path_interim_data / "png", f"{case_id}_{tooth_name}_sd_tensor")

    #     tensor_inputs = tensor_inputs.to(DEVICE)

    #     with torch.no_grad():
    #         tensor_inputs = tensor_inputs.unsqueeze(0)
    #         tensor_output = tooth_model(tensor_inputs)

    #     # Move result to CPU, set values of mask to lookup values
    #     tooth_model_result = tensor_output.detach().cpu().numpy().squeeze()
    #     tooth_model_result_argmax = np.argmax(tooth_model_result, 0)
    #     tooth_model_conf = tooth_model_result.squeeze()[1]

    #     # Un-pad results as necessary:
    #     if not all(padding) == 0:
    #         tooth_model_result_argmax = tooth_model_result_argmax[pad_x0:80-pad_x1, pad_y0:80-pad_y1, pad_x0:80-pad_z1]
    #         tooth_model_conf = tooth_model_conf[pad_x0:80-pad_x1, pad_y0:80-pad_y1, pad_x0:80-pad_z1]

    #     review_pngs(tooth_model_result_argmax, [0, 1, 2], path_interim_data / "png", f"{case_id}_{tooth_name}_label_out")

    #     # Back-fill output arrays to full image size.
    #     tooth_model_conf_out = np.zeros_like(im_array_in)
    #     tooth_model_conf_out[x0:x1, y0:y1, z0:z1] = np.where(tooth_model_conf > 0, tooth_model_conf, 0)
        
    #     tooth_mask = np.where(tooth_model_result_argmax == 1, 1, 0)
        
    #     # Use binary closing to tidy up pulp segmentations
    #     pulp_mask = np.where(tooth_model_result_argmax == 2, 1, 0)
    #     pulp_mask = binary_dilation(pulp_mask, iterations=1)
    #     pulp_mask = binary_erosion(pulp_mask, iterations=1)

    #     tooth_model_result_argmax_out = np.zeros_like(im_array_in)
    #     tooth_model_result_argmax_out[x0:x1, y0:y1, z0:z1] = tooth_mask
    #     tooth_model_result_argmax_out[x0:x1, y0:y1, z0:z1] = np.where(pulp_mask == 1, 2, tooth_model_result_argmax_out[x0:x1, y0:y1, z0:z1])

    #     tooth_label_out = np.where(tooth_model_result_argmax_out == 1, tooth_model_result_argmax_out, 0)
    #     pulp_label_out = np.where(tooth_model_result_argmax_out == 1, PULP_LABEL, 0)

    #     return {
    #         "tooth": tooth_name, 
    #         "target_label": tooth_model_result_argmax_out, 
    #         "tooth_label_confidence": tooth_model_conf, 
    #         "argmax": tooth_model_result_argmax,
    #         "tooth_label": tooth_label_out,
    #         "pulp_label": pulp_label_out
    #     }

    # # TODO: Make parallel.
    # # TODO: Lock file for tooth model?
    # tooth_results = []
    # for k, v in YOLO_TEETH_LOOKUP.items():
    #     tooth_results.append(_run_teeth_finder(k, v, "lower"))
    #     tooth_results.append(_run_teeth_finder(k, v, "upper"))

    # for d in tooth_results:
    #     if d is None:
    #         continue

    #     # Deliberately bad to find overlaps
    #     label_array_out = label_array_out + d["tooth_label"] + d["pulp_label"]
    #     label_array_out = np.clip(label_array_out, a_min=0, a_max=54)

    # Output review image if using .nii.gz
    if ".nii.gz" in path_image_in.name:
        nibabel_im_in = nibabel.load(path_image_in)
        im_out = nibabel.Nifti1Image(label_array_out, nibabel_im_in.affine, nibabel_im_in.header, dtype=nibabel_im_in.header.get_data_dtype())
        nibabel.save(im_out, path_interim_data / f"{case_id}_labels.nii.gz")

    return label_array_out

    # * If time allows, model the overall tooth centre shape and use this + plausibility detection to see if any were
    #   missed or incorrectly labelled.



    # TODO: Fix labelling to output format from Toothfairy 3 forum
    # Multi-class segmentation map of the oral-pharyngeal space with: 
    # 0 - Background, 1 - Lower Jawbone, 2 - Upper Jawbone, 3 - Left Inferior Alveolar Canal, 4 - Right Inferior Alveolar Canal, 
    # 5 - Left Maxillary Sinus, 6 - Right Maxillary Sinus, 7 - Pharynx, 
    # 11 - Upper Right Central Incisor, 12 - Upper Right Lateral Incisor, 13 - Upper Right Canine, 14 - Upper Right First Premolar, 
    # 15 - Upper Right Second Premolar, 16 - Upper Right First Molar, 17 - Upper Right Second Molar, 18 - Upper Right Third Molar (Wisdom Tooth), 
    # 21 - Upper Left Central Incisor, 22 - Upper Left Lateral Incisor, 23 - Upper Left Canine, 24 - Upper Left First Premolar, 
    # 25 - Upper Left Second Premolar, 26 - Upper Left First Molar, 27 - Upper Left Second Molar, 28 - Upper Left Third Molar (Wisdom Tooth), 
    # 31 - Lower Left Central Incisor, 32 - Lower Left Lateral Incisor, 33 - Lower Left Canine, 34 - Lower Left First Premolar, 
    # 35 - Lower Left Second Premolar, 36 - Lower Left First Molar, 37 - Lower Left Second Molar, 38 - Lower Left Third Molar (Wisdom Tooth), 
    # 41 - Lower Right Central Incisor, 42 - Lower Right Lateral Incisor, 43 - Lower Right Canine, 44 - Lower Right First Premolar, 
    # 45 - Lower Right Second Premolar, 46 - Lower Right First Molar, 47 - Lower Right Second Molar, 48 - Lower Right Third Molar (Wisdom Tooth) 
    # 50 - Tooth Pulp 
    # 51 - Left Incisive Nerve 52 - Right Incisive Nerve 53 - Lingual Nerve



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
        "yolo_upper_teeth": Path("C:/data/tf3/yolo_models/yolo_tooth_finder_upper_jaw.pt"),
        "seg_tooth": Path("C:/data/tf3/seg_models/per_tooth_unet_80_80_80.pkl"),
        "seg_large_anatomy": Path("C:/data/tf3/seg_models/non_tooth_anatomy_96_96_96.pkl"),
        "seg_canals": Path("C:/data/tf3/seg_models/canals_from_jawbone_96_96_96.pkl"),
        "seg_bridges_crowns": Path("C:/data/tf3/seg_models/bridges_crowns_implants_96_96_96.pkl")
    }

    for path_image in tqdm(list_images[:1]):
        result = segment_one_image(path_image, path_output_data, model_paths)



if __name__ == "__main__":
    main()


