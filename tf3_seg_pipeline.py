import imageio
import numpy as np
import logging

import torch
from ultralytics import YOLO
from pathlib import Path
from skimage.transform import resize
from skimage.measure import label
from scipy.ndimage import binary_dilation
import SimpleITK as sitk

from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader

from monai.data.utils import decollate_batch
from monai.networks.nets.unet import UNet
from monai.networks.nets.unetr import UNETR
from monai.inferers.utils import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureType,
    LoadImage,
    LoadImaged,
    Orientation,
    Orientationd,
    ResizeWithPadOrCrop,
    SaveImage,
    Spacing,
    Spacingd,
    ScaleIntensityRanged
)

from tqdm import tqdm

import yaml

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

YOLO_TEETH_LOOKUP = {
    0: "left_central_incisor",
    1: "left_lateral_incisor",
    2: "left_canine",
    3: "left_first_premolar",
    4: "left_second_premolar",
    5: "left_first_molar",
    6: "left_second_molar",
    7: "left_third_molar",
    8: "right_central_incisor",
    9: "right_lateral_incisor",
   10: "right_canine",
   11: "right_first_premolar",
   12: "right_second_premolar",
   13: "right_first_molar",
   14: "right_second_molar",
   15: "right_third_molar"
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

TOOTH_TO_NEIGHBOUR_LOOKUP = {
    18: 17,
    17: 16,
    16: 15,
    15: 14, 
    14: 13, 
    13: 12, 
    12: 11,
    11: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    27: 28,
    38: 37,
    37: 36,
    36: 35,
    35: 34, 
    34: 33, 
    33: 32, 
    32: 31,
    31: 41,
    41: 42,
    42: 43,
    43: 44,
    44: 45,
    45: 46,
    46: 47,
    47: 48,
}

PULP_LABEL = 50

# Moved in helper functions to give flat structure for docker. Similarly moved models to algo directory.

def torch_dilate_conv3d(tensor_in: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """
    Perform binary dilation of a 3D torch tensor (using torch.nn.functional.conv3d).
    (Not strictly the same as binary dilation, but works similarly enough). 

    Args:
        tensor_in (torch.Tensor): Tensor to perform dilation.
        iterations (int, optional): Number of iterations to perform. Defaults to 1.

    Returns:
        torch.Tensor: Dilated tensor (in torch.float32 format).
    """
    assert len(tensor_in.shape) == 4 or len(tensor_in.shape) == 5, \
        f"Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: {tensor_in.shape}." 
    assert torch.all(torch.logical_or(tensor_in == 0, tensor_in == 1)), "Input must be binary" 

    # Set up dilation kernel (all ones, 3x3x3 cube)
    kernel = np.ones((1, 1, 3, 3, 3), dtype=np.float32)
    kernel_tensor = torch.Tensor(kernel).to(tensor_in.device)

    # If tensor has only 4 dimensions, unsqueeze before diation, squeeze after.
    if len(tensor_in.shape) == 4:
        tensor_in = tensor_in.unsqueeze(0)
        squeeze_result = True
    else:
        squeeze_result = False
    
    torch_result = tensor_in.type_as(kernel_tensor)

    for _ in range(iterations):
        torch_result = torch.clamp(torch.nn.functional.conv3d(torch_result, kernel_tensor, padding=1), 0, 1)
    
    return torch_result.squeeze(0).type_as(tensor_in) if squeeze_result else torch_result.type_as(tensor_in)

def torch_erode_conv3d(tensor_in: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """
    Perform binary erosion on 3D torch tensor (using torch.nn.functional.conv3d)
    (Not strictly the same as binary erosion, but works similarly enough). 

    Args:
        tensor_in (torch.Tensor): Tensor to perform erosion.
        iterations (int, optional): Number of iterations to perform. Defaults to 1.

    Returns:
        torch.Tensor: Eroded tensor (in torch.float32 format).
    """
    assert len(tensor_in.shape) == 4 or len(tensor_in.shape) == 5, \
        f"Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: {tensor_in.shape}." 
    assert torch.all(torch.logical_or(tensor_in == 0, tensor_in == 1)), "Input must be binary" 

    # Set up kernel for erosion (must be of shape [1, 1, 3, 3, 3])
    kernel = np.array(
        [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
         [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.float32)
    kernel = np.expand_dims(np.expand_dims(kernel, 0), 0)

    kernel_tensor = torch.Tensor(kernel).to(tensor_in.device)

    # If tensor has only 4 dimensions, unsqueeze before diation, squeeze after.
    if len(tensor_in.shape) == 4:
        tensor_in = tensor_in.unsqueeze(0)
        squeeze_result = True
    else:
        squeeze_result = False
    
    torch_result = tensor_in.type_as(kernel_tensor)

    for _ in range(iterations):
        conv_result = torch.nn.functional.conv3d(torch_result, kernel_tensor, padding=1)
        torch_result = torch.where(conv_result == 7, 1, 0).to(torch.float32)
    
    return torch_result.squeeze(0).type_as(tensor_in) if squeeze_result else torch_result.type_as(tensor_in)

def calculate_mip(np_img: np.ndarray, axes: int | list[int] = 0) -> np.ndarray | list[np.ndarray]:
    """
    Calculate MIP from original image along given axis, between slices (if provided).

    Args:
        np_img (np.ndarray): Numpy array of image (3d)
        axes (int, optional): Axis/axes along which to calculate MIP. Defaults to 0.

    Returns:
        Maximum intensity projection along the axis/axes.
    """
    if isinstance(axes, int):
        axes = [axes]

    mips = []

    for a in axes:
        mips.append(np.max(np_img, axis = a))

    if len(mips) == 1:
        return mips[0]
    else:
        return mips


def calculate_2d_bounds_as_fraction(label_array: np.ndarray, 
                                    labels: int | list[int], 
                                    axes: int | list[int] = 0) -> tuple[float, float, float, float] | list[tuple[float, float, float, float]] | None:
    """
    Calculate the 2D bounds of one or more given labels (combined) within a 3D mask, as a fraction of the equivalent axis' MIP image.

    Args:
        np_img (np.ndarray): Numpy array of mask (3d)
        labels (int | list[int]): Label or labels from which to calculate bounds. Will combine these
        axes (int | list[int], optional): Axis or axes along which to calculate bounds. Defaults to 0.

    Returns:
        tuple[float, float, float, float] | list[tuple[float, float, float, float]] | None: Bounds of all labels along given axis/axes. Returns None if label not available.
    """
    if isinstance(labels, int): 
        labels = [labels]
    if isinstance(axes, int): 
        axes = [axes]

    output = []
    
    mask = np.zeros_like(label_array)
        
    for lab in labels:
        mask = np.where(label_array == lab, 1, mask)

    for a in axes:
        mask_2d = np.max(mask, axis = a)

        # If there's nothing in the final mask, return all None
        if not np.any(mask_2d):
            return None
        
        arr = np.where(mask_2d != 0)
        i0, i1, j0, j1 = np.min(arr[0]), np.max(arr[0]), np.min(arr[1]), np.max(arr[1]) 
        fi0, fi1, fj0, fj1 = i0 / mask_2d.shape[1], i1 / mask_2d.shape[1], j0 / mask_2d.shape[0], j1 / mask_2d.shape[0]  
        
        # Ends up being in Ymin, Ymax, Xmin, Xmax order; fix here.
        output.append((fj0, fj1, fi0, fi1))
    
    if len(output) == 1:
        return output[0]
    else:
        return output


def yolo_bounds_from_image(image: Path | np.ndarray,
                           yolo_model,
                           png_output_dir: Path | None = None,
                           combine_multiple: bool = True,
                           min_conf: float = 0.66) -> dict:
    """
    Run yolo.predict on an image, output review png and return a dict of boxes and confidences per label.
    If a label is repeated in the yolo output, either combine these or return a list.

    Args:
        image: Path to input png image.
        yolo_model: YOLO model to use for prediction.
        png_output_dir: Path to output png image directory.
        combine_multiple: If True - Combine all boxes with the same label, return the lowest confidence.
            If False - Return all boxes as a list. Default is True.
        min_conf: Minimum confidence threshold. Default is 0.66.

    Returns:
        Dict of box/boxes per label and their confidence values ({"box": box/boxes in xyxy format (measured in pixels);
            "conf": confidence value for that box/those boxes}).
    """
    result = yolo_model(source=image, conf=min_conf, verbose=False)
    output = {}

    # Only passing one image at a time
    r = result[0]

    if png_output_dir is not None and isinstance(image, Path):
        png_output_dir.mkdir(exist_ok=True, parents=True)
        r.save(filename=png_output_dir / f"{image.stem}_out.png")

    for box in r.boxes:
        c = int(box.cls.item())
        label = r.names[c]
        conf = box.conf.item()

        # Detach box and output in X0,Y0,X1,Y1 format
        box = box.xyxy.detach().cpu().numpy()[0]

        if label in output.keys() and combine_multiple:
            if conf > output[label]["conf"]:
                conf = output[label]["conf"]

            bb_x0 = np.min((output[label]["box"][0], box[0]))
            bb_y0 = np.min((output[label]["box"][1], box[1]))
            bb_x1 = np.max((output[label]["box"][2], box[2]))
            bb_y1 = np.max((output[label]["box"][3], box[3]))
            bb = np.asarray([bb_x0, bb_y0, bb_x1, bb_y1])

            output[label] = {"box": bb, "conf": conf}
        elif label in output.keys() and not combine_multiple:
            if not isinstance(output[label]["box"], list):
                output[label]["box"] = [output[label]["box"], box]
                output[label]["conf"] = [output[label]["conf"], conf]
            else:
                output[label]["box"].append(box)
                output[label]["conf"].append(conf)
        else:
            output[label] = {"box": box, "conf": conf}

    return output


def review_pngs(array: np.ndarray, axes: list[int], path_outdir: Path, filestem: str):
    path_outdir.mkdir(exist_ok=True, parents=True)
    mips = calculate_mip(array.astype(np.float32), axes=axes)

    for mip, a in zip(mips, axes):
        mip_out = mip * 255 / np.max(mip)
        imageio.imwrite(path_outdir / f"{filestem}_axis_{a}.png", mip_out.astype('uint8'))


def segment_one_image(path_image_in: Path, path_interim_data: Path, model_path_dict: dict, run_in_debug_mode: bool = False) -> torch.Tensor:
    """
    This has been written as one long script due to time constraints and the lack of requirement for re-use.

    Args:
        path_image_in: Path to input image.
        path_interim_data: Path to store interim data.
        model_path_dict: Dictionary of model paths.

    Returns: 
        Tensor containing semantic segmentation labels.
    """
    #################################
    # YOLO bone and tooth detectors
    #################################

    # Load image
    im = LoadImage()(path_image_in)
    case_id = path_image_in.name.split(".")[0]

    # Create output array
    im_array_in = im.numpy()

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
    # LR is axis 0, AP is axis 13.
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
    image_mip_axis_lr_out = image_mip_axis_lr_out * (255 / 4000)
    imageio.imwrite(path_image_mip_axis_lr, image_mip_axis_lr_out.astype('uint8'))
    
    lr_model_result = yolo_bounds_from_image(path_image_mip_axis_lr, YOLO(model_path_dict["yolo_axis_lr"]), path_image_mip_axis_lr.parent)

    # Process AP axis MIP
    image_mip_axis_ap_out = resize(image_mips[1], (640, 640))
    image_mip_axis_ap_out = image_mip_axis_ap_out * (255 / 4000)
    imageio.imwrite(path_image_mip_axis_ap, image_mip_axis_ap_out.astype('uint8'))

    ap_model_result = yolo_bounds_from_image(path_image_mip_axis_ap, YOLO(model_path_dict["yolo_axis_ap"]), path_image_mip_axis_ap.parent)

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
        mip_tooth_finder_lower_out = mip_tooth_finder_lower_out * (255 / 4000) 
        imageio.imwrite(path_image_mip_lower, mip_tooth_finder_lower_out.astype('uint8'))

        yolo_lower_teeth = yolo_bounds_from_image(path_image_mip_lower, YOLO(model_path_dict["yolo_lower_teeth"]),
                                                  path_image_mip_lower.parent, combine_multiple=False, min_conf=0.4)

        # YOLO output is not commensurate with 3D image frame - need to transpose X and Y for each box.
        for k, v in yolo_lower_teeth.items():
            if isinstance(v["box"], list):
                boxes = boxes = [np.array([box[1], box[0], box[3], box[2]]) for box in v["box"]]
                yolo_lower_teeth[k]["box"] = boxes
            else:
                yolo_lower_teeth[k]["box"] = np.array([v["box"][1], v["box"][0], v["box"][3], v["box"][2]])

    if z0_upper is not None:
        path_image_mip_upper = path_interim_data / "yolo_2" / "upper_teeth" / f"{case_id}_upper_teeth.png"
        path_image_mip_upper.parent.mkdir(exist_ok=True, parents=True)

        im_array_tooth_finder_upper = im_array_tooth_finder[:, :, z0_upper: z1_upper]
        mip_tooth_finder_upper = calculate_mip(im_array_tooth_finder_upper, axes=[2])

        mip_tooth_finder_upper_out = resize(mip_tooth_finder_upper, (640, 640))
        mip_tooth_finder_upper_out = mip_tooth_finder_upper_out * (255 / 4000)
        imageio.imwrite(path_image_mip_upper, mip_tooth_finder_upper_out.astype('uint8'))

        yolo_upper_teeth = yolo_bounds_from_image(path_image_mip_upper, YOLO(model_path_dict["yolo_upper_teeth"]),
                                                  path_image_mip_upper.parent, combine_multiple=False, min_conf=0.4)
        
        # YOLO output is not commensurate with 3D image frame - need to transpose X and Y for each box.
        for k, v in yolo_upper_teeth.items():
            if isinstance(v["box"], list):
                boxes = [np.array([box[1], box[0], box[3], box[2]]) for box in v["box"]]
                yolo_upper_teeth[k]["box"] = boxes
            else:
                yolo_upper_teeth[k]["box"] = np.array([v["box"][1], v["box"][0], v["box"][3], v["box"][2]])

    #######################################################
    # Search jawbones, pharynx, sinuses, implants, bridges
    #######################################################
    print("Run large anatomy search")

    # Set up MONAI dataloader, dataset (Note: managing using batch dataset loader due to deadline constraints)
    la_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode="nearest"),
        ScaleIntensityRanged(keys=["image"],a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True)
    ]

    la_data = [{"image": path_image_in}]
    la_ds = Dataset(data=la_data, transform=Compose(la_transforms))
    la_loader = DataLoader(la_ds, batch_size=1, num_workers=1)

    # Load large anatomy model
    large_anatomy_model = UNETR(
        in_channels=1,
        out_channels=6,
        img_size=(80, 80, 80),
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    large_anatomy_model.to(DEVICE)

    if DEVICE == torch.device("cpu"):
        large_anatomy_model.load_state_dict(torch.load(model_path_dict["seg_large_anatomy"], weights_only=True, map_location=torch.device('cpu')))
    else:
        large_anatomy_model.load_state_dict(torch.load(model_path_dict["seg_large_anatomy"], weights_only=True))
    large_anatomy_model.eval()

    with torch.no_grad():
        for data in la_loader:
            la_inputs = data["image"].to(DEVICE)
            
            if DEVICE == torch.device("cpu"):
                la_out = sliding_window_inference(la_inputs, (80, 80, 80), 24, large_anatomy_model, 0.5)
            else:
                with torch.autocast("cuda"):
                    la_out = sliding_window_inference(la_inputs, (80, 80, 80), 24, large_anatomy_model, 0.5)

            for la_inference_out in decollate_batch(la_out):
                la_inference_out = AsDiscrete(argmax=True)(la_inference_out)
                la_inference_out = Spacing(pixdim=(0.3, 0.3, 0.3), mode="nearest")(la_inference_out)
                la_inference_out = ResizeWithPadOrCrop(im.shape)(la_inference_out)

                if run_in_debug_mode:
                    si = SaveImage(output_dir=path_interim_data / "test_segmentations", output_postfix="large_anatomy_from_torch", separate_folder=False)
                    si(la_inference_out)

    # Correct labels - lower, upper jawbones are correct (1=1, 2=2) (set per-label)
    # Sinuses and pharynx are off by 2 (3=5, 4=6, 5=7)

    label_tensor_out = la_inference_out.type(torch.int8)
    jawbone_tensor = torch.where(label_tensor_out == 1, 1, 0)
    label_tensor_out = torch.where(label_tensor_out > 2, label_tensor_out + 2, label_tensor_out)

    if run_in_debug_mode:
        si = SaveImage(output_dir=path_interim_data, output_postfix="large_anatomy", separate_folder=False)
        si(label_tensor_out)

        # label_array_out = label_tensor_out.squeeze().cpu().numpy()
        # review_pngs(label_array_out, [0, 1, 2], path_interim_data / "png", f"{case_id}_large_anatomy_out")

    
    ###########################################
    # Search canals/nerves from lower jawbone
    ###########################################
    # If no lower jawbone was detected, skip these. Otherwise:
    if "lower_jaw" in lr_model_result.keys() or "lower_jaw" in ap_model_result.keys():
        # Set up MONAI dataloader, dataset (Note: managing using batch dataset loader due to deadline constraints)
        print("Run canals search")

        canals_transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"],a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True)
        ]

        canals_data = [{"image": path_image_in}]
        canals_ds = Dataset(data=canals_data, transform=Compose(canals_transforms))
        canals_loader = DataLoader(canals_ds, batch_size=1, shuffle=False, num_workers=1)

        # Load Canal Segmentation model
        canals_model = UNETR(
            in_channels=2,
            out_channels=6,
            img_size=(64, 64, 64),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )

        canals_model.to(DEVICE)

    if DEVICE == torch.device("cpu"):
        canals_model.load_state_dict(torch.load(model_path_dict["seg_canals"], weights_only=True, map_location=torch.device('cpu')))
    else:
        canals_model.load_state_dict(torch.load(model_path_dict["seg_canals"]))
        canals_model.eval()

        with torch.no_grad():
            for data in canals_loader:
                image = data["image"].to(DEVICE)
                canals_inputs = torch.cat([image, jawbone_tensor.unsqueeze(0)], dim=1)

                if DEVICE == torch.device("cpu"):
                    canals_out = sliding_window_inference(canals_inputs, (64, 64, 64), 24, canals_model, 0.5)
                else:
                    with torch.autocast("cuda"):
                        canals_out = sliding_window_inference(canals_inputs, (64, 64, 64), 24, canals_model, 0.5)

                for canals_inference_out in decollate_batch(canals_out):
                    canals_inference_out = AsDiscrete(argmax=True)(canals_inference_out)
                    
                    if run_in_debug_mode:
                        si = SaveImage(output_dir=path_interim_data / "test_segmentations", output_postfix="canals_from_torch", separate_folder=False)
                        si(canals_inference_out)

        # # L/R inferior alveolar canals (1=3, 2=4)
        # # L/R incisive canals and Lingual canal  (3=51, 4=52, 5=53)  - Could use dilation of the canals to fill the hollows in the jawbone 
        label_tensor_out = torch.where(canals_inference_out == 1, 3, label_tensor_out)
        label_tensor_out = torch.where(canals_inference_out == 2, 4, label_tensor_out)
        label_tensor_out = torch.where(canals_inference_out == 3, 51, label_tensor_out)
        label_tensor_out = torch.where(canals_inference_out == 4, 52, label_tensor_out)
        label_tensor_out = torch.where(canals_inference_out == 5, 53, label_tensor_out)

        # As the jawbone mask does a good job at leaving gaps for the canals, dilate these to fill.
        jawbone_mask = torch.where(label_tensor_out == 1, 1, 0)

        for canal_value in [3, 4, 103, 104, 105]:
            canal_mask = torch.where(label_tensor_out == canal_value, 1, 0)
            canal_mask = torch_dilate_conv3d(canal_mask, iterations=3)
            label_tensor_out = torch.where(canal_mask == 1, canal_value, label_tensor_out)
        
        # Overwrite with lower jawbone mask to cover over-expansion.
        label_tensor_out = torch.where(jawbone_mask == 1, 1, label_tensor_out)

        if run_in_debug_mode:
            si = SaveImage(output_dir=path_interim_data, output_postfix="canals", separate_folder=False)
            si(label_tensor_out)

            # label_array_out = label_tensor_out.squeeze().cpu().numpy()
            # review_pngs(label_array_out, [0, 1, 2], path_interim_data / "png", f"{case_id}_canals_out")
        
    # For each tooth found - find centres, crop the tooth. (TOOTH_LOOKUP is in the form label_value: label_name).
    # * Currently, just take highest confidence if there's more than one box for that label.
    # * Tooth patch model ROI == 80 x 80 x 80 voxels
    # * For each tooth centre, create a signed distance map from a small sphere at this point to the rest of the patch.

    print("Start individual tooth searches")

    # Load per-tooth model
    tooth_model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        channels=(4, 8, 16, 32, 64),
        strides=(2, 2, 2, 2),
    )

    tooth_model.to(DEVICE)
    
    if DEVICE == torch.device("cpu"):
        tooth_model.load_state_dict(torch.load(model_path_dict["seg_tooth"], weights_only=True, map_location=torch.device('cpu')))
    else:
        tooth_model.load_state_dict(torch.load(model_path_dict["seg_tooth"]))
    
    tooth_model.eval()

    def _run_teeth_finder(tooth_name, row):
        if row == "lower":
            yolo_result = yolo_lower_teeth
            z0_in = z0_lower if z0_lower is not None else None
            z1_in = z1_lower if z1_lower is not None else None
            lookup = LOWER_TEETH_LOOKUP
        else:
            yolo_result = yolo_upper_teeth
            z0_in = z0_upper if z0_upper is not None else None
            z1_in = z1_upper if z1_upper is not None else None
            lookup = UPPER_TEETH_LOOKUP

        label_value = lookup[tooth_name]

        if yolo_result is None or tooth_name not in yolo_result.keys():
            return

        if isinstance(yolo_result[tooth_name]["conf"], list):
            print(f"{case_id} {tooth_name} - {len(yolo_result[tooth_name]['conf'])} boxes found.")
            yolo_result[tooth_name]["box"] = yolo_result[tooth_name]["box"][np.argmax(yolo_result[tooth_name]["conf"])]
            yolo_result[tooth_name]["conf"] = yolo_result[tooth_name]["conf"][np.argmax(yolo_result[tooth_name]["conf"])]
            logging.debug(f"{case_id} {tooth_name} - Max confidence: {yolo_result[tooth_name]['conf']}")

            # Note: Tooth finder image is indexed from the uncropped side in Y, and uncropped in X.

        bb = [(yolo_result[tooth_name]["box"][0] / 640) * im_array_tooth_finder.shape[0],
              (yolo_result[tooth_name]["box"][1] / 640)  * im_array_tooth_finder.shape[1],
              z0_in,
              (yolo_result[tooth_name]["box"][2] / 640) * im_array_tooth_finder.shape[0],
              (yolo_result[tooth_name]["box"][3] / 640) * im_array_tooth_finder.shape[1],
              z1_in
            ]
        
        tooth_centre = [int((bb[0] + bb[3]) / 2), int((bb[1] + bb[4]) / 2), int((bb[2] + bb[5]) / 2)]
       
        x0 = bb[0]
        y0 = bb[1]
        z0 = bb[2]

        x1 = bb[0]
        y1 = bb[1]
        z1 = bb[2]

        z_length = z1 - z0 if z1 - z0 > 80 else 80

        # Pad tooth patch to min ROI (64 x 64 x 64) if necessary.
        pad_x0 = 0 if x0 > 0 else -1 * (tooth_centre[0] - 40)
        pad_y0 = 0 if y0 > 0 else -1 * (tooth_centre[1] - 40)
        pad_z0 = 0 if z0 > 0 else -1 * (tooth_centre[2] - int(z_length / 2))
        pad_x1 = 0 if x1 < im_array_in.shape[0] else (tooth_centre[0] + 40) - im_array_in.shape[0] -1 
        pad_y1 = 0 if y1 < im_array_in.shape[1] else (tooth_centre[1] + 40) - im_array_in.shape[1] -1
        pad_z1 = 0 if z1 < im_array_in.shape[2] else (tooth_centre[2] + int(z_length / 2)) - im_array_in.shape[2] -1

        tooth_image = im_array_in[x0:x1, y0:y1, z0:z1]
        padding = ((pad_x0, pad_x1), (pad_y0, pad_y1), (pad_z0, pad_z1))

        if not all([x == 0 and y == 0 for x, y in padding]):
            tooth_image_pad = np.pad(tooth_image, padding, mode='constant', constant_values=0)
        else:
            tooth_image_pad = tooth_image

        tooth_sd = np.zeros_like(tooth_image_pad)
        tooth_sd[int(tooth_image_pad.shape[0] / 2), int(tooth_image_pad.shape[1] / 2), int(tooth_image_pad.shape[2] / 2)] = 1
        tooth_sd = binary_dilation(tooth_sd, iterations=2).astype(np.uint8)

        itk_tooth_sd = sitk.GetImageFromArray(tooth_sd)
        itk_smdm_out = sitk.SignedMaurerDistanceMap(itk_tooth_sd, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)

        # Clip image to 0-3000; SD to -50-0. Set these to 0-1.
        tooth_image_pad = ((np.clip(tooth_image_pad, a_min=0, a_max=3000)) / 3000)
        tooth_sd = ((np.clip(sitk.GetArrayViewFromImage(itk_smdm_out), a_min=-50, a_max=0)) / 50 ) + 1

        # Concatenate Image and SD along axis 0
        tensor_tooth_im = EnsureChannelFirst()(tooth_image_pad, meta_dict=im.meta)
        tensor_tooth_sd = EnsureChannelFirst()(tooth_sd, meta_dict=im.meta)

        tensor_inputs = torch.cat([tensor_tooth_im, tensor_tooth_sd], dim=0)
        tensor_inputs = EnsureType(dtype=np.float32)(tensor_inputs)

        if run_in_debug_mode:
            review_pngs(tensor_tooth_im.numpy()[0], [0, 1, 2], path_interim_data / "png", f"{case_id}_{row}_{tooth_name}_im_tensor")

        tensor_inputs = tensor_inputs.to(DEVICE)

        with torch.no_grad():
            tensor_inputs = tensor_inputs.unsqueeze(0)
            tensor_output = sliding_window_inference(tensor_inputs, (80, 80, 80), 24, tooth_model, 0.75)

        # Move result to CPU, set values of mask to lookup values
        tooth_model_result_argmax = AsDiscrete(argmax=True)(tensor_output.squeeze())
        tooth_model_conf = tensor_output.squeeze()[1]

        # Un-pad results as necessary:
        if not all([x == 0 and y == 0 for x, y in padding]):
            tooth_model_result_argmax = tooth_model_result_argmax[:, pad_x0:80-pad_x1, pad_y0:80-pad_y1, pad_z0:tooth_image_pad.shape[2]-pad_z1]
            tooth_model_conf = tooth_model_conf[pad_x0:80-pad_x1, pad_y0:80-pad_y1, pad_z0:tooth_image_pad.shape[2]-pad_z1]

        # Back-fill output arrays to full image size
        # Tooth voxel confidence image
        tooth_model_conf_out = torch.zeros_like(im, device=DEVICE)
        tooth_model_conf_out[x0:x1, y0:y1, z0:z1] = torch.where(tooth_model_conf > 0, tooth_model_conf, 0)

        # Pulp and tooth labels
        # Use binary closing (erosion of dilation) to tidy up pulp segmentations
        pulp_mask = torch.where(tooth_model_result_argmax == 2, 1, 0)
        pulp_mask = torch_dilate_conv3d(pulp_mask, iterations=2)
        pulp_mask = torch_erode_conv3d(pulp_mask, iterations=2)
        tooth_model_result_argmax = torch.where(pulp_mask == 1, 2, tooth_model_result_argmax)

        # Un-crop, set tooth and pulp to their label values for output.
        tooth_model_result_argmax_out = torch.zeros_like(im, device=DEVICE)
        tooth_model_result_argmax_out[x0:x1, y0:y1, z0:z1] = tooth_model_result_argmax

        tooth_mask = torch.where(tooth_model_result_argmax == 1, 1, 0)
        tooth_mask_cc = label(tooth_mask.to("cpu").squeeze())
        
        if np.max(tooth_mask_cc) > 0:
            tooth_mask_largest_cc = tooth_mask_cc == np.argmax(np.bincount(tooth_mask_cc.flat)[1:])+1
            tooth_label_out = torch.from_numpy(tooth_mask_largest_cc).unsqueeze(0).to(DEVICE)
        else:
            tooth_label_out = tooth_mask

        tooth_label_out = torch.where(tooth_model_result_argmax_out == 1, label_value, 0)
        pulp_label_out = torch.where(tooth_model_result_argmax_out == 2, PULP_LABEL, 0)

        

        # if DEBUG:
            # review_pngs(tooth_model_result_argmax_out, [0, 1, 2], path_interim_data / "png", f"{case_id}_{tooth_name}_{row}_argmax")

        # Fix orientation of masks/images out
        tooth_model_conf_out = Orientation(axcodes="RAS")(tooth_model_conf_out.unsqueeze(0))
        tooth_label_out = Orientation(axcodes="RAS")(tooth_label_out.unsqueeze(0))
        pulp_label_out = Orientation(axcodes="RAS")(pulp_label_out.unsqueeze(0))

        return {
            "tooth": tooth_name,
            "target_label": label_value,
            "tooth_model_conf": tooth_model_conf_out.squeeze().detach().type(torch.half),
            "tooth_label": tooth_label_out.squeeze().detach().type(torch.int8),
            "pulp_label": pulp_label_out.squeeze().detach().type(torch.int8)
        }

    # TODO: Make parallel.
    # TODO: Lock file for tooth model?
    tooth_results = {}

    for row in ["upper", "lower"]:
        for v in tqdm(YOLO_TEETH_LOOKUP.values()):
            result = _run_teeth_finder(v, row)

            if result is not None:
                tooth_results[result["target_label"]] = result
    
    # for d in tooth_results.values():
    #     label_tensor_out = label_tensor_out + d["tooth_label"] + d["pulp_label"]

    # Manage overlapping tooth labels
    for k, d in tooth_results.items():
        # If no neighbours, no changes.
        if k not in TOOTH_TO_NEIGHBOUR_LOOKUP.keys() or TOOTH_TO_NEIGHBOUR_LOOKUP[k] not in tooth_results.keys():
            label_tensor_out = label_tensor_out + d["tooth_label"] + d["pulp_label"]
        else:
            di = tooth_results[TOOTH_TO_NEIGHBOUR_LOOKUP[k]]
            overlap_mask = torch.logical_and(d["tooth_label"], di["tooth_label"])

            # If no overlap, no changes.
            if torch.max(overlap_mask) == 0:
                label_tensor_out = label_tensor_out + d["tooth_label"] + d["pulp_label"]
            else:
                d_conf_overlap = d["tooth_model_conf"] * overlap_mask
                di_conf_overlap = di["tooth_model_conf"] * overlap_mask

                # If the confidence for the "other tooth" is higher for a given voxel, set that voxel to zero 
                # for this tooth. Otherwise, keep the voxel.
                d["tooth_label"] = torch.where(di_conf_overlap > d_conf_overlap, 0, d["tooth_label"])
                di["tooth_label"] = torch.where(d_conf_overlap > di_conf_overlap, 0, di["tooth_label"])

                # Then, add this tooth (and pulp) to the output tensor.
                label_tensor_out = label_tensor_out + d["tooth_label"] + d["pulp_label"]

    if run_in_debug_mode:
        label_array_out = label_tensor_out.cpu().squeeze().numpy()
        review_pngs(label_array_out, [0, 1, 2], path_interim_data / "png", f"{case_id}_all_labels")
    
        # Output review image
        si = SaveImage(output_dir=path_interim_data, output_postfix="all_labels", separate_folder=False)
        si(label_tensor_out)

    print(f"{case_id} complete")
    return label_tensor_out


def main():
    path_case_ids_yaml = Path("C:/data/tf3/case_id_lists.yaml")
    path_data_dir = Path("C:/data/tf3/images_rolm")
    path_output_data = Path("C:/data/tf3/test_pipeline_out")

    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    test_cases = d_case_ids["test"]
    list_images = ([path_data_dir / f"{x}.nii.gz" for x in test_cases])

    model_paths = {
        "yolo_axis_lr": Path("tf3_evaluation/algorithm/yolo_localiser_640_axis_0.pt"),
        "yolo_axis_ap": Path("tf3_evaluation/algorithm/yolo_localiser_640_axis_1.pt"),
        "yolo_lower_teeth": Path("tf3_evaluation/algorithm/yolo_tooth_finder_lower_jaw.pt"),
        "yolo_upper_teeth": Path("tf3_evaluation/algorithm/yolo_tooth_finder_upper_jaw.pt"),
        "seg_tooth": Path("tf3_evaluation/algorithm/per_tooth_unet_80_80_80.pkl"),
        "seg_large_anatomy": Path("tf3_evaluation/algorithm/non_tooth_anatomy_80_80_80.pkl"),
        "seg_canals": Path("tf3_evaluation/algorithm/canals_from_jawbone_64_64_64.pkl"),
    }

    for path_image in list_images:
        if not (path_output_data / f"{path_image.name.split(".")[0]}_all_labels.nii.gz").exists():
            segment_one_image(path_image, path_output_data, model_paths, run_in_debug_mode=True)


if __name__ == "__main__":
    main()


