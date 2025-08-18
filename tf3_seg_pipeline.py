from types import SimpleNamespace

import imageio
import nibabel
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from skimage.transform import resize
import SimpleITK as sitk

import yaml

from tf3_yolo_bounds.yolo_utils import yolo_bounds_from_image, calculate_mip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def segment_one_image(im_array_in: np.ndarray, case_id: str, path_interim_data: Path, models: dict) -> np.ndarray:
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
    # Note: Due to the way MIP images are created, Z is axis 1 (Y) in LR images and axis 0 (X) in AP images.
    
    path_image_mip_axis_lr = path_interim_data / "yolo_1" / f"axis_lr" / f"{case_id}_axis_lr_in.png"
    path_image_mip_axis_ap = path_interim_data / "yolo_1" / f"axis_ap" / f"{case_id}_axis_ap_in.png"
    
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

    print(lr_model_result)
    print(ap_model_result)

    print()




    # Use results to construct upper and lower teeth images
    # Run YOLO tooth finder on these

    # Run overall anatomy model

    # Run bridge/implant finder (cropped to lower jaw box)
    # Run canal finder (cropped to lower jaw box)

    # Get centroid of each tooth from YOLO results, run per-tooth model on patch centred at these
    # * Check if any centre is within a bridge/crown/implant
    # * Check how exactly patch images were created for these
    # * Ensure edge handling matches.
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

    for path_image in list_images[:1]:
        im = nibabel.load(path_image)
        im_array = im.get_fdata()

        segment_one_image(im_array, path_image.name.split(".nii.gz")[0], path_output_data, model_paths)


if __name__ == "__main__":
    main()


