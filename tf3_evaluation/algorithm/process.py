from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import json
from typing import Dict, Tuple

from tf3_seg_pipeline_final import segment_one_image

from evalutils import SegmentationAlgorithm
from evalutils.validators import UniqueImagesValidator, UniquePathIndicesValidator

def get_default_device():
    """Set device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class ToothFairy3_OralPharyngealSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path('/input/images/cbct/'),
            output_path=Path('/output/images/oral-pharyngeal-segmentation/'),
            )
        
        # Create output directory if it doesn't exist
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)
        
        # Create metadata output directory
        self.metadata_output_path = Path('/output/metadata/')
        if not self.metadata_output_path.exists():
            self.metadata_output_path.mkdir(parents=True)
        
        # Initialize device
        self.device = get_default_device()
        print(f"Using device: {self.device}")

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_nodules = self.predict(path_input_image=input_image_file_path, sitk_image_loaded=input_image)

        # Write resulting segmentation to output location
        segmentation_path = self._output_path / input_image_file_path.name
        if not self._output_path.exists():
            self._output_path.mkdir()
        sitk.WriteImage(segmented_nodules, str(segmentation_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    @torch.no_grad()
    def predict(self, *, path_input_image: Path, sitk_image_loaded: sitk.Image) -> sitk.Image:
        models = {
            "yolo_axis_lr": Path("/opt/app/models/yolo_models/yolo_localiser_640_axis_0.pt"),
            "yolo_axis_ap": Path("/opt/app/models/yolo_models/yolo_localiser_640_axis_1.pt"),
            "yolo_lower_teeth": Path("/opt/app/models/yolo_models/yolo_tooth_finder_lower_jaw.pt"),
            "yolo_upper_teeth": Path("/opt/app/models/yolo_models/yolo_tooth_finder_upper_jaw.pt"),
            "seg_tooth": Path("/opt/app/models/seg_models/per_tooth_unet_64_64_64_best_metric_epoch_70.pkl"),
            "seg_large_anatomy": Path("/opt/app/models/seg_models/non_tooth_anatomy_80_80_80.pkl"),
            "seg_canals": Path("/opt/app/models/seg_models/canals_from_jawbone_64_64_64.pkl"),
        }
            
        output_array = segment_one_image(path_input_image, Path("/output/tmp"), models)
        output_array = output_array.detach().cpu().squeeze()
        
        output_array = np.flip(output_array, [0, 1])        # Fix sitk rotation of 180* around Z
        output_array = np.transpose(output_array, (2,1,0))  # Switch from numpy (k, j, i) to sitk (i, j, k) ordering.

        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(sitk_image_loaded)

        return output_image
        

if __name__ == "__main__":
    ToothFairy3_OralPharyngealSegmentation().process()
