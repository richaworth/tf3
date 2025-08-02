
from monai.config.type_definitions import KeysCollection
from monai.transforms import MapTransform
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

import torch

class EditLabelsd(MapTransform):
    """
    Change the labels in an image using a lookup list of tuples.
    """
    def __init__(self, keys: KeysCollection, list_old_new_labels: list[tuple]) -> None:
        super().__init__(keys)
        self.list_on_labels = list_old_new_labels

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = torch.zeros_like(d[key])

            for old, new in self.list_on_labels:
                result = torch.where(d[key] == old, new, result)
            
            d[key] = result
            
        return d

# TODO: This doesn't work, but isn't currently necessary.
# class BinaryClosingd(MapTransform):
#     """
#     Apply binary closing (N erosions followed by N dilations) to one or more labels.
#     """
#     def __init__(self, keys: KeysCollection, list_target_labels: int | list[tuple], iterations: int = 2) -> None:
#         super().__init__(keys)
#         self.target_labels = [list_target_labels] if not isinstance(list_target_labels, list) else list_target_labels
#         self.iterations = iterations

#     def __call__(self, data):
#         d = dict(data)

#         for key in self.keys:
#             # Erosion must be done in CPU if using scikit. 
#             label_image = d[key].detach().cpu().numpy()
#             result = np.zeros_like(label_image)

#             # For each target label - get that label as a binary mask; apply erosions; apply dilations, add to the results image with original label value.
#             for label in self.target_labels:
#                 a = np.where(label_image == label, 1, 0)
#                 a = binary_erosion(a, iterations=self.iterations)
#                 a = binary_dilation(a, iterations=self.iterations)
                
#                 result = np.where(a, label, result)

#             d[key] = torch.from_numpy(result)
            
#         return d

