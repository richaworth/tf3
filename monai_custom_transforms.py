
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
