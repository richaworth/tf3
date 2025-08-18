# https://faelnl.medium.com/training-a-yolov8-object-segmentation-model-for-your-needs-9fe845e2ff77

import os
import cv2
import yaml
import shutil
import pandas as pd
import polars as pl
import multiprocessing
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from shapely.geometry import Polygon
from pathlib import Path
from matplotlib import pyplot as plt

# base path must be the same as our previous dataset_path
path_label_images = Path("C:/data/yolo_localiser_data_640/train/axis_0/label_review")
path_output_data = Path("C:/data/yolo_localiser_data_640/train/axis_0/label_contours")

### Mask to Poly ###

def mask_to_polygon(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        polygon = contour.reshape(-1, 2)
        polygon_norm = polygon.astype(float)
        polygon_norm[:, 0] /= mask.shape[1]  # X
        polygon_norm[:, 1] /= mask.shape[0]  # Y
        polygon_norm = np.round(polygon_norm, 4)

        polygon_shapely = Polygon(polygon_norm)
        polygon_simplified = polygon_shapely.simplify(0.002, preserve_topology=True)
        polygons.append(polygon_simplified)

    return polygons

def polygon_to_yolo(polygon):
    x, y = polygon.exterior.coords.xy
    xy = []
    for xx, yy in zip(x, y):
        xy.append(xx)
        xy.append(yy)
    return xy

def polygon_to_mask(polygon, shape):
    mk = np.zeros(shape, dtype=np.uint8)
    x, y = polygon.exterior.coords.xy
    xy = [
        [int(xx * shape[1]), int(yy * shape[0])]
        for xx, yy in zip(x, y)
    ]
    cv2.fillConvexPoly(mk, np.array(xy, dtype='int32'), color=255)
    return mk

