from ultralytics.data.utils import visualize_image_annotations

label_map = {
    0: "lower_jaw",
    1: "upper_jaw",
    2: "lower_teeth",
    3: "upper_teeth"
}

visualize_image_annotations(
    "C:/data/yolo_localiser_data_640/train/axis_0/images/ToothFairy3F_002.png",
    "C:/data/yolo_localiser_data_640/train/axis_0/labels/ToothFairy3F_002.txt",
    label_map=label_map
)
