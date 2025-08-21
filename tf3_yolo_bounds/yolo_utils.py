from pathlib import Path

import numpy as np


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
    
def calculate_2d_bounds_as_xywh(label_array: np.ndarray, 
                                labels: int | list[int], 
                                axes: int | list[int] = 0) -> tuple[float, float, float, float] | list[tuple[float, float, float, float]] | None:
    """
    Calculate the 2D bounds of one or more given labels (combined) within a 3D mask, in YOLO-compatible
    X centre, Y centre, width, height format (as fraction of the equivalent axis' MIP image.)

    Args:
        np_img (np.ndarray): Numpy array of mask (2d or 3d)
        labels (int | list[int]): Label or labels from which to calculate bounds. Will combine these
        axes (int | list[int], optional): Axis or axes along which to calculate bounds. Ignored if numpy array is 2D.
            Defaults to 0.

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
        # Allow for 2D or 3D
        mask_2d = mask if len(mask.shape) == 2 else np.max(mask, axis = a)

        # If there's nothing in the final mask, return all None
        if not np.any(mask_2d):
            return None
        
        arr = np.where(mask_2d != 0)
        i0, i1, j0, j1 = np.min(arr[0]), np.max(arr[0]), np.min(arr[1]), np.max(arr[1]) 
        fi0, fi1, fj0, fj1 = i0 / mask_2d.shape[1], i1 / mask_2d.shape[1], j0 / mask_2d.shape[0], j1 / mask_2d.shape[0]  
        
        # Ends up that i, j == Y, X axes once saved in png. As such:
        x = (fj0 + fj1) /2
        y = (fi0 + fi1) /2
        w = fj1 - fj0
        h = fi1 - fi0
        
        output.append((x, y, w, h))
    
    if len(output) == 1:
        return output[0]
    else:
        return output

