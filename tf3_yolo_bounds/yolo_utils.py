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