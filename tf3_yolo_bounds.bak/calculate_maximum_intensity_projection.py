import numpy as np
import nibabel
from pathlib import Path
from skimage.transform import resize

import imageio

def calculate_mip(np_img: np.ndarray, axes: int | list[int] = 0) -> np.ndarray | list[np.ndarray]:
    """
    Calculate MIP from original image along given axis.

    Args:
        np_img (np.ndarray): Numpy array of image (3d)
        axis (int): Axis along which to calculate MIP.
    
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

def calculate_2d_bounds_as_fraction(np_img: np.ndarray, 
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
    
    mask = np.zeros_like(np_img)
        
    for lab in labels:
        mask = np.where(np_img == lab, 1, mask)

    for a in axes:
        mask_2d = np.max(np_img, axis = a)

        # If there's nothing in the final mask, return all zeros
        if not np.any(mask_2d):
            return None
        
        a = np.where(mask_2d != 0)
        i0, i1, j0, j1 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        fi0, fi1, fj0, fj1 = i0 / mask_2d.shape[0], i1 / mask_2d.shape[0], j0 / mask_2d.shape[1], j1 / mask_2d.shape[1]
        
        output.append((fi0, fi1, fj0, fj1))
    
    if len(output) == 1:
        return output[0]
    else:
        return output

def main(path_images_in: Path = Path("C:/data/tf3/images_rolm"), 
         path_labels_in: Path = Path("C:/data/tf3/labels_rolm"), 
         path_yolo_out: Path = Path("C:/data/tf3/yolo_out"), 
         target_size: int = 512):
    """_summary_

    Args:
        path_images_in (Path): _description_
        path_labels_in (Path): _description_
        path_yolo_out (Path): _description_
        target_size (int, optional): _description_. Defaults to 512.
    """
    path_yolo_out.mkdir(exist_ok=True, parents=True)

    for path_image in list(path_images_in.glob("*.nii.gz")):  # TODO - remove [:10]
        case_id = path_image.name.split(".nii.gz")[0]
        image = nibabel.load(path_image)
        im_labels = nibabel.load(path_labels_in / f"{case_id}.nii.gz")

        mip_axes = [0, 1, 2]

        im_array = image.get_fdata()
        lab_array = im_labels.get_fdata()

        im_array_largest = np.max(im_array.shape)

        pad_max = [im_array_largest - x for x in im_array.shape]
        pad_start = [int(x / 2) for x in pad_max]
        pad_end = [x - pad_start[i] for i, x in enumerate(pad_max)]

        # Clip im_array to [-1000, 3000] == [0, 255] (for saving as png)
        im_array = np.clip(im_array, -1000, 3000)
        im_array = (im_array + 1000) * (255 / 4000)

        im_square = np.pad(im_array, ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])))
        lab_square = np.pad(lab_array, ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])))

        mips = calculate_mip(im_square, axes=mip_axes)

        bounds_labels = [1, 2, 3, 4]

        bounds = calculate_2d_bounds_as_fraction(lab_square, bounds_labels, axes=[0, 1, 2])

        # Save pngs, yolo text files.
        for a in mip_axes:
            mip_out = resize(mips[a], (target_size, target_size))
            imageio.imwrite(path_yolo_out / f"{case_id}_axis_{a}.png", mip_out.astype('uint8'))  # TODO - fix

            # Remove existing yolo bounds file if it exists to prevent accidental double-writing.
            if (path_yolo_out / f"{case_id}_axis_{a}.txt").exists():
                (path_yolo_out / f"{case_id}_axis_{a}.txt").unlink()

            for lab, bb in zip(bounds_labels, bounds):
                if bb is None:
                    print(f"{path_image.name} - No bounds available for {lab}")
                else:
                    line = f"{path_yolo_out / f"{case_id}_axis_{a}.png"} {lab} bounds: {bb}\n"                   
                    with (path_yolo_out / f"{case_id}_axis_{a}.txt").open("a") as f:
                        f.write(line)
   
    
if __name__ == "__main__":
    main()
