import torch
import numpy as np

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


def _test_erode_dilate():
    """
    Toy code to test erosion, dilation.
    """
    # Create tensor with seed points.
    im = np.zeros((1, 4, 4, 4), dtype=np.float32)
    im[0, 1, 1, 1] = 1
    im[0, 3, 3, 2] = 1

    print("Test 4D tensor")
    im_t_4d = torch.Tensor(im)
    # print(im_t_4d)

    im_t_4d_dil = torch_dilate_conv3d(im_t_4d, 1)
    print(im_t_4d_dil)
    im_t_4d_ero = torch_erode_conv3d(im_t_4d_dil, 1)
    print(im_t_4d_ero)

    print("Test 5D tensor")
    im_t_5d = torch.Tensor(np.expand_dims(im, 0))
    print(im_t_5d)

    im_t_5d_dil = torch_dilate_conv3d(im_t_5d, 2)
    print(im_t_5d_dil)
    im_t_5d_ero = torch_erode_conv3d(im_t_5d_dil, 2)
    print(im_t_5d_ero)

    # Create tensor with non-binary values for assertion test.
    im_bad = im * 0.5
    im_bad_t = torch.Tensor(im_bad)

    try:
        _ = torch_dilate_conv3d(im_bad_t, 2)
    except AssertionError as error:
        print(f"Failure (expected) due to {error}")

if __name__ == "__main__":
    _test_erode_dilate()
