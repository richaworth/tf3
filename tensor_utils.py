import torch
import numpy as np

def torch_dilate_conv(tensor_in: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    assert len(tensor_in.shape) == 4 or len(tensor_in.shape) == 5, f"Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: {tensor_in.shape}" 

    # Set up dilation kernel (all ones, 3x3x3 cube)
    kernel = np.ones((1, 1, 3, 3, 3), np.float32)
    kernel_tensor = torch.Tensor(kernel)

    # If tensor has only 4 dimensions, unsqueeze before diation, squeeze after.
    if len(tensor_in.shape) == 4:
        tensor_in = tensor_in.unsqueeze(0)
        squeeze_result = True
    else:
        squeeze_result = False

    for _ in range(iterations):
        torch_result = torch.clamp(torch.nn.functional.conv3d(tensor_in, kernel_tensor, padding=1), 0, 1)
    
    return torch_result.squeeze() if squeeze_result else torch_result

def torch_erode_conv(tensor_in, iterations) -> torch.Tensor:
    assert len(tensor_in.shape) == 4 or len(tensor_in.shape) == 5, f"Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: {tensor_in.shape}" 
    
    # Set up kernel for erosion 
    kernel = np.array(
        [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
         [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.float32)
    
    kernel_tensor = torch.Tensor(kernel)

    # If tensor has only 4 dimensions, unsqueeze before diation, squeeze after.
    if len(tensor_in.shape) == 4:
        tensor_in = tensor_in.unsqueeze(0)
        squeeze_result = True
    else:
        squeeze_result = False

    for _ in range(iterations):
        torch_result = torch.clamp(torch.nn.functional.conv3d(tensor_in, kernel_tensor, padding=1), 0, 1)
    
    return torch_result.squeeze() if squeeze_result else torch_result


# def torch_erode_conv(tensor_in, iterations) -> torch.Tensor:
#     kernel = np.array(
#         [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#          [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.float32)
        
#     if len(tensor_in.shape) < 5:
#         while len(kernel.shape) < len(tensor_in.shape):
#             kernel = np.expand_dims(kernel, 0)
#             padding.append(1)
#             stride.append(1)
#             dilation.append(1)
        
#     kernel_tensor = torch.Tensor(kernel)
#     torch_result = tensor_in

#     for _ in range(iterations):
#         torch_result = torch.clamp(torch.nn.functional.conv3d(torch_result, kernel_tensor, padding=padding, stride=stride, dilation=dilation), 0, 1)

#     return torch_result

def test_erode_dilate():
    # Create tensor with seed points.
    im = np.zeros((1, 4, 4, 4), dtype=np.float32)
    im[0, 1, 1, 1] = 1
    im[0, 3, 3, 2] = 1

    im_t_4d = torch.Tensor(im)
    print(im_t_4d)

    im_t_4d_dil = torch_dilate_conv(im_t_4d, 1)
    print(im_t_4d_dil)

    im_t_5d = torch.Tensor(np.expand_dims(im, 0))
    print(im_t_4d)

    im_t_4d_dil = torch_dilate_conv(im_t_4d, 2)
    print(im_t_4d_dil)

    # im_t_ero = torch_erode_conv(im_t_dil, 1)
    # print(im_t_ero)

    # print(im_t_ero == im_t)

    # im_t_dil = torch_dilate_conv(im_t, 2)
    # print(im_t_dil)
    # im_t_ero = torch_erode_conv(im_t_dil, 2)
    # print(im_t_ero)

    # print(im_t_ero == im_t)


if __name__ == "__main__":
    test_erode_dilate()
