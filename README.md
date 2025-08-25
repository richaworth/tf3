# tf3
MICCAI GC 2025 - Toothfairy 3

Deadline for submission - 24th August 2025

! Pipeline
Approach is based loosely on "Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net" 
(Payer et al 2020) (https://cpb-ap-se2.wpmucdn.com/blogs.auckland.ac.nz/dist/1/670/files/2020/06/2020PayerVISAPP.pdf)

1. Image -> Localisation model (bones+teeth) - get region of interest bounds.
    - YOLO Object detection in 2D, using maximum intensity projection along X and Y axes.

2. Image and X/Y bounds -> Tooth bounding boxes - get region of interest bounds.
    - YOLO Object detection in 2D, using maximum intensity projection over the bounds from 1, along Z axis.

3. Image -> Maxilla/Mandible and Non-bony structures (sinuses, pharynx) segmentation 
    - UNETR of 3D images. Resolution slightly lowered due to memory issues.

4. Image + jawbone seg -> nerve canals
    - UNETR of 3D images + mandible mask. Resolution slightly lowered due to memory issues.

5. Image + teeth bounds (+ signed distance image to centre of tooth) -> individual teeth.
    - Unet of tooth images + signed distance to centre of tooth.

6. Combine results and output.


