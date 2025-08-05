# tf3
MICCAI GC 2025 - Toothfairy 3

Deadline for submission - 15th August 2025

! Pipeline
Using the same kind of approach as described in "Coarse to Fine Vertebrae Localization and Segmentation
with SpatialConfiguration-Net and U-Net" Payer et al 2020 (https://cpb-ap-se2.wpmucdn.com/blogs.auckland.ac.nz/dist/1/670/files/2020/06/2020PayerVISAPP.pdf)

1. Image -> Localisation model (bones OR bones+teeth) - get region of interest bounds, general shape.
    - Requires images, bones (or bones+teeth) labels (potentially as signed distance images)
    - Currently training.

2. Image + localisation -> Jawbone segmentation (localisation result for bones + image in; bones out)
    - TODO: Needs to be trained.

3. Image + jawbone seg -> non-bony structures (canals, sinuses, implants)
    - TODO: Non-bony structure label images to be created. Potentially two sets.
    - TODO: Needs to be trained.

4. Image + teeth from localiser -> landmark finding network
    - TODO: Construct an image full of -np.inf (will be clipped before training to zero) for any case without each given tooth
    - Tight cropping around the teeth localiser    
    - TODO: Needs to be written.
    - TODO: Needs to be trained.

5. Image + jawbone seg + landmarks -> Per-tooth segmentation - one tooth model (or a small number of models), cortex and pulp, applied at each landmark.
    - Requires per-tooth cropped label images + related CT (created)
    - TODO: Needs to be trained.
    - Note: If landmark is within an implant/bridge, do not segment.

6. Combine results and output.


