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

4. Image + jawbone seg -> landmark finding network - if landmarks can be at the tooth/bone interface, this may be best (probably handles missing teeth reasonably)
    - TODO: Requires per-tooth landmark labels (locate missing - probably manually or by warping).

5. Image + jawbone seg + landmarks -> Per-tooth segmentation - one tooth model (or a small number of models), cortex and pulp, applied at each landmark.
    - TODO: Requires per-tooth cropped label images + related CT. 
    - TODO: Needs to be trained.
    - Note: If landmark is within an implant/bridge, do not segment.
    - TODO: Needs to be trained.

6. Combine results and output.


