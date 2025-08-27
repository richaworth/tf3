FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output /output/metadata /output/tmp /opt/app/models/seg_models /opt/app/models/yolo_models \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install --user -r requirements.txt
RUN python -m pip install "monai[nibabel, tifffile, pynrrd, pydicom, einops]"

# RUN apt-get update && apt-get install -y python3-opencv
RUN python -m pip install opencv-python-headless
RUN python -m pip install triton

RUN python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip check torch
RUN python -m pip check torchvision

# COPY --chown=user:user per_tooth_unet_best_metric_epoch_50.pkl /opt/app/models/seg_models/
COPY --chown=user:user per_tooth_unet_64_64_64_best_metric_epoch_70.pkl /opt/app/models/seg_models/
COPY --chown=user:user non_tooth_anatomy_80_80_80.pkl /opt/app/models/seg_models/
COPY --chown=user:user canals_from_jawbone_64_64_64.pkl /opt/app/models/seg_models/
COPY --chown=user:user yolo_localiser_640_axis_0.pt /opt/app/models/yolo_models/
COPY --chown=user:user yolo_localiser_640_axis_1.pt /opt/app/models/yolo_models/
COPY --chown=user:user yolo_tooth_finder_lower_jaw.pt /opt/app/models/yolo_models/
COPY --chown=user:user yolo_tooth_finder_upper_jaw.pt /opt/app/models/yolo_models/

COPY --chown=user:user __init__.py /opt/app/
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user tf3_seg_pipeline_final.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
