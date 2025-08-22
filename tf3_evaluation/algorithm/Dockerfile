FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output /output/metadata /output/tmp /opt/app/models/seg_models /opt/app/models/yolo_models \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/

COPY --chown=user:user per_tooth_unet_best_metric_epoch_50.pkl /opt/app/models/seg_models/
COPY --chown=user:user non_tooth_anatomy_80_80_80.pkl /opt/app/models/seg_models/
COPY --chown=user:user canals_from_jawbone_64_64_64.pkl /opt/app/models/seg_models/
COPY --chown=user:user yolo_localiser_640_axis_0.pt /opt/app/models/yolo_models/
COPY --chown=user:user yolo_localiser_640_axis_1.pt /opt/app/models/yolo_models/
COPY --chown=user:user yolo_tooth_finder_lower_jaw.pt /opt/app/models/yolo_models/
COPY --chown=user:user yolo_tooth_finder_upper_jaw.pt /opt/app/models/yolo_models/

RUN python -m pip install --user -r requirements.txt

# RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python-headless
RUN pip install "monai[all]"

COPY --chown=user:user __init__.py /opt/app/
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user tf3_seg_pipeline_final.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
