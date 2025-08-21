from ultralytics import YOLO
from pathlib import Path

def main():
    # for axis in [0, 1]:
    for axis in [1]:  # Note - axis 0 has completed 400 epochs.
        data_yaml = Path(f"C:/code/python/tf3/tf3_yolo_bounds/mip_bounds_yolo_axis_{axis}.yaml")
        output_dir = Path(f"C:/data/tf3/yolo_init_out/axis_{axis}")
        output_dir.mkdir(exist_ok=True, parents=True)

        # TODO = remove resume bodging
        model_in = output_dir / "train" / "weights" / "last.pt" if (output_dir / "train" / "weights" / "last.pt").exists() else "yolo11l.pt"
        training_epochs = 400 if model_in == "yolo11l.pt" else 100
        
        model = YOLO(model_in, task="detect", verbose=True)
        result = model.train(data=data_yaml, 
                             epochs=training_epochs, 
                             imgsz=640, 
                             batch=-1, 
                             seed=1, 
                             overlap_mask=True, 
                             save_period=50, 
                             project=output_dir,
                             flipud=0,          # Invert Y - given we want to preserve the geometry, remove this.
                             fliplr=0,          # Invert X - given we've already done this manually, remove this.
                             scale=0.1,         # Scale - given we want to preserve the geometry, only allow a tiny amount of this.
                             shear=10,          # Shear - given we want to preserve the geometry, only allow a tiny amount of this.
                             degrees=0,         # Rotation - given we want to preserve the geometry, remove this.
                             translate=0.05,    # Translation - given we want to preserve the geometry, only allow a tiny amount of this.
                             mosaic=0,          # Mosaic several images together - given we want to preserve the geometry, remove this.
                             hsv_h=0,           # Hue - not useful for greyscale images.
                             hsv_s=0,           # Saturation - not useful for greyscale images.
                             hsv_v=0.1,         # Brightness - given CBCT is in standardised units (HU), tweaking this more than marginally is pointless.
                             )

    """
    Full verbose set of arguments going into the model:
    agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=-1, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, 
    close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, 
    data=C:\code\python\tf3\tf3_yolo_bounds\mip_bounds_yolo_axis_0.yaml, degrees=10, deterministic=True, 
    device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=400, erasing=0.4, exist_ok=False, 
    fliplr=0, flipud=0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0, hsv_s=0, hsv_v=0.1, imgsz=640, 
    int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, 
    mode=train, model=yolo11l.pt, momentum=0.937, mosaic=0, multi_scale=False, name=train, nbs=64, nms=False, opset=None, 
    optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, 
    profile=False, project=C:\data\tf3\yolo_init_out\axis_0, rect=False, resume=False, retina_masks=False, save=True, 
    save_conf=False, save_crop=False, save_dir=C:\data\tf3\yolo_init_out\axis_0\train, save_frames=False, save_json=False, 
    save_period=1, save_txt=False, scale=0.1, seed=1, shear=10, 
    show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, 
    stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, 
    visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
    """

if __name__ == "__main__":
    main()
