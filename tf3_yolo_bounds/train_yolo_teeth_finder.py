from ultralytics import YOLO
from pathlib import Path

def main():
    for set in ["upper", "lower"]:
        data_yaml = Path(f"C:/code/python/tf3/tf3_yolo_bounds/mip_teeth_yolo_dataset_{set}.yaml")
        output_dir = Path(f"C:/data/tf3/yolo_teeth_finder/{set}")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        model = YOLO("yolo11l.pt", task="detect", verbose=True)
        result = model.train(data=data_yaml, 
                             epochs=800, 
                             patience=50,
                             imgsz=640, 
                             batch=20, 
                             seed=1, 
                             overlap_mask=False, 
                             save_period=50, 
                             project=output_dir,
                             flipud=0,           # Invert Y - given we want to preserve the geometry, remove this.
                             fliplr=0,           # Invert X - given we've already done this manually, remove this.
                             scale=0.1,          # Scale - given we want to preserve the geometry, only allow a tiny amount of this.
                             shear=10,           # Shear - given we want to preserve the geometry, only allow a tiny amount of this.
                             degrees=10,        # Rotation - given we want to preserve the geometry, only allow a tiny amount of this.
                             translate=0.05,   # Translation - given we want to preserve the geometry, only allow a tiny amount of this.
                             mosaic=0,           # Mosaic several images together - given we want to preserve the geometry, remove this.
                             hsv_h=0,            # Hue - not useful for greyscale images.
                             hsv_s=0,            # Saturation - not useful for greyscale images.
                             hsv_v=0.1,          # Brightness - given CBCT is in standardised units (HU), tweaking this more than marginally is pointless. 
                             lr0=0.01,
                             lrf=0.00001
                             )

if __name__ == "__main__":
    main()
