from pathlib import Path
import shutil

def main():
    path_data_dirs = [
        Path("C:/data/tf3/yolo_localiser_data_640/train/axis_0"),
        Path("C:/data/tf3/yolo_localiser_data_640/test/axis_0"),
        Path("C:/data/tf3/yolo_localiser_data_640/val/axis_0"),
        Path("C:/data/tf3/yolo_localiser_data_640/train/axis_1"),
        Path("C:/data/tf3/yolo_localiser_data_640/test/axis_1"),
        Path("C:/data/tf3/yolo_localiser_data_640/val/axis_1"),
        Path("C:/data/tf3/yolo_teeth_data_640/train/lower_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640/test/lower_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640/val/lower_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640/train/upper_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640/test/upper_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640/val/upper_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640_3mm_slices/train/lower_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640_3mm_slices/test/lower_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640_3mm_slices/val/lower_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640_3mm_slices/train/upper_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640_3mm_slices/test/upper_teeth"),
        Path("C:/data/tf3/yolo_teeth_data_640_3mm_slices/val/upper_teeth")
    ]

    for path_data_dir in path_data_dirs:
        if not (path_data_dir / "labels.bak").exists():
            shutil.move(path_data_dir / "labels", path_data_dir / "labels.bak")
            (path_data_dir / "labels").mkdir(parents=True, exist_ok=True)

        for path_label_txt in (path_data_dir / "labels.bak").glob("*.txt"):
            file_name = path_label_txt.name

            with path_label_txt.open("r") as f:
                for line in f.readlines():
                    label, x0, x1, y0, y1 = line.split()

                    xc = (float(x0) + float(x1)) / 2
                    yc = (float(y0) + float(y1)) / 2
                    w = float(x1) - float(x0)
                    h = float(y1) - float(y0)

                    with (path_data_dir / "labels" / file_name).open("a") as f_out:
                        f_out.write(f"{label}\t{xc:.4f}\t{yc:.4f}\t{w:.4f}\t{h:.4f}\n")





if __name__ == "__main__":
    main()
