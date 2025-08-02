from pathlib import Path
import yaml
import random

def main(path_data_dir = Path("C:/data/tf3")):
    path_images = path_data_dir / "images_rolm"
    path_labels = path_data_dir / "labels_rolm"
    path_case_ids_yaml = path_data_dir / "case_id_lists.yaml"
    path_case_ids_yaml_10 = path_data_dir / "case_id_lists_10_cases.yaml"

    seed = 0
    random.seed(seed)

    assert not path_case_ids_yaml.exists(), f"{path_case_ids_yaml} exists - DO NOT OVERWRITE"

    # Get case IDs, shuffle and distribute into train/test/validate - 70%/15%/15% split. 
    case_ids = [label.name.split("_mirrored.nii.gz")[0] for label in path_labels.glob("*_mirrored.nii.gz")]
    assert all([(path_images / f"{case_id}.nii.gz").exists() for case_id in case_ids])
    assert all([(path_labels / f"{case_id}.nii.gz").exists() for case_id in case_ids])
    assert all([(path_labels / f"{case_id}_mirrored.nii.gz").exists() for case_id in case_ids])

    # Create very small lists for testing transforms, preprocessing etc. (NOT SHUFFLED)
    with path_case_ids_yaml_10.open("w") as f:
        d_case_ids_10 = {
            "train": list(case_ids[:6]),
            "val": list(case_ids[6:8]),
            "test": list(case_ids[8:10])
        }
        yaml.dump(d_case_ids_10, f)

    random.shuffle(case_ids)

    d_case_ids = {
        "train": list(case_ids[:int(len(case_ids)*0.7)]),
        "val": list(case_ids[int(len(case_ids)*0.7):int(len(case_ids)*0.85)]),
        "test": list(case_ids[int(len(case_ids)*0.85):])
    }

    with open(path_case_ids_yaml, "w") as f:
        yaml.dump(d_case_ids, f)



if __name__ == "__main__":
    main()
