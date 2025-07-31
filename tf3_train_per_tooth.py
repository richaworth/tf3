import logging
from pathlib import Path
import random
import shutil
import time

from monai.losses.dice import DiceLoss

from monai.data.utils import decollate_batch 
from monai.data.dataset import PersistentDataset, Dataset
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.networks.layers.factories import Norm
from monai.networks.nets.unet import UNet
from monai.transforms import (AsDiscrete, AsDiscreted, RandSpatialCropd, Compose, LoadImaged, SaveImaged, RandShiftIntensityd, ScaleIntensityRanged,
    Orientationd, RandAffined, EnsureChannelFirstd, RandFlipd, Spacingd, SpatialPadd)

from monai.utils.misc import set_determinism
import numpy as np
from tqdm import tqdm

import torch

ROI_SIZE = (72, 72, 72)

def train_model(ld_train: list[dict], 
                ld_val: list[dict], 
                path_output_dir: Path,
                model,
                model_name: str,
                list_initial_transforms: list,
                list_additional_training_transforms: list,
                n_labels: int, 
                amp: bool = True,
                path_checkpoint_model: Path | None = None,
                checkpoint_epoch: int | None = None,
                deterministic_training_seed: int | None = 1):

    set_determinism(deterministic_training_seed)

    train_transforms = Compose(list_initial_transforms + list_additional_training_transforms)
    val_transforms = Compose(list_initial_transforms)

    train_ds = PersistentDataset(data=ld_train, transform=train_transforms, cache_dir=path_output_dir / f"cache_train_{model_name}")
    val_ds = PersistentDataset(data=ld_val, transform=val_transforms, cache_dir=path_output_dir / f"cache_val_{model_name}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device("cuda:0")
    model.to(device)
    
    loss_function = DiceLoss(sigmoid=True, include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.GradScaler("cuda") if amp else None

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    max_epochs = 1000
    val_interval = 50
    checkpoint_interval = 10 # Save a checkpoint of the training in case restarting is required (temporary).
    major_checkpoint_interval = 100 # Save a permanent copy of the current training at major intervals.
    no_improvement_threshold = 5 # If no improvement in the metric within N validation cycles, stop training.
    count_no_improvement = 0
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    epoch_times = []
    total_start = time.time()

    # Load checkpoints
    if path_checkpoint_model is not None:
        torch.load(path_checkpoint_model, model.state_dict())
        
    min_epochs = 0 if checkpoint_epoch is None else checkpoint_epoch
    path_previous_best_metric = None
    path_previous_checkpoint = path_checkpoint_model

    path_final_model = path_output_dir / f"{model_name}.pth"

    for epoch in range(min_epochs, max_epochs):
        epoch_start = time.time()
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            if amp and scaler is not None:
                with torch.autocast("cuda"):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            logging.info(
                f"{step}/{len(train_ds) // train_loader.batch_size},"
                f" train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            path_new_checkpoint = path_output_dir / f"checkpoint_{model_name}_epoch_{epoch + 1}.pth"
            logging.debug(f"Saving current checkpoint model {path_new_checkpoint}.")
            torch.save(model.state_dict(), path_new_checkpoint)

            if (epoch - checkpoint_interval + 1) % major_checkpoint_interval == 0:
                logging.debug(f"Keeping major interval checkpoint model {path_previous_checkpoint}.")
            else:
                logging.debug(f"Removing previous checkpoint model {path_previous_checkpoint}.")
                path_previous_checkpoint.unlink()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            for val_data in tqdm(val_loader):
                    with torch.no_grad():
                        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=n_labels)])
                        post_label = Compose([AsDiscrete(to_onehot=n_labels)])

                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )

                        sw_batch_size = 4
                        if amp:
                            with torch.autocast("cuda"):
                                val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, sw_batch_size, model, overlap=0.75)
                                val_outputs_as_discrete = [post_pred(i) for i in decollate_batch(val_outputs)]
                                val_labels_as_discrete = [post_label(i) for i in decollate_batch(val_labels)]
                                dice_metric(y_pred=val_outputs_as_discrete, y=val_labels_as_discrete)
                        else:
                            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, sw_batch_size, model, overlap=0.75)

                            val_outputs_as_discrete = [post_pred(i) for i in decollate_batch(val_outputs)]
                            val_labels_as_discrete = [post_label(i) for i in decollate_batch(val_labels)]
                            dice_metric(y_pred=val_outputs_as_discrete, y=val_labels_as_discrete)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)

                path_best_metric = path_output_dir / f"{model_name}_best_metric_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), path_best_metric)
                logging.info(f"Saved new best metric model {path_best_metric}")

                if path_previous_best_metric is not None:
                    path_previous_best_metric.unlink()

                path_previous_best_metric = path_best_metric
                count_no_improvement = 0
            else:
                count_no_improvement = count_no_improvement + 1
            
            logging.info(
                f"current epoch: {epoch + 1} current"
                f" mean dice: {metric:.4f}"
                f" best mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

            if count_no_improvement == no_improvement_threshold:
                logging.info(f"No improvement after {count_no_improvement * val_interval} epochs. Stopping training.")
                shutil.move(path_previous_best_metric, path_final_model)
                break

        logging.info(f"time consuming of epoch {epoch + 1}" f" is: {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    logging.info(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
        f" total time: {(time.time() - total_start):.4f}"
    )

    return (
        max_epochs,
        epoch_loss_values,
        metric_values,
        epoch_times,
        best_metrics_epochs_and_time,
    )


def test_model(ld_test: list[dict], path_checkpoint_model: Path, model, preprocessing_transforms: list, postprocessing_transforms: list):
    test_ds = Dataset(data=ld_test, transform=Compose(preprocessing_transforms))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)
    post_pred = Compose(postprocessing_transforms)

    device = torch.device("cuda:0")
    model.to(device)

    torch.load(path_checkpoint_model, model.state_dict(), weights_only=False)

    model.eval()
    
    with torch.no_grad():
        for d in test_loader:
            images = d["image"].to(device)

            d["pred"] = sliding_window_inference(images, ROI_SIZE, 4, model)
            _ = [post_pred(i) for i in decollate_batch(d)]
            
            

def main(path_output_dir: Path = Path("C:/data/tf3_localiser_output/")):
    """
    Train low-resolution localiser model (upper jaw, upper teeth, lower teeth, lower jaw) from preprocessed images (see tf3_preprocess_images.py)

    Args:
        path_output_dir (_type_, optional): Path to output directory (will create cache directories, test searches as necessary).
            Defaults to Path("C:/data/tf3_localiser_output/").
    """
    (path_output_dir / "logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(message)s", 
                        handlers=[logging.FileHandler(path_output_dir / "logs/localiser_training_log.txt"), logging.StreamHandler()])
    deterministic_seed = 0
    random.seed(deterministic_seed)

    path_data_dir = Path("C:/data/tf3")

    # Get labels etc. from dataset json
    path_images = path_data_dir / "imagesTr"
    path_labels = path_data_dir / "labelsTr"

    assert path_data_dir.exists(), f"Data directory {path_data_dir} is missing - can not continue."

    # Get case IDs, shuffle and distribute into train/test/validate - 70%/15%/15% split. 
    # TODO: Output these lists as text files for review.
    case_ids = [label.name.split(".")[0] for label in path_labels.glob("*.nii.gz")]
    assert all([(path_images / f"{case_id}_0000.nii.gz").exists() for case_id in case_ids])
    random.shuffle(case_ids)

    case_ids_train = case_ids[:int(len(case_ids)*0.7)]
    case_ids_val = case_ids[int(len(case_ids)*0.7):int(len(case_ids)*0.85)]
    case_ids_test = case_ids[int(len(case_ids)*0.85):]

    # Labels - Background == 0, Tooth == 1, Pulp == 2.
    n_labels = 3   

    # Image intensity scaling - increased to -1000-4000 
    # SD intensity scaling - values are linear distance (in voxels) - clipping -50 to zero as input is only max 96x96x96.
    # In case tooth centre is too close to the edge of the image, Pad to ROI_SIZE. 
    initial_transforms = [
        LoadImaged(keys=["image", "label", "sd"]),
        EnsureChannelFirstd(keys=["image", "label", "sd"]),
        Orientationd(keys=["image", "label", "sd"], axcodes="RAS"),
        Spacingd(keys=["image", "label", "sd"], pixdim=(0.3, 0.3, 0.3), mode=("bilinear", "nearest", "bilinear")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["sd"], a_min=-50, a_max=0, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=["image", "label", "sd"], mode="constant", spatial_size=ROI_SIZE, constant_values=0)
    ]

    # RandFlipd over axis 2 (SI axis) only (RoLM cases already exist in the dataset, and flipping along the AP axis makes no sense when all images are aligned).
    # Rotation range is +/- 15* (again, because input images are aligned, the potential angle a tooth can be at is reduced)  - TODO (consider) increasing around SI axis.
    train_only_transforms = [            
        RandAffined(keys=["image", "label", "sd"], mode=('bilinear', 'nearest', 'bilinear'), prob=0.9, 
                    rotate_range=(np.pi/15, np.pi/15, np.pi/60), scale_range=(0.2, 0.2, 0.2)),
        RandFlipd(keys=["image", "label", "sd"], prob=0.5, spatial_axis=2),
        RandSpatialCropd(keys=["image", "label", "sd"], roi_size=ROI_SIZE),
        RandShiftIntensityd(keys=["image", "sd"], factors=0.10, prob=0.90),
        RandShiftIntensityd(keys=["image", "sd"], offsets=0.10, prob=0.90),
    ]

    # TODO: Preprocessing transforms to get "image" and "sd" like patches from 

    postprocessing_transforms = [
        AsDiscreted(keys="pred", argmax=True, to_onehot=n_labels),
        SaveImaged(keys="pred", resample=True)
    ]
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=n_labels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )

    path_per_tooth_images = path_data_dir / "images_per_tooth_cropped"
    path_per_tooth_labels = path_data_dir / "labels_per_tooth_cropped"
    path_per_tooth_sd = path_data_dir / "labels_tooth_centre_sd"

    ld_train = []
    ld_val = []
    ld_test = []

    # Construct train/test/val split, per-tooth, lists of dicts
    for case_ids, ld in [(case_ids_train, ld_train), (case_ids_val, ld_val), (case_ids_test, ld_test)]:
        for c in case_ids:
            l_per_tooth_ids = [x.name.split(".nii.gz")[0] for x in path_per_tooth_images.glob(f"{c}_*.nii.gz")]
            for id in l_per_tooth_ids:
                d = {"image": path_per_tooth_images / f"{id}.nii.gz", 
                    "label": path_per_tooth_labels / f"{id}.nii.gz", 
                    "sd": path_per_tooth_sd / f"{id}_cropped.nii.gz"}
                dm = {"image": path_per_tooth_images / f"{id}_mirrored.nii.gz", 
                    "label": path_per_tooth_labels / f"{id}_mirrored.nii.gz", 
                    "sd": path_per_tooth_sd / f"{id}_mirrored_cropped.nii.gz"}
                ld.append(d)
                ld.append(dm)

    train_model(ld_train, ld_val, path_output_dir, model, "per_tooth_unet_diceloss", initial_transforms, train_only_transforms, n_labels, amp=False, 
                deterministic_training_seed=deterministic_seed)
    
    test_model(ld_test, path_output_dir / "per_tooth_unet_diceloss.pth", model, initial_transforms, postprocessing_transforms)
    


if __name__ == "__main__":
    main()
