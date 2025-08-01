import yaml
import logging
from pathlib import Path
import random
import shutil

from monai.losses.dice import DiceLoss

from monai.data.utils import decollate_batch 
from monai.data.dataset import PersistentDataset, Dataset
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.networks.layers.factories import Norm
from monai.networks.nets.unet import UNet
from monai.transforms import (AsDiscrete, RandCropByPosNegLabeld, Compose, LoadImaged, SaveImage, RandShiftIntensityd, ScaleIntensityRanged,
    Orientationd, RandAffined, EnsureChannelFirstd, SpatialPadd, Spacingd, ResampleToMatchd)

from monai.utils.misc import set_determinism
import numpy as np
from tqdm import tqdm

import torch

ROI_SIZE = (96, 96, 96)
AMP = True

def train_model(ld_train: list[dict], 
                ld_val: list[dict], 
                path_output_dir: Path,
                model,
                model_name: str,
                list_initial_transforms: list,
                list_additional_training_transforms: list,
                n_labels: int, 
                path_checkpoint_model: Path | None = None,
                checkpoint_epoch: int | None = None,
                deterministic_training_seed: int | None = 1):

    set_determinism(deterministic_training_seed)

    train_transforms = Compose(list_initial_transforms + list_additional_training_transforms)
    val_transforms = Compose(list_initial_transforms)

    train_ds = PersistentDataset(data=ld_train, transform=train_transforms, cache_dir=path_output_dir / f"cache_train_{model_name}")
    val_ds = PersistentDataset(data=ld_val, transform=val_transforms, cache_dir=path_output_dir / f"cache_val_{model_name}")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda:0")
    model.to(device)

    max_epochs = 1000
    val_interval = 10
    checkpoint_interval = 10 # Save a checkpoint of the training in case restarting is required (temporary).
    major_checkpoint_interval = 100 # Save a permanent copy of the current training at major intervals.
    no_improvement_threshold = 5 # If no improvement in the metric within N validation cycles, stop training.
    
    loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Load checkpoints
    if path_checkpoint_model is not None:
        path_model_ckpt_previous = path_checkpoint_model
        path_opt_ckpt_previous = path_checkpoint_model.parent / f"{path_checkpoint_model.stem}_opt.{path_checkpoint_model.suffix}"
        torch.load(path_checkpoint_model, model.state_dict())
        torch.load(path_opt_ckpt_previous)
    else:
        path_model_ckpt_previous = None
        path_opt_ckpt_previous = None

    lr_epoch = -1 if checkpoint_epoch is None else checkpoint_epoch -1
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, last_epoch=lr_epoch)

    scaler = torch.GradScaler("cuda") if AMP else None
    torch.backends.cudnn.benchmark = True

    post_trans = Compose([AsDiscrete(argmax=True, to_onehot=n_labels)])
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=n_labels)
    
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs = [[], []]
    epoch_loss_values = []
    metric_values = []

    path_previous_best_metric = None
    path_previous_best_metric_opt = None

    path_final_model = path_output_dir / f"{model_name}.pkl"


    min_epochs = 0 if checkpoint_epoch is None else checkpoint_epoch
    
    for epoch in range(min_epochs, max_epochs):
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{max_epochs}")

        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()

            if AMP:
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
            logging.debug(f"{step}/{len(train_ds) // train_loader.batch_size} train_loss: {loss.item():.4f}")

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Save checkpoint (deleting non-major checkpoints)
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            path_model_ckpt_new = path_output_dir / f"checkpoint_{model_name}_epoch_{epoch + 1}.pkl"
            path_opt_ckpt_new = path_model_ckpt_new.parent / f"{path_model_ckpt_new.stem}_opt.{path_model_ckpt_new.suffix}"
            logging.info(f"Saving current checkpoint model: {path_model_ckpt_new}")
            logging.debug(f"Saving current checkpoint optimiser: {path_opt_ckpt_new}")

            torch.save(model.state_dict(), path_model_ckpt_new)
            torch.save(optimizer.state_dict(), path_opt_ckpt_new)

            if (epoch - checkpoint_interval + 1) % major_checkpoint_interval == 0:
                logging.debug(f"Keeping major interval checkpoint model {path_model_ckpt_previous}.")
            elif path_model_ckpt_previous is not None:
                logging.debug(f"Removing previous checkpoint model {path_model_ckpt_previous}.")
                logging.debug(f"Removing previous checkpoint optimiser {path_opt_ckpt_previous}.")
                path_model_ckpt_previous.unlink()
                path_opt_ckpt_previous.unlink()
            
            path_model_ckpt_previous = path_model_ckpt_new
            path_opt_ckpt_previous = path_opt_ckpt_new

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader):
                    val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))
                    
                    if AMP:
                        with torch.autocast("cuda"):
                            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 4, model, 0.5)
                    else:
                        val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 4, model, 0.5)

                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs[0].append(best_metric)
                    best_metrics_epochs[1].append(best_metric_epoch)

                    path_best_metric = path_output_dir / f"{model_name}_best_metric_epoch_{epoch + 1}.pkl"
                    path_best_metric_opt = path_best_metric.parent / f"{path_best_metric.stem}_opt.{path_best_metric.suffix}"
                    torch.save(model.state_dict(), path_best_metric)
                    torch.save(optimizer.state_dict(), path_best_metric_opt)
                    
                    logging.info(f"Saved new best metric model {path_best_metric}")

                    # Manage previous best metric models
                    if path_previous_best_metric is not None:
                        path_previous_best_metric.unlink()
                        path_previous_best_metric_opt.unlink()

                    path_previous_best_metric = path_best_metric
                    path_previous_best_metric_opt = path_best_metric_opt
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

    logging.info(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    return (max_epochs, epoch_loss_values, metric_values, best_metrics_epochs)


def test_model(ld_test: list[dict], 
               path_checkpoint_model: Path, 
               model, 
               preprocessing_transforms: list, 
               postprocessing_transforms: list, 
               n_labels: int,
               overwrite: bool = False):
    # Calculate DICE if possible
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=n_labels)

    test_ds = Dataset(data=ld_test, transform=Compose(preprocessing_transforms))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)

    post_trans = Compose(postprocessing_transforms)

    device = torch.device("cuda:0")
    model.to(device)

    torch.load(path_checkpoint_model, model.state_dict(), weights_only=False)

    model.eval()
    
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_inputs, test_labels = (test_data["image"].to(device), test_data["label"].to(device))
                    
            if AMP:
                with torch.autocast("cuda"):
                    test_outputs = sliding_window_inference(test_inputs, ROI_SIZE, 4, model, 0.5)
            else:
                test_outputs = sliding_window_inference(test_inputs, ROI_SIZE, 4, model, 0.5)

            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            
            dice_metric(y_pred=test_outputs, y=test_labels)
            print(dice_metric.aggregate())  # TODO - get case id/image name and print with this. Save to CSV.         


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

    assert path_data_dir.exists(), f"Data directory {path_data_dir} is missing - can not continue."

    # Create or load case IDs and train/test/val split
    path_case_ids_yaml = path_data_dir / "case_id_lists.yaml"
    with path_case_ids_yaml.open("r") as f:
        d_case_ids = yaml.load(f)

    n_labels = 5
    
    # Image transforms (non-random first, so PersistentDataset can function)
    # Clipping/scaling HU - No real reason for a_max to be above 2000 HU for localisation model (metalwork is included in the teeth/jaw labels).
    # Initial images are already set to 1, 1, 1 voxel size, so no rescaling required.

    initial_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ResampleToMatchd("label", "image", mode="nearest", padding_mode="zeros"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, mode="constant", constant_values=0),
        # SaveImaged(keys=["image"], output_postfix="initial_image"),
        # SaveImaged(keys=["label"], output_postfix="initial_label")
    ]

    train_only_transforms = [            
        RandAffined(keys=["image", "label"], mode=("bilinear", "nearest"), prob=0.9, 
                    rotate_range=(np.pi/15, np.pi/15, np.pi/15), scale_range=(0.1, 0.1, 0.1)),            
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=ROI_SIZE,
                               pos=1, neg=1, num_samples=4, allow_smaller=False),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.90),
        # SaveImaged(keys=["image"], output_postfix="rand_trans_image"),
        # SaveImaged(keys=["label"], output_postfix="rand_trans_label")
    ]

    # TODO: Test on original res images (downsample/upsample in Compose)
    postprocessing_transforms = [
        AsDiscrete(argmax=True),
        SaveImage(output_dir=path_output_dir / "test_segmentations")
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

    path_localiser_images = path_data_dir / "images_localiser_rolm"
    path_localiser_labels = path_data_dir / "labels_localiser_rolm"

    ld_train = []
    ld_val = []
    ld_test = []

    for c in d_case_ids["train"]:
        ld_train.append({"image": path_localiser_images / f"{c}.nii.gz", "label": path_localiser_labels / f"{c}.nii.gz"})
        ld_train.append({"image": path_localiser_images / f"{c}_mirrored.nii.gz", "label": path_localiser_labels / f"{c}_mirrored.nii.gz"})

    for c in d_case_ids["val"]:
        ld_val.append({"image": path_localiser_images / f"{c}.nii.gz", "label": path_localiser_labels / f"{c}.nii.gz"})
        ld_val.append({"image": path_localiser_images / f"{c}_mirrored.nii.gz", "label": path_localiser_labels / f"{c}_mirrored.nii.gz"})

    for c in d_case_ids["test"]:
        ld_test.append({"image": path_data_dir / "images_rolm" / f"{c}.nii.gz", "label": path_localiser_labels / f"{c}.nii.gz"})
        ld_test.append({"image": path_localiser_images / f"{c}_mirrored.nii.gz", "label": path_localiser_labels / f"{c}_mirrored.nii.gz"})

    train_model(ld_train, ld_val, path_output_dir, model, "localiser_unet_diceloss", initial_transforms, train_only_transforms, n_labels,
                deterministic_training_seed=deterministic_seed)
    
    test_model(ld_test, path_output_dir / "localiser_unet_diceloss_best_metric.pkl", model, initial_transforms, postprocessing_transforms, 
               n_labels)

    
if __name__ == "__main__":
    main()
