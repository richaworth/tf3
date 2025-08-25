import logging
from pathlib import Path
import shutil
import yaml

from monai.losses.dice import DiceCELoss

from monai.data.utils import decollate_batch 
from monai.data.dataset import PersistentDataset, Dataset
from monai.data.dataloader import DataLoader
from monai.metrics.meandice import DiceMetric
from monai.networks.nets.unet import UNet
from monai.inferers.utils import sliding_window_inference

from monai.transforms import (
    AsDiscrete, 
    ConcatItemsd,
    Compose, 
    DeleteItemsd,
    EnsureChannelFirstd, 
    EnsureTyped,
    LoadImaged, 
    Orientationd, 
    SpatialPadd,
    RandAffined, 
    RandCropByPosNegLabeld, 
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandShiftIntensityd, 
    ResampleToMatchd, 
    ScaleIntensityRanged,
    SaveImage, 
)

from monai.utils.misc import set_determinism
import numpy as np
from tqdm import tqdm

import torch

ROI_SIZE = (64, 64, 64)  # 0.3mm^3 voxels ~ 25mm high; 15mm wide, 15mm deep (https://www.cdhp.org/what-is-the-average-size-of-each-tooth/)
AMP = False
DEVICE = "cuda:0"
           
def train_model(ld_train: list[dict], 
                ld_val: list[dict], 
                path_output_dir: Path,
                model,
                model_name: str,
                list_train_trans: list,
                lst_val_trans: list,
                n_labels: int, 
                path_checkpoint_model: Path | None = None,
                checkpoint_epoch: int | None = None,
                deterministic_training_seed: int | None = 1,
                n_workers: int = 4, 
                batch_size: int = 4
                ):

    set_determinism(deterministic_training_seed)

    train_transforms = Compose(list_train_trans)
    val_transforms = Compose(lst_val_trans)
   
    train_ds = PersistentDataset(data=ld_train, transform=train_transforms, cache_dir=path_output_dir / f"cache_train_{model_name}")
    val_ds = PersistentDataset(data=ld_val, transform=val_transforms, cache_dir=path_output_dir / f"cache_val_{model_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size/2), shuffle=False, num_workers=n_workers)

    device = torch.device(DEVICE)
    model.to(device)

    max_epochs = 20
    val_interval = 2
    checkpoint_interval = 10 # Save a checkpoint of the training in case restarting is required (temporary).
    major_checkpoint_interval = 10 # Save a permanent copy of the current training at major intervals.
    no_improvement_threshold = 5 # If no improvement in the metric within N validation cycles, stop training.
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Load checkpoints
    if path_checkpoint_model is not None:
        path_model_ckpt_previous = path_checkpoint_model
        path_opt_ckpt_previous = path_checkpoint_model.parent / f"{path_checkpoint_model.stem}_opt{path_checkpoint_model.suffix}"
        torch.load(path_checkpoint_model, model.state_dict())
        torch.load(path_opt_ckpt_previous, optimizer.state_dict())
        optimizer.param_groups[0]["initial_lr"] = 1e-4
    else:
        path_model_ckpt_previous = None
        path_opt_ckpt_previous = None

    lr_epoch = -1 if checkpoint_epoch is None else checkpoint_epoch -1
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-7, last_epoch=lr_epoch)

    scaler = torch.GradScaler("cuda") if AMP else None

    post_label_trans = Compose([AsDiscrete(to_onehot=n_labels)])
    post_trans = Compose([AsDiscrete(argmax=True, to_onehot=n_labels)])
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=n_labels)
    
    best_dsc = -1
    best_dsc_epoch = -1
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
            step+=1
            inputs, labels = (batch_data["inputs"].to(device), batch_data["label"].to(device))
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
            path_opt_ckpt_new = path_model_ckpt_new.parent / f"{path_model_ckpt_new.stem}_opt{path_model_ckpt_new.suffix}"
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
                    val_inputs, val_labels = (val_data["inputs"].to(device), val_data["label"].to(device))
                    
                    if AMP:
                        with torch.autocast("cuda"):
                            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 2, model, 0.5)
                    else:
                        val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 2, model, 0.5)

                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label_trans(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                dsc = dice_metric.aggregate().item()
                metric_values.append((dsc))

                if dsc > best_dsc:
                    best_dsc = dsc
                    best_dsc_epoch = epoch + 1
                    best_metrics_epochs[0].append(best_dsc)
                    best_metrics_epochs[1].append(best_dsc_epoch)

                    path_best_metric = path_output_dir / f"{model_name}_best_metric_epoch_{epoch + 1}.pkl"
                    path_best_metric_opt = path_best_metric.parent / f"{path_best_metric.stem}_opt{path_best_metric.suffix}"
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
                    f" mean dice: {dsc:.4f}"
                    f" best mean dice: {best_dsc:.4f} "
                    f"at epoch: {best_dsc_epoch}"
                )

                if count_no_improvement == no_improvement_threshold:
                    logging.info(f"No improvement after {count_no_improvement * val_interval} epochs. Stopping training.")
                    shutil.move(path_previous_best_metric, path_final_model)
                    break

    logging.info(f"Training completed, best_metric: {best_dsc:.4f} at epoch: {best_dsc_epoch}")
    return (max_epochs, epoch_loss_values, metric_values, best_metrics_epochs)


def test_model(ld_test: list[dict], 
               path_checkpoint_model: Path, 
               model, 
               preprocessing_transforms: list, 
               postprocessing_transforms: list, 
               n_labels: int,
               n_workers: int = 4, 
               batch_size: int = 4):
    # Calculate DICE if possible
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=n_labels)

    test_ds = Dataset(data=ld_test, transform=Compose(preprocessing_transforms))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    post_trans = Compose(postprocessing_transforms)

    device = torch.device(DEVICE)
    model.to(device)

    torch.load(path_checkpoint_model, model.state_dict(), weights_only=False)

    model.eval()
    
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_inputs, test_labels = (test_data["inputs"].to(device), test_data["label"].to(device))
                    
            if AMP:
                with torch.autocast("cuda"):
                    test_outputs = sliding_window_inference(test_inputs, ROI_SIZE, 4, model, 0.5)
            else:
                test_outputs = sliding_window_inference(test_inputs, ROI_SIZE, 4, model, 0.5)

            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            
            dice_metric(y_pred=test_outputs, y=test_labels)
            print(dice_metric.aggregate())


def main(path_output_dir: Path = Path("C:/data/tf3_per_tooth_output_last_minute_panic/")):
    """
    Train jaw bone and canal anatomy model from preprocessed images (see tf3_preprocess_images.py)

    Args:
        path_output_dir (_type_, optional): Path to output directory (will create cache directories, test searches as necessary).
            Defaults to Path("C:/data/tf3_per_tooth_output/").
    """
    (path_output_dir / "logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(message)s", 
                        handlers=[logging.FileHandler(path_output_dir / "logs/localiser_training_log.txt"), logging.StreamHandler()])
    deterministic_seed = 0

    path_data_dir = Path("C:/data/tf3")

    path_images = path_data_dir / "images_per_tooth_cropped"
    path_labels = path_data_dir / "labels_per_tooth_cropped"

    assert path_data_dir.exists(), f"Data directory {path_data_dir} is missing - can not continue."
    
    # Create or load case IDs and train/test/val split
    path_case_ids_yaml = path_data_dir / "case_id_lists.yaml"

    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    n_labels = 4  # 0: background, 1: tooth, 2: pulp, 3: non-target tooth.

    initial_transforms = [
        LoadImaged(keys=["image", "localiser", "label"]),
        EnsureChannelFirstd(keys=["image", "localiser", "label"]),
        Orientationd(keys=["image", "localiser", "label"], axcodes="RAS"),
        ResampleToMatchd(["localiser", "label"], "image", mode=("bilinear", "nearest"), padding_mode="zeros"),
        ScaleIntensityRanged("image", a_min=-1000, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged("localiser", a_min=-20, a_max=0, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=["image", "localiser", "label"], spatial_size=ROI_SIZE),
    ]

    train_only_transforms = [
        RandAffined(keys=["image", "localiser", "label"], mode=("bilinear", "nearest", "nearest"), prob=0.8, 
                    rotate_range=(np.pi/15, np.pi/15, np.pi/60), scale_range=(0.2, 0.2, 0.2), padding_mode="border",
                    translate_range=(20, 20, 20)),
        RandAffined(keys=["localiser"], mode=("nearest"), prob=0.8,  scale_range=(0.1, 0.1, 0.1), padding_mode="zeros", translate_range=(10, 10, 30)),
        RandCropByPosNegLabeld(keys=["image", "label", "localiser"], label_key="label", spatial_size=ROI_SIZE,
                        pos=1, neg=1, num_samples=4, allow_smaller=False),           
        RandFlipd(keys=["image", "localiser", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.8),
        RandGaussianNoised(keys=["image"], prob=0.2),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandGaussianSharpend(keys=["image"], prob=0.2),
        ConcatItemsd(keys=["image", "localiser"], name="inputs"),
        DeleteItemsd(keys=["image", "localiser"]),
        EnsureTyped(keys="inputs", dtype=np.float32)
    ]

    test_val_only_transforms = [
        ConcatItemsd(keys=["image", "localiser"], name="inputs"),
        DeleteItemsd(keys=["image", "localiser"]),
        EnsureTyped(keys="inputs", dtype=np.float32)
    ]

    train_transforms = initial_transforms + train_only_transforms
    test_val_transforms = initial_transforms + test_val_only_transforms

    postprocessing_transforms = [
        AsDiscrete(argmax=True),
        SaveImage(output_dir=path_output_dir / "test_segmentations", separate_folder=False)
    ]

    model = UNet(
        spatial_dims=len(ROI_SIZE),
        in_channels=2,
        out_channels=n_labels,
        channels=(4, 8, 16, 32, 64),
        strides=(2, 2, 2, 2),
    )

    path_tooth_centre_sd = path_data_dir / "labels_tooth_centre_sd"

    ld_train = []
    ld_val = []
    ld_test = []

    # Construct train/test/val split, per-tooth, lists of dicts
    for case_ids, ld in [(d_case_ids["train"], ld_train), (d_case_ids["val"], ld_val), (d_case_ids["test"], ld_test)]:
        for c in case_ids:
            for path_image in path_images.glob(f"{c}*.nii.gz"):
                stem = path_image.name.split(".nii.gz")[0]

                d = {"image": path_images / f"{stem}.nii.gz", 
                    "label": path_labels / f"{stem}.nii.gz", 
                    "localiser": path_tooth_centre_sd / f"{stem}_cropped.nii.gz"}

                ld.append(d)
    
    model_name = "per_tooth_unet_64_64_64"

    train_model(ld_train, ld_val, path_output_dir, model, model_name, train_transforms, test_val_transforms, n_labels,
                deterministic_training_seed=deterministic_seed, n_workers=4, batch_size=10)
    
    test_model(ld_test, path_output_dir / f"{model_name}_epoch_20.pkl", model, 
               test_val_transforms, postprocessing_transforms, n_labels, n_workers=1, batch_size=1)


if __name__ == "__main__":
    main()
