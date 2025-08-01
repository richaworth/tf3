import yaml
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import matplotlib.pyplot as plt

from monai.data.utils import decollate_batch 
from monai.data.dataset import PersistentDataset, Dataset
from monai.data.dataloader import DataLoader
from monai.handlers.utils import from_engine
from monai.losses.dice import DiceLoss
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monai.transforms.post.array import AsDiscrete
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd, RandCropByPosNegLabeld
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.intensity.dictionary import RandShiftIntensityd, ScaleIntensityRanged
from monai.transforms.spatial.dictionary import Orientationd, RandAffined, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd

from monai.utils.misc import set_determinism
import numpy as np
import onnxruntime
from tqdm import tqdm

import torch

def val_one_batch(model, device, val_data, dice_metric, amp: bool = False):
    with torch.no_grad():
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=38)])
        post_label = Compose([AsDiscrete(to_onehot=38)])

        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )

        roi_size = (64, 64, 64)
        sw_batch_size = 1
        if amp:
            with torch.autocast("cuda"):
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs_as_discrete = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels_as_discrete = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs_as_discrete, y=val_labels_as_discrete)
        else:
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            val_outputs_as_discrete = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels_as_discrete = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs_as_discrete, y=val_labels_as_discrete)
    
    torch.cuda.empty_cache()
        


def main(path_output_dir: Path = Path("C:/data/tf3_output"), overwrite: bool = False):
    """
    Run ToothFairy3 model training

    Args:
        outdir (Path): Path to output directory (also contains any interim data constructed during training)
        overwrite (bool): overwrite existing output files?

    Returns:
        onnx file for each trained model
    """
    set_determinism(1)

    path_data_dir = Path("C:/data/tf3")

    # Get labels etc. from dataset yaml
    path_images = path_data_dir / "imagesTr"
    path_labels = path_data_dir / "labelsTr"
    path_bony_labels = path_data_dir / "labelsTr_bony"

    assert path_data_dir.exists(), f"Data directory {path_data_dir} is missing - can not continue."

    # Get case IDs, shuffle and distribute into train/test/validate - 70%/15%/15% split. 
    # TODO: Output these lists as text files for review.
    random.seed(0)
    case_ids = [label.name.split(".")[0] for label in path_labels.glob("*.nii.gz")]
    assert all([(path_images / f"{case_id}_0000.nii.gz").exists() for case_id in case_ids])
    random.shuffle(case_ids)

    case_ids_train = case_ids[:int(len(case_ids)*0.7)]
    case_ids_val = case_ids[int(len(case_ids)*0.7):int(len(case_ids)*0.85)]
    case_ids_test = case_ids[int(len(case_ids)*0.85):]

    # Image transforms (non-random first, so PersistentDataset can function)
    # Clipping/scaling HU - Air (-1000HU) to Metal (4000HU to differentiate from very hard tooth enamel (~3000HU)).
    # No mirroring, some small rotations, shear, scaling for variability.
    initial_transforms = [
        LoadImaged(keys=["image"]),
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.3, 0.3, 0.3), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=4000, b_min=0.0, b_max=1.0, clip=True, dtype=None),
        CropForegroundd(keys=["image", "label"], source_key="label", margin=10)
    ]

    train_only_transforms = [            
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.9, 
                    rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64),
                               pos=1, neg=1, num_samples=6, image_key="image", image_threshold=0.01),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ]
    
    train_transforms = Compose(initial_transforms + train_only_transforms)
    val_transforms = Compose(initial_transforms)

    # Prototype data - list dicts, dataset, loader.
  
    ld_train = [{"image": path_images / f"{c}_0000.nii.gz", "label": path_bony_labels / f"{c}.nii.gz"} for c in case_ids_train]
    ld_val = [{"image": path_images / f"{c}_0000.nii.gz", "label": path_bony_labels / f"{c}.nii.gz"} for c in case_ids_val]

    train_ds = PersistentDataset(data=ld_train, transform=train_transforms, cache_dir=path_output_dir / "cache_labels_bony_train")
    val_ds = Dataset(data=ld_val, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    # Use automatic mixed precision?
    amp = True

    device = torch.device("cuda:0")
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=38,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    
    loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    scaler = torch.GradScaler("cuda") if amp else None

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    max_epochs = 300
    val_interval = 1  # do validation for every epoch
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    epoch_times = []
    total_start = time.time()
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
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
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size},"
                f" train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            
            for val_data in tqdm(val_loader):
                val_one_batch(model, device, val_data, dice_metric, amp)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(model.state_dict(), "best_metric_model.pth")
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current"
                f" mean dice: {metric:.4f}"
                f" best mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

        print(f"time consuming of epoch {epoch + 1}" f" is: {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    print(
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

if __name__ == "__main__":
    main()
