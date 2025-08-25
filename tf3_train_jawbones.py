import logging
from pathlib import Path
import random
import shutil
import yaml

from monai.losses.dice import DiceCELoss

from monai.data.utils import decollate_batch 
from monai.data.dataset import PersistentDataset, Dataset
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.networks.nets.unetr import UNETR
from monai.config.type_definitions import KeysCollection

from monai.transforms import (
    AsDiscrete, 
    Compose, 
    CropForegroundd,
    EnsureChannelFirstd, 
    EnsureTyped,
    LabelFilterd, 
    LoadImaged, 
    MapTransform,
    Orientationd, 
    RandAffined, 
    RandCropByPosNegLabeld, 
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandShiftIntensityd, 
    Spacing,
    Spacingd, 
    ScaleIntensityRanged,
    SaveImage, 
    SpatialPadd
)

from monai.utils.misc import set_determinism
import numpy as np
from tqdm import tqdm

import torch

# ROI_SIZE = (96, 96, 96)  # Version from 16/08/25; 11:25
ROI_SIZE = (80, 80, 80)
AMP = True
DEVICE = "cuda:0"

class EditLabelsd(MapTransform):
    """
    Change the labels in an image using a lookup list of tuples.
    """
    def __init__(self, keys: KeysCollection, list_old_new_labels: list[tuple]) -> None:
        super().__init__(keys)
        self.list_on_labels = list_old_new_labels

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = torch.zeros_like(d[key])

            for old, new in self.list_on_labels:
                result = torch.where(d[key] == old, new, result)
            
            d[key] = result
            
        return d
            

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
                batch_size: int = 4,
                max_epochs: int = 400,
                ):

    set_determinism(deterministic_training_seed)

    train_transforms = Compose(list_train_trans)
    val_transforms = Compose(lst_val_trans)
   
    train_ds = PersistentDataset(data=ld_train, transform=train_transforms, cache_dir=path_output_dir / f"cache_train_{model_name}")
    val_ds = PersistentDataset(data=ld_val, transform=val_transforms, cache_dir=path_output_dir / f"cache_val_{model_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=n_workers)

    device = torch.device(DEVICE)
    model.to(device)

    val_interval = 1
    checkpoint_interval = 2 # Save a checkpoint of the training in case restarting is required (temporary).
    major_checkpoint_interval = 100 # Save a permanent copy of the current training at major intervals.
    no_improvement_threshold = 5 # If no improvement in the metric within N validation cycles, stop training.
        
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Load checkpoints
    if path_checkpoint_model is not None:
        path_model_ckpt_previous = path_checkpoint_model
        path_opt_ckpt_previous = path_checkpoint_model.parent / f"{path_checkpoint_model.stem}_opt{path_checkpoint_model.suffix}"
        model.load_state_dict(torch.load(path_checkpoint_model, weights_only=True))

        torch.load(path_opt_ckpt_previous, optimizer.state_dict())
        optimizer.param_groups[0]["initial_lr"] = 1e-4
        min_epochs = checkpoint_epoch
    else:
        path_model_ckpt_previous = None
        path_opt_ckpt_previous = None
        min_epochs = 0
        
    lr_epoch = min_epochs - 1
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, last_epoch=lr_epoch)

    scaler = torch.GradScaler("cuda") if AMP else None

    post_label = AsDiscrete(to_onehot=n_labels)
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
    
    for epoch in range(min_epochs, max_epochs):
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{max_epochs}")

        model.train()
        epoch_loss = 0
        step = 0

        # for batch_data in tqdm(train_loader):
        #     step += 1
        #     inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
        #     optimizer.zero_grad()

        #     if AMP:
        #         with torch.autocast("cuda"):
        #             outputs = model(inputs)
        #             loss = loss_function(outputs, labels)
        #         scaler.scale(loss).backward()
        #         scaler.step(optimizer)
        #         scaler.update()
        #     else:
        #         outputs = model(inputs)
        #         loss = loss_function(outputs, labels)
        #         loss.backward()
        #         optimizer.step()

        #     epoch_loss += loss.item()
        #     logging.debug(f"{step}/{len(train_ds) // train_loader.batch_size} train_loss: {loss.item():.4f}")

        # lr_scheduler.step()
        # epoch_loss /= step
        # epoch_loss_values.append(epoch_loss)
        # logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # # Save checkpoint (deleting non-major checkpoints)
        # if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
        #     path_model_ckpt_new = path_output_dir / f"checkpoint_{model_name}_epoch_{epoch + 1}.pkl"
        #     path_opt_ckpt_new = path_model_ckpt_new.parent / f"{path_model_ckpt_new.stem}_opt{path_model_ckpt_new.suffix}"
        #     logging.info(f"Saving current checkpoint model: {path_model_ckpt_new}")
        #     logging.debug(f"Saving current checkpoint optimiser: {path_opt_ckpt_new}")

        #     torch.save(model.state_dict(), path_model_ckpt_new)
        #     torch.save(optimizer.state_dict(), path_opt_ckpt_new)

        #     if (epoch - checkpoint_interval + 1) % major_checkpoint_interval == 0:
        #         logging.debug(f"Keeping major interval checkpoint model {path_model_ckpt_previous}.")
        #     elif path_model_ckpt_previous is not None:
        #         logging.debug(f"Removing previous checkpoint model {path_model_ckpt_previous}.")
        #         logging.debug(f"Removing previous checkpoint optimiser {path_opt_ckpt_previous}.")
        #         path_model_ckpt_previous.unlink()
        #         path_opt_ckpt_previous.unlink()
            
        #     path_model_ckpt_previous = path_model_ckpt_new
        #     path_opt_ckpt_previous = path_opt_ckpt_new

        if (epoch + 1) % val_interval == 0:
            model.eval()

            with torch.no_grad():
                for val_data in tqdm(val_loader):
                    val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))
                    
                    if AMP:
                        with torch.autocast("cuda"):
                            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 2, model, 0.5)
                    else:
                        val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 2, model, 0.5)

                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                dsc = dice_metric.aggregate().item()
                metric_values.append((dsc))
                print(dsc)

                # if dsc > best_dsc:
                #     best_dsc = dsc
                #     best_dsc_epoch = epoch + 1
                #     best_metrics_epochs[0].append(best_dsc)
                #     best_metrics_epochs[1].append(best_dsc_epoch)

                #     path_best_metric = path_output_dir / f"{model_name}_best_metric_epoch_{epoch + 1}.pkl"
                #     path_best_metric_opt = path_best_metric.parent / f"{path_best_metric.stem}_opt{path_best_metric.suffix}"
                #     torch.save(model.state_dict(), path_best_metric)
                #     torch.save(optimizer.state_dict(), path_best_metric_opt)
                    
                #     logging.info(f"Saved new best metric model {path_best_metric}")

                #     # Manage previous best metric models
                #     if path_previous_best_metric is not None:
                #         path_previous_best_metric.unlink()
                #         path_previous_best_metric_opt.unlink()

                #     path_previous_best_metric = path_best_metric
                #     path_previous_best_metric_opt = path_best_metric_opt
                #     count_no_improvement = 0
                # else:
                #     count_no_improvement = count_no_improvement + 1

                # logging.info(
                #     f"current epoch: {epoch + 1} current"
                #     f" mean dice: {dsc:.4f}"
                #     f" best mean dice: {best_dsc:.4f} "
                #     f"at epoch: {best_dsc_epoch}"
                # )

                # if count_no_improvement == no_improvement_threshold:
                #     logging.info(f"No improvement after {count_no_improvement * val_interval} epochs. Stopping training.")
                #     shutil.move(path_previous_best_metric, path_final_model)
                #     break

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
    torch.load(path_checkpoint_model, model.state_dict(), weights_only=False)

    model.to(device)
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


def main(path_output_dir: Path = Path("C:/data/tf3_jawbones_only/")):
    """
    Train jaw bone and canal anatomy model from preprocessed images (see tf3_preprocess_images.py)

    Args:
        path_output_dir (_type_, optional): Path to output directory (will create cache directories, test searches as necessary).
            Defaults to Path("C:/data/tf3_jawbones_output/").
    """
    (path_output_dir / "logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(message)s", 
                        handlers=[logging.FileHandler(path_output_dir / "logs/gross_anatomy.txt"), logging.StreamHandler()])
    deterministic_seed = 0

    path_data_dir = Path("C:/data/tf3")

    path_images = path_data_dir / "images_rolm"
    path_labels = path_data_dir / "labels_rolm"

    assert path_data_dir.exists(), f"Data directory {path_data_dir} is missing - can not continue."
    
    # Create or load case IDs and train/test/val split
    path_case_ids_yaml = path_data_dir / "case_id_lists.yaml"

    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    # Image transforms (non-random first, so PersistentDataset can function)
    # Clipping/scaling HU - No real reason for a_max to be above 2000 HU for localisation model (metalwork is included in the teeth/jaw labels).

    # Excluding bridges/crowns, implants, canals, all other anatomy is in this model (due to time restrictions from server crashes and illness).
    # Jaw bones are 1, 2. 
    # Left Maxillary Sinus == 5, Right Maxillary Sinus == 6, Pharynx == 7
    # NB: Canals are 3, 4, 103, 104, 105; bridges and crowns are 8 and 9.

    filter_labels = [1, 2, 5, 6, 7]
    rename_labels = [(x, i + 1) for i, x in enumerate(filter_labels)]
    n_labels = len(rename_labels) + 1

    initial_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        LabelFilterd("label", filter_labels), 
        EditLabelsd("label", rename_labels),
        ScaleIntensityRanged("image", a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode="nearest"),
        SpatialPadd(["image", "label"], ROI_SIZE),
    ]

    train_only_transforms = [
        RandAffined(keys=["image", "label"], mode=("bilinear", "nearest"), prob=0.9, 
                    rotate_range=(np.pi/15, np.pi/15, np.pi/15), scale_range=(0.1, 0.1, 0.1)),            
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=ROI_SIZE,
                               pos=1, neg=1, num_samples=4, allow_smaller=False),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.90),
        RandGaussianNoised(keys=["image"], prob=0.2),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandGaussianSharpend(keys=["image"], prob=0.2),
    ]

    test_val_only_transforms = []
    train_transforms = initial_transforms + train_only_transforms
    test_val_transforms = initial_transforms + test_val_only_transforms

    # TODO: Convert to dict transforms; convert onehot to real labels again (potentially outside this transform - can use fuction from tf3_preprocess_images.py).
    postprocessing_transforms = [
        AsDiscrete(argmax=True),
        Spacing(pixdim=(0.3, 0.3, 0.3), mode="nearest"),
        SaveImage(output_dir=path_output_dir / "test_segmentations")
    ]
    
    model = UNETR(
        in_channels=1,
        out_channels=n_labels,
        img_size=ROI_SIZE,
        feature_size=32,
        # feature_size=16,   # Version from 16/08/25; 11:25
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        # proj_type="conv",   # Version from 16/08/25; 11:25
        norm_name="instance",
        # norm_name="batch",   # Version from 16/08/25; 11:25
        res_block=True,
        dropout_rate=0.0,
    )

    ld_train = []
    ld_val = []
    ld_test = []

    # Construct train/test/val split, lists of dicts
    for case_ids, ld in [(d_case_ids["train"], ld_train), (d_case_ids["val"], ld_val), (d_case_ids["test"], ld_test)]:
        for c in case_ids:
            d = {"image": path_images / f"{c}.nii.gz", 
                "label": path_labels / f"{c}.nii.gz"}
            dm = {"image": path_images / f"{c}_mirrored.nii.gz", 
                  "label": path_labels / f"{c}_mirrored.nii.gz"}
            ld.append(d)
            ld.append(dm)

    n_workers = 4
    batch_size = 6

    train_model(ld_train[:10], ld_val[:10], path_output_dir, model, "all_non_tooth_anatomy", train_transforms, test_val_transforms, n_labels,
                deterministic_training_seed=1, n_workers=4, batch_size=5, checkpoint_epoch=402, 
                path_checkpoint_model=path_output_dir / "checkpoint_all_non_tooth_anatomy_test_epoch_402.pkl", max_epochs=403)
    
    # test_model(ld_test[:2], path_output_dir / "all_non_tooth_anatomy_best_metric_epoch_400.pkl", model, test_val_transforms, postprocessing_transforms, 
    #            n_labels, n_workers=1, batch_size=1)

    
if __name__ == "__main__":
    main()
