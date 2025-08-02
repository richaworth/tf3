import logging
from pathlib import Path
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
    AsDiscreted, 
    ConcatItemsd,
    Compose, 
    CropForegroundd,
    DeleteItemsd,
    DistanceTransformEDTd,
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
    RemoveSmallObjectsd, 
    ResampleToMatchd, 
    ScaleIntensityRanged,
    SaveImage, 
    SaveImaged
)

from monai.utils.misc import set_determinism
import numpy as np
from tqdm import tqdm

import torch

ROI_SIZE = (96, 96, 96)
AMP = False

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
                batch_size: int = 4
                ):

    set_determinism(deterministic_training_seed)

    train_transforms = Compose(list_train_trans)
    val_transforms = Compose(lst_val_trans)
   
    train_ds = PersistentDataset(data=ld_train, transform=train_transforms, cache_dir=path_output_dir / f"cache_train_{model_name}",)
    val_ds = PersistentDataset(data=ld_val, transform=val_transforms, cache_dir=path_output_dir / f"cache_val_{model_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # device = torch.device("cuda:0")  # TODO - return to cuda (CPU for testing on laptop ONLY)
    device = torch.device("cpu")
    model.to(device)

    max_epochs = 2
    val_interval = 1
    checkpoint_interval = 1 # Save a checkpoint of the training in case restarting is required (temporary).
    major_checkpoint_interval = 100 # Save a permanent copy of the current training at major intervals.
    no_improvement_threshold = 5 # If no improvement in the metric within N validation cycles, stop training.
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, last_epoch=lr_epoch)

    scaler = torch.GradScaler("cuda") if AMP else None

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

        # for batch_data in tqdm(train_loader):
        #     step += 1
        #     inputs, labels = (batch_data["inputs"].to(device), batch_data["label"].to(device))
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
               n_workers: int = 4, 
               batch_size: int = 4):
    # Calculate DICE if possible
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=n_labels)

    test_ds = Dataset(data=ld_test, transform=Compose(preprocessing_transforms))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    post_trans = Compose(postprocessing_transforms)

    device = torch.device("cuda:0")
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
            print(dice_metric.aggregate())  # TODO - get case id/image name and print with this. Save to CSV.         


def main(path_output_dir: Path = Path("C:/data/tf3_localiser_output/")):
    """
    Train jaw bone and canal anatomy model from preprocessed images (see tf3_preprocess_images.py)

    Args:
        path_output_dir (_type_, optional): Path to output directory (will create cache directories, test searches as necessary).
            Defaults to Path("C:/data/tf3_localiser_output/").
    """
    (path_output_dir / "logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(message)s", 
                        handlers=[logging.FileHandler(path_output_dir / "logs/localiser_training_log.txt"), logging.StreamHandler()])
    deterministic_seed = 0

    path_data_dir = Path("C:/data/tf3")

    path_images = path_data_dir / "images_rolm"
    path_labels = path_data_dir / "labels_rolm"

    assert path_data_dir.exists(), f"Data directory {path_data_dir} is missing - can not continue."
    
    # Create or load case IDs and train/test/val split
    path_case_ids_yaml = path_data_dir / "case_id_lists_10_cases.yaml"

    with path_case_ids_yaml.open("r") as f:
        d_case_ids = dict(yaml.safe_load(f))

    # Image transforms (non-random first, so PersistentDataset can function)
    # Clipping/scaling HU - No real reason for a_max to be above 2000 HU for localisation model (metalwork is included in the teeth/jaw labels).
    # Initial images are already set to 1, 1, 1 voxel size, so no rescaling required.

    # TODO - convert initialiser image to signed distance

    # Jaw bones are 1, 2. Implants are 10. Canals are 3, 4, 103, 104, 105.
    filter_labels_local = [1, 2]
    rename_labels_local = [(x, i + 1) for i, x in enumerate(filter_labels_local)]
    
    filter_labels = [1, 2, 3, 4, 10, 103, 104, 105]
    rename_labels = [(x, i + 1) for i, x in enumerate(filter_labels)]
    n_labels = len(rename_labels) + 1

    initial_transforms = [
        LoadImaged(keys=["image", "localiser", "label"]),
        EnsureChannelFirstd(keys=["image", "localiser", "label"]),
        Orientationd(keys=["image", "localiser", "label"], axcodes="RAS"),
        ResampleToMatchd("localiser", "image", mode="nearest", padding_mode="zeros"),
        LabelFilterd("localiser", filter_labels_local), 
        EditLabelsd("localiser", rename_labels_local),
        LabelFilterd("label", filter_labels), 
        EditLabelsd("label", rename_labels),
        RemoveSmallObjectsd("localiser"),  # Clean small pockets of voxels in case these appear in localiser (unlikely)
        ScaleIntensityRanged("image", a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(["image", "localiser", "label"], "localiser", margin=20, allow_smaller=True),
        # SaveImaged(keys=["image"], output_postfix="init_image"),  # Comment out unless testing.
        # SaveImaged(keys=["localiser"], output_postfix="init_local"),  # Comment out unless testing.
        # SaveImaged(keys=["label"], output_postfix="init_label"),  # Comment out unless testing.
    ]

    train_only_transforms = [
        RandAffined(keys=["image", "localiser", "label"], mode=("bilinear", "nearest", "nearest"), prob=0.9, 
                    rotate_range=(np.pi/15, np.pi/15, np.pi/15), scale_range=(0.1, 0.1, 0.1)),            
        RandCropByPosNegLabeld(keys=["image", "localiser", "label"], label_key="label", spatial_size=ROI_SIZE,
                               pos=1, neg=1, num_samples=4, allow_smaller=False),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.90),
        RandGaussianNoised(keys=["image"], prob=0.2),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandGaussianSharpend(keys=["image"], prob=0.2),
        ConcatItemsd(keys=["image", "localiser"], name="inputs"),
        DeleteItemsd(keys=["image", "localiser"]),
    ]

    test_val_only_transforms = [
        ConcatItemsd(keys=["image", "localiser"], name="inputs"),
        DeleteItemsd(keys=["image", "localiser"]),
        EnsureTyped(keys="inputs", dtype=np.float32)
    ]

    train_transforms = initial_transforms + train_only_transforms
    test_val_transforms = initial_transforms + test_val_only_transforms

    # TODO: Convert to dict transforms; convert onehot to real labels again (potentially outside this transform - can use fuction from tf3_preprocess_images.py).
    postprocessing_transforms = [
        AsDiscrete(argmax=True),
        SaveImage(output_dir=path_output_dir / "test_segmentations")
    ]
    
    model = UNETR(
        in_channels=2,
        out_channels=n_labels,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="conv",
        norm_name="batch",
        res_block=True,
        dropout_rate=0.0,
    )

    path_localiser_labels = path_data_dir / "labels_localiser_rolm"

    ld_train = []
    ld_val = []
    ld_test = []

    # Construct train/test/val split, per-tooth, lists of dicts
    for case_ids, ld in [(d_case_ids["train"], ld_train), (d_case_ids["val"], ld_val), (d_case_ids["test"], ld_test)]:
        for c in case_ids:
            d = {"image": path_images / f"{c}.nii.gz", 
                "label": path_labels / f"{c}.nii.gz", 
                "localiser": path_localiser_labels / f"{c}.nii.gz"}
            dm = {"image": path_images / f"{c}_mirrored.nii.gz", 
                "label": path_labels / f"{c}_mirrored.nii.gz", 
                "localiser": path_localiser_labels / f"{c}_mirrored.nii.gz"}
            ld.append(d)
            ld.append(dm)

    train_model(ld_train, ld_val, path_output_dir, model, "jaw_unetr_diceceloss", train_transforms, test_val_transforms, n_labels,
                deterministic_training_seed=deterministic_seed, n_workers=1, batch_size=1, checkpoint_epoch=1, path_checkpoint_model=path_output_dir / "checkpoint_jaw_unetr_diceceloss_epoch_1.pkl")
    
    test_model(ld_test, path_output_dir / "jaw_unetr_diceceloss_best_metric.pkl", model, test_val_transforms, postprocessing_transforms, 
               n_labels, n_workers=1, batch_size=1)

    
if __name__ == "__main__":
    main()
