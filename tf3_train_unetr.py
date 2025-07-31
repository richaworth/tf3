import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import matplotlib.pyplot as plt

from monai.data.utils import decollate_batch 
from monai.data.dataset import PersistentDataset
from monai.data.dataloader import DataLoader
from monai.handlers.utils import from_engine
from monai.losses.dice import DiceCELoss
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.networks.nets import unetr

from monai.transforms.post.array import AsDiscrete
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundD, RandCropByPosNegLabelD
from monai.transforms.io.dictionary import LoadImageD
from monai.transforms.intensity.dictionary import RandShiftIntensityD, ScaleIntensityRangeD
from monai.transforms.spatial.dictionary import OrientationD, RandAffineD, SpacingD
from monai.transforms.utility.dictionary import EnsureChannelFirstD

from monai.utils.misc import set_determinism
import numpy as np
import onnxruntime
from tqdm import tqdm

import torch

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

    # Get labels etc. from dataset json
    path_images = path_data_dir / "imagesTr"
    path_labels = path_data_dir / "labelsTr"
    path_bony_labels = path_data_dir / "labels_bones_only"

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
        LoadImageD(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        OrientationD(keys=["image", "label"], axcodes="RAS"),
        SpacingD(keys=["image", "label"], pixdim=(0.3, 0.3, 0.3), mode=("bilinear", "nearest")),
        ScaleIntensityRangeD(keys=["image"], a_min=-1000, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
    ]

    train_only_transforms = [            
        RandAffineD(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.9, 
                    rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
        RandCropByPosNegLabelD(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96),
                               pos=1, neg=1, num_samples=8, image_key="image", image_threshold=0.01),
        RandShiftIntensityD(keys=["image"], offsets=0.10, prob=0.50),
    ]
    
    train_transforms = Compose(initial_transforms + train_only_transforms)
    val_transforms = Compose(initial_transforms)

    # Localisation/bony segmentation only 
    ld_train = [{"image": path_images / f"{c}_0000.nii.gz", "label": path_bony_labels / f"{c}.nii.gz"} for c in case_ids_train]
    ld_val = [{"image": path_images / f"{c}_0000.nii.gz", "label": path_bony_labels / f"{c}.nii.gz"} for c in case_ids_val]

    train_ds = PersistentDataset(data=ld_train, transform=train_transforms, cache_dir=path_output_dir / "cache_labels_bony_train")
    val_ds = PersistentDataset(data=ld_val, transform=val_transforms, cache_dir=path_output_dir / "cache_labels_bony_val")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    # Set up model - testing UNETR. Defaults from MONAI tutorial unetr_btcv_segmentation_3d.ipynb

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda")

    model = unetr.UNETR(
        in_channels=1,
        out_channels=3,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Define validation and train functions
    def validation(epoch_iterator_val):
        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 8, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val

    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

            epoch_iterator.set_description(  # noqa: B038
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            
            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), path_output_dir / f"best_metric-step_{step}.pth")
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                else:
                    torch.save(model.state_dict(), path_output_dir / f"checkpoint-step_{step}.pth")
            global_step += 1
        return global_step, dice_val_best, global_step_best

    max_iterations = 20000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=3)
    post_pred = AsDiscrete(argmax=True, to_onehot=3)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
    model.load_state_dict(torch.load(os.path.join(path_output_dir / "BT", "best_metric_model.pth"), weights_only=True))

    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")

if __name__ == "__main__":
    main()
