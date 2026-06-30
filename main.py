import argparse
import os
import yaml
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, random_split

# my modules
import unet
from loss import WeightedDiceBCELoss
from dataset import Dataset, DualRotation
from functions import (
    train,
    plot_losses,
    evaluate,
    show_model_outputs,
    test_model_augmentations,
)


def init_weights(m):
    """initialize weights for Conv2d and ConvTranspose2d layers using Kaiming Normal initialization."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        # Set biases to zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_augmentations(config):
    """
    Define data augmentations based on the configuration.
    If augmentations are disabled in the config, return None.
    Otherwise, return a composition of augmentations
    """
    if not config["augmentation"]["enabled"]:
        return None

    return T.Compose(
        [
            DualRotation(degrees=45, fill_img=0.75, fill_mask=0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4),
        ]
    )


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Trainiere U-Net Modelle mit spezifischen Konfigurationen"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Pfad zur .yaml Konfigurationsdatei"
    )
    return parser.parse_args()


def main():
    # 0. Load config
    args = get_args()
    cfg = load_config(args.config)

    # 1. Set seeds
    torch.manual_seed(41)
    random.seed(41)
    np.random.seed(41)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(cfg["paths"]["save_model"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["plot_loss"]), exist_ok=True)

    # 2. Prepare datasets and dataloaders
    # define augmentations
    augmentations = get_augmentations(cfg)

    # full dataset with and without augmentations
    dataset_aug = Dataset(
        images_dir=cfg["paths"]["images"],
        masks_dir=cfg["paths"]["masks"],
        augmentation=augmentations,
    )
    dataset_no_aug = Dataset(
        images_dir=cfg["paths"]["images"],
        masks_dir=cfg["paths"]["masks"],
        augmentation=None,
    )

    # Split: 60% Train, 25% Val, 15% Test
    # We split the base dataset to have consistent indices
    full_len = len(dataset_no_aug)
    train_size = int(0.6 * full_len)
    val_size = int(0.25 * full_len)
    test_size = full_len - train_size - val_size

    train_sub, val_sub, test_sub = random_split(
        dataset_no_aug, [train_size, val_size, test_size]
    )

    # Train set gets augmentations (if enabled), Val/Test remain clean
    train_dataset = torch.utils.data.Subset(dataset_aug, train_sub.indices)
    val_dataset = torch.utils.data.Subset(dataset_no_aug, val_sub.indices)
    test_dataset = torch.utils.data.Subset(dataset_no_aug, test_sub.indices)

    # DataLoaders
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"].get("test_batch_size", 1),
        shuffle=False,
    )

    # initialize model
    print(f"Initialisiere Modell (Tiefe: {cfg['model']['depth']})")
    model = unet.UNetMini(
        feature_channels=cfg["model"]["feature_channels"], depth=cfg["model"]["depth"]
    ).to(device)

    # initialize weights if specified
    if cfg["model"].get("init_weights", False):
        model.apply(init_weights)

    # Loss & Optimizer & Scheduler
    # Loss = 0.5 * Weighted BCE + 0.5 * DICE_loss
    criterion = WeightedDiceBCELoss(weight_bce=0.5, weight_dice=0.5, pos_weight=2.0).to(
        device
    )

    optimizer = optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=0.0
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["training"]["patience"]
    )

    save_path = cfg["paths"]["save_model"]
    training_enabled = cfg["training"]["run_training"]

    # Either load existing model or train a new one
    if not training_enabled and os.path.exists(save_path):
        print(f"Lade existierendes Modell von {save_path}")

    else:
        print("Starte Training")
        train_losses, val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=cfg["training"]["epochs"],
            device=device,
            print_every=1,
            save_path=save_path,
            scheduler=scheduler,
            save_best_val=cfg["training"]["save_best_val"],
        )

    # load the best model for evaluation
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    # plot losses and save figure
    plot_losses(
        model,
        train_losses,
        val_losses,
        model_name=cfg["reporting"]["plot_title"],
        save_path=cfg["paths"]["plot_loss"],
    )

    # evaluate model on test set
    print("Evaluating model on test set")
    IoU, dice_coeff, pixel_accuracy = evaluate(model, test_loader, device)
    print(
        f"IoU: {np.mean(IoU):.4f}, Dice coefficient: {np.mean(dice_coeff):.4f}, Pixel Accuracy: {np.mean(pixel_accuracy):.4f}"
    )

    # show some model outputs
    show_model_outputs(
        model,
        test_loader,
        device,
        cols=6,
        rows=3,
        save_path=cfg["paths"]["plot_output"],
    )

    # Test model with augmentations (Rotation, Flip, Jitter)
    print("Testing model with augmentations...")

    # define augmentations for testing with certainty
    augmentations = T.Compose(
        [
            DualRotation(degrees=45, fill_img=0.75, fill_mask=0),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    )

    test_model_augmentations(
        model,
        augmentations,
        device,
        test_indices=test_sub.indices,   
        postprocessor=None,
        rows=7,
        data_path="./data/",
        save_path=cfg["paths"]["plot_output"].replace(".png", "_augmented.png"),
        seed=41,
    )

if __name__ == "__main__":
    main()
