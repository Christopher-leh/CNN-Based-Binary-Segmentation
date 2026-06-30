import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import random

from dataset import Dataset


def show_images(images, cols=4, titles=None, big_title=None, save_path=None):
    """
    Display a list of images in a grid format.
    """

    if not images:
        return

    n = len(images)

    # Calculate the number of rows needed based on the number of images and columns
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 3, rows * 3))

    if big_title:
        plt.suptitle(big_title, fontsize=16)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if titles and i < len(titles):
            plt.title(titles[i])
        cmap = "gray"
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def show_model_outputs(
    model,
    data_loader,
    device,
    cols=4,
    rows=3,
    postprocess=None,
    title=None,
    save_path=None,
):
    """
    Display model outputs alongside input images and target masks.
    still has parameter for postprocessing, since i experimented with it.
    """

    model.eval()

    # create lists to hold images and titles for the show_images function
    images = []
    titles = []

    with torch.no_grad():
        for i, (inputs, masks) in enumerate(data_loader):

            inputs = inputs.to(device)
            masks = masks.to(device)

            # check for postprocessing(wasnt implemented)
            if postprocess:
                outputs = postprocess(model(inputs))
                outputs = (outputs > 0.5).float()
            else:
                outputs = model(inputs)
                outputs = (outputs > 0.5).float()  # binarization

            for j in range(inputs.size(0)):

                # break the loop if we have enough images to display
                if len(images) >= cols * rows:
                    break

                # change tensors to numpy arrays on cpu for visualization
                inp_img = (
                    inputs[j].cpu().squeeze().numpy()
                )  # squeeze to remove channel dimension for grayscale images
                mask_img = masks[j].cpu().squeeze().numpy()
                out_img = outputs[j].cpu().squeeze().numpy()
                images.append(inp_img)
                titles.append("Input Image")
                images.append(mask_img)
                titles.append("Target")
                images.append(out_img)
                titles.append("Model Output")
    # use the show_images function to display the images in a grid
    show_images(images, cols=cols, titles=titles, big_title=title, save_path=save_path)


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    print_every=5,
    save_path="./models/cnn_model.pth",
    scheduler=None,
    save_best_val=True,
):
    """
    Train the model and validate it.
    """
    # lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        # running loss for the epoch
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # accumulate the loss
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(
            train_loader.dataset
        )  # average loss for the epoch
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            # Validation loop
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Step the scheduler if given, with special handling for ReduceLROnPlateau
        # different handling for ReduceLROnPlateau, since it requires the validation loss to step, while other schedulers step per epoch.
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}"
            )

        # Save the best model based on validation loss, if save_best_val is True
        # saves only if the current epoch's validation loss is lower than all previous epochs' validation losses.
        if save_best_val and (epoch == 0 or epoch_val_loss < min(val_losses[:-1])):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # save model and losses
            checkpoint = {
                "model_state_dict": model.state_dict(),  # model weights
                "train_losses": train_losses,
                "val_losses": val_losses,
                "epoch": epoch,  # epoch number of the best model
            }
            best_epoch = epoch + 1
            torch.save(checkpoint, save_path)

    if save_best_val:
        print(
            f"   -> Model saved at epoch {best_epoch} with Val Loss: {min(val_losses):.4f}"
        )
    else:
        # save the model at the end of training
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": num_epochs,
        }
        torch.save(checkpoint, save_path)
        print(f"   -> Model saved at epoch {num_epochs}")

    # return the training and validation losses for plotting
    return train_losses, val_losses


def evaluate(model, dataloader, device, threshold=0.5, postprocess=None):
    """
    Evaluate the model on the given dataloader.
    Again postprocess is not implemented, but the parameter is still there for future use.
    Returns lists of IoU, Dice coefficient, and Pixel Accuracy for each batch.
    threshold for binarization of model outputs. It could be helpful to adjust this if the model outputs are favoring one class too much.
    """

    model.eval()
    IoU_list = []
    dice_coeff_list = []
    pixel_accuracy_list = []

    # loop over data
    with torch.no_grad():
        for inputs, masks in dataloader:

            inputs = inputs.to(device)
            masks = masks.to(device).float()
            outputs = model(inputs)

            # Binarize outputs based on the threshold
            outputs = (outputs > threshold).float()

            if postprocess:
                outputs = postprocess(outputs)

            # Initialize sums for metrics
            inter_sum = 0.0
            union_sum = 0.0
            total_sum = 0.0
            correct_pixels = 0
            total_pixels = 0

            # Calculate intersection, union, and total
            intersection = (masks * outputs).sum()
            total = (masks + outputs).sum()
            union = total - intersection

            # Update sums
            union_sum += union.item()
            inter_sum += intersection.item()
            total_sum += total.item()

            # Calculate correct pixels for accuracy
            correct_pixels += (outputs == masks).sum().item()
            total_pixels += outputs.numel()

            # Calculate IoU, Dice coefficient, and Pixel Accuracy
            IoU = inter_sum / max(union_sum, 1e-10)
            dice_coeff = 2 * inter_sum / max(total_sum, 1e-10)
            pixel_accuracy = correct_pixels / total_pixels

            # Append metrics to lists
            IoU_list.append(IoU)
            dice_coeff_list.append(dice_coeff)
            pixel_accuracy_list.append(pixel_accuracy)

    # chose to return lists and not means for more flexibility and insight
    return IoU_list, dice_coeff_list, pixel_accuracy_list


def plot_losses(model, train_losses, val_losses, model_name="Model", save_path=None):
    """
    Plot training and validation losses over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.ylim(0, 0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss over Epochs for {model_name}")
    plt.legend()
    # save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()


def test_model_augmentations(
    model,
    augmentations,
    device,
    test_indices,  #  Indizes aus main.py übergeben
    postprocessor=None,
    rows=3,
    data_path="./data/",
    save_path=None,
    seed=41,  # for reproducibility of augmentations
):
    """
    Test the model on an augmented test set and display results.
    test_indices: Indices of the test set to be used for evaluation. This allows for consistent evaluation across different models or runs.
    """

    # create a dataset with augmentations
    dataset_aug = Dataset(
        images_dir=f"{data_path}images/",
        masks_dir=f"{data_path}masks/",
        augmentation=augmentations,
    )
    model.eval()

    # create a DataLoader for the augmented test set, with the given test_indices to ensure consistent evaluation.
    test_dataset = torch.utils.data.Subset(dataset_aug, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # seed for reproducibility of augmentations
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # evaluate the model on the augmented test set
    # if postprocessor is provided, it will be applied to the model outputs before evaluation.
    IoU, dice_coeff, pixel_accuracy = evaluate(
        model, test_loader, device, postprocess=postprocessor
    )

    print("\nEvaluation on Test Set with Augmentations:")
    print(
        f"mean values: IoU: {np.mean(IoU):.4f}, "
        f"Dice coefficient: {np.mean(dice_coeff):.4f}, "
        f"Pixel Accuracy: {np.mean(pixel_accuracy):.4f}"
    )

    # show model outputs on the augmented test set
    show_model_outputs(
        model,
        test_loader,
        device,
        cols=6,
        rows=rows,
        title=None,
        save_path=save_path,
    )
