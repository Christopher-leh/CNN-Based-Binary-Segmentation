import torch
import torch.nn as nn


class dice_loss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Soft Dice Loss for binary segmentation tasks.
        smooth is a small constant added to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # flatten the inputs and targets to 1D tensors
        inputs_flat = inputs.reshape(-1)
        targets_flat = targets.reshape(-1)

        # calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        total = inputs_flat.sum() + targets_flat.sum()

        # calculate dice coefficient
        dice_coeff = (2 * intersection + self.smooth) / (total + self.smooth)

        # return dice loss as 1 - dice coefficient
        return 1 - dice_coeff


class dice_bce_loss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, smooth=1e-10):
        """
        Combined Dice and Binary Cross Entropy Loss.
        weight_dice and weight_bce are the weights for the Dice and BCE losses respectively.
        smooth is a small constant added to avoid division by zero in the Dice loss.
        """
        super().__init__()
        self.dice_loss = dice_loss(smooth)
        self.bce_loss = nn.BCELoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)

        # return the weighted sum of Dice and BCE losses
        return self.weight_dice * dice + self.weight_bce * bce


class WeightedDiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, pos_weight=5.0, smooth=1e-6):
        """
        combines Dice Loss and Weighted Binary Cross Entropy Loss for segmentation tasks.
        weight_dice and weight_bce are the weights for the Dice and BCE losses respectively.
        pos_weight is the penalty for errors on the object (screw) vs background.
        pos_weight > 1 forces the model to take fine details seriously.
        """
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, inputs, targets):

        # flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # calculate dice loss
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_score

        # calculate weighted binary cross entropy loss
        # has to be done manually to include pos_weight

        # clamp inputs to avoid log(0)
        inputs_clamped = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

        # Formula: -(weight * target * log(input) + (1-target) * log(1-input))
        bce_loss = -(
            self.pos_weight * targets * torch.log(inputs_clamped)
            + (1 - targets) * torch.log(1 - inputs_clamped)
        ).mean()

        # return the weighted sum of Dice and BCE losses
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss
