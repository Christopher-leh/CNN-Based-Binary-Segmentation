Computer Vision Project - CNN-Based Binary Segmentation

This project implements a Convolutional Neural Network (CNN) for binary image segmentation using PyTorch. The model is based on the U-Net architecture and is trained to segment objects from images.
The project compares two models, one trained with and one without data augmentation and evaluates both on a clean and a augmented test set to asses the robustness of the models.

## Structure
- `main.py`: Main script to train and evaluate the CNN model.
- `unet.py`: Contains the U-Net architecture definition.
- `dataset.py`: Contains dataset and augmentation class.
- `functions.py`: Utility functions for training, evaluation, and visualization.
- `loss.py`: Custom loss functions for segmentation tasks.
- `train_data_loop.ipynb`: Jupyter notebook for creating segmentation masks using a data loop.
- `config_aug.yaml`: Configuration file for training with data augmentation.
- `config_no_aug.yaml`: Configuration file for training without data augmentation.
- `report/`: Directory containing the project report and figures.
- `data/`: Directory containing the image and mask datasets.

## Data
 Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. International Journal of Computer Vision 129(4): 1038-1059, 2021. Available at: https://www.mvtec.com/company/research/datasets/mvtec-ad

## Usage
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
2. Configure the training parameters in `config_aug.yaml` or `config_no_aug.yaml`.
    One for training with data augmentation and one without.

3. Run the main script:
   ```
   python main.py --config config_aug.yaml
   ```
   or
   ```
   python main.py --config config_no_aug.yaml
   ```  
# CNN-Based-Binary-Segmentation
