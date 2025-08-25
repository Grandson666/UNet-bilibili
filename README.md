# UNet-bilibili

## Project Overview

`UNet-bilibili` is a lightweight and educational implementation of the U-Net architecture for image segmentation tasks, written in PyTorch. The project demonstrates the essential logic of U-Net and provides a clear reference for learning and experimentation on medical image segmentation, particularly with the DRIVE dataset.

## Project Structure

```
UNet-bilibili/
│
├── data.py
├── model.py
├── train.py
├── predict.py
└── __pycache__/
```

## File Details

### data.py

- Implements a custom dataset class for loading images and masks from the DRIVE dataset.
- Defines folder structure and transformation pipelines for preprocessing.
- Provides the `get_dataloader` function to generate PyTorch dataloaders for training and testing.

### model.py

- Contains the implementation of the U-Net architecture:
  - `DoubleConv`, `Down`, `Up`, and `OutConv` building blocks.
  - The `UNet` class, which organizes the full model structure and forward pass.

### train.py

- Provides the training workflow for the U-Net model.
- Initializes the model, optimizer, and loss function.
- Executes the training loop and saves the trained model weights to disk.

### predict.py

- Demonstrates how to use a trained U-Net model for inference on test images.
- Loads the saved model and applies it to new images.
- Saves the predicted segmentation mask to disk.

## Quick Start

1. Install dependencies:  
   ```
   pip install torch torchvision pillow numpy
   ```

2. Ensure the `DRIVE` dataset is placed in the project directory.

3. Train the model:  
   ```
   python train.py
   ```

4. Predict on a test image:  
   ```
   python predict.py
   ```

## License

This repository is for educational use and does not include a specific license.
