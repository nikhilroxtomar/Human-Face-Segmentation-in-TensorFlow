# Human Face Segmentation in TensorFlow

This repository contains the code for `Multiclass Segmentation` on the human faces using `Landmark Guided Face Parsing` (LaPa) dataset. 

The following models are used:
- [UNET](https://arxiv.org/abs/1505.04597)

Models to be used in future:
- 
- DEEPLABV3+
- more...

# Dataset
The LaPa dataset contains the training, validation and testing dataset. Each dataset have images, segmentation mask and the 106 human facial key points.

Original Image             |  Grayscale Mask           | RGB Mask
:-------------------------:|:-------------------------:|:-------------------------:
![](img/image.jpg)  |  ![](img/grayscale_mask.png)  |  ![](img/rgb_mask.png)
