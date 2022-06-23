# Rot-CNN

Standard Convolutional Neural Networks (ConvNets) rely on data augmentation (particularly, rotating training images) to deal with rotation invariant pictures.  We design a new convolution layer that is rotation invariant by nature.  As a result, we don't need to rotate the training images for preprocessing.

# Installation

Tested under Ubuntu with Python 3.10.

```
pip install -r requirements.txt
```

## Usage 

Execute `main.py` in each folder.  The required datasets are automatically downloaded.

Here are several commands for you to use : 

``` 
    --dataset 1     #choose dataset MNIST
    --dataset 2     #choose dataset FashionMNIST
    --train         #retrain a model, default : false, load trained model
    --augment       #do data augmentation, default : false
```
