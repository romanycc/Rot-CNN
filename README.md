# Rot-CNN

## Background
* Normally, CNN uses data augmentation to successfully predict the image which have been rotated. 

* We try to design a new method to deal with the rotated image instead of using data augmentation.

* Function flip_filter is called at the stage of initiallization and backward propagation to restrict the filter to be symmetric.

* Ideally, the new model should work well on the rotated image, and additionally reduce training time by significantlly decrease the number of parameters.

## Usage 

* Download the whole code, and execute main.py in each folder.

* The dataset will automatically download to the data folder.

* Here are several commands for you to use : 

``` 
    --dataset 1     #choose dataset MNIST
    --dataset 2     #choose dataset FashionMNIST
    --train         #retrain a model, default : false, load trained model
    --augment       #do data augmentation, default : false
```
