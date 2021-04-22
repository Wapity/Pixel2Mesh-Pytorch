# Pixel2Mesh-Pytorch
Improved Pytorch version of Tensorflow Pixel2Mesh that converts 2D RGB Images in 3D Meshes, with ResNet for perceptual feature pooling network and Stereo Input of RGB Images under different angles of camera. 
## Requirements
Install required packages in requirements.txt
## Dataset
Download the dataset using download.py and unzip the files.
A complete list of training data can be found in data/training_data/trainer_res.txt for the ResNet model and data/training_data/trainer_stereo.txt for Stereo-ResNet. The available files in the dataset are automatically detected by the loader.
The training data is placed in data/ShapeNet/ and contains large samples of the full dataset.
The testing data is placed in data/Test which is used for computing F-scores.

## Training 
- VGG or ResNet :
```python
python3 train_res.py
```
- Stereo-ResNet : 
```python
python3 train_str.py
```
After every epoch of training, checkpoints and outputs are saved into corresponding folder in temp/RES or temp/STR, depending on which model is being trained. A folder is created with name the date it has been created so we can easily compare current results wiith previous epochs.
Outputs are based on the testing lists, test_list.txt for ResNet and test_list_str.txt for Stereo-ResNet.
Before running the demos or continuing training, the most recent checkpoint must be placed into data/checkpoints.
The path in args can be changed in the respective script to use the appropriate checkpoint.

## Demo 
- VGG : 
```python
python3 demo_vgg.py
```
- ResNet : 
```python
python3 demo_res.py
```
- Stereo-ResNet : 
```python
python3 demo_str.py
```
Change locations of checkpoints or testing list 

## Compute the F-scores on the test set 
```python
python3 compute_f1.py
```
