# Pixel2Mesh-Pytorch-TUM

1) Install required packages in requirements.txt
2) Download the dataset using download.py and unzip the files
3) For training : 
- VGG : 
```python
python3 train_vgg.py
```
- ResNet : 
```python
python3 train_res.py
```
- Stereo-ResNet : 
```python
python3 train_str.py
```
4) For demo :
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

5) Compute the F-scores on the test set :
```python
python3 compute_f1.py
```
A complete list of training data can be found in data/training_data/trainer_res.txt for the ResNet model and data/training_data/trainer_stereo.txt for Stereo-ResNet. The available files in the dataset are automatically detected by the loader.

The training data is placed in data/ShapeNet/ and contains large samples of the full dataset.
The testing data is placed in data/Test which is used for computing F-scores.

After every epoch of training, checkpoints and outputs are saved into corresponding folder in temp/RES or temp/STR, depending on which model is being trained. A folder is created with name the date it has been created so we can easily compare current results wiith previous epochs.
Outputs are based on the testing lists, test_list.txt for ResNet and test_list_str.txt for Stereo-ResNet.
Before running the demos or continuing training, the most recent checkpoint must be placed into data/checkpoints.
The path in args can be changed in the respective script to use the appropriate checkpoint.
