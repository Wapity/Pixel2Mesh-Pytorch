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
For all, arguments can be changed in the args. The list of all training data is in data/training_data/trainer_res.txt for ResNet and trainer_stereo.txt for Stereo-ResNet. The available files in the dataset are automatically detected by the loader. The training dataset is put in data/ShapeNet/ and contains large sample of full dataset. The testing data is put in data/Test for computing the F-scores. After each epoch of training, checkpoints and outputs are savec in temp/RES or temp/STR. The outputs are based on the testing lists test_list.txt for ResNet and test_list_str.txt for Stereo-ResNet. In data/ckeckpoints are placed the final checkpoints for running the demos of the models or mre training. 

