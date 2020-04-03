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
Change arguments with the parser 

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
