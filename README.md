# Pixel2Mesh-Pytorch-TUM

Put ShapeNetP2M in the data/training_data folder
///////
To train the ResNet+GCN with a single image :
```
python train_res.py
```
or
```
train_str.py
```
to train the ResNet+GCN with two images :

```
python str_res.py
```
To train one can specify parameters, type
```
python train_res.py --help
```
or
```
python train_str.py --help
```
to check available parameters. These parameters can be added directly in the console, e.g for training 10 epochs :
```
python train_res.py --epochs 10
```
When training train_res checkpoints are saved and outputs are created after each epoch. When training train_str just checkpoints are created.

The demos can be run using any of the previous checkpoints created with the good network and create meshes into data/outputs folder.

The demo with VGG is running by transfer of the tensorflow weights (checkpoints/tf_vgg_checkpoint.pt). For training Res and Str, the initialization is made with the same weights (ckeckpoints/).
