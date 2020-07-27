# Sequential Depth Completion with Confidence Estimation for 3D Model Reconstruction
### Introduction 
This is a deep neural network for depth completion using sequence information.

The full demo video can be found [here](https://youtu.be/g0GUbv6nbiM)

![](ezgif.com-video-to-gif.gif)

The code will be release soon!

### Installation

    conda create -n venv python=3.7
    conda activate venv
    conda install pytorch==1.1.0 torchvision cudatoolkit=9.0 -c pytorch
    conda install -c conda-forge opencv
    pip install pycuda
    pip install Cython

Using pip for installing the other dependence packages

Next, compiling the library for using warping function. We upgraded this library from the original [here](https://github.com/simon-donne/defusr).

Make sure that the CUDA version on your machine is 9.0. You can check whether the folder `/usr/local/cuda-9.0` exists or not.

    cd depth-completion-seq-net/MYTH
    python setup.py install
    
### Preparation
You need to prepare your dataset followed our template except for KITTI dataset (It should be organized as same as the original).

Dataset structure should look like this:

```
|--train # for training model
    |-- sequence 1
        |--depth*.png  # ground truth depth, '*' here the index 
        |--image*.png  # RGB image
        |--sparse*.png # sparse depth input
        |--pose*.txt   # camera-to-world, 4×4 matrix in homogeneous coordinates
    |-- sequence 2
    ...
|--val # for evaluating our model based on some metrics: RMSE, MAE...
    |-- sequence 1
        |--depth*.png  # ground truth depth, '*' here the index 
        |--image*.png  # RGB image
        |--sparse*.png # sparse depth input
        |--pose*.txt   # camera-to-world, 4×4 matrix in homogeneous coordinates
    |-- sequence 2
    ...
|--test # for depth prediction without ground truth depth
    |-- sequence 1
        |--image*.png  # RGB image
        |--sparse*.png # sparse depth input
        |--pose*.txt   # camera-to-world, 4×4 matrix in homogeneous coordinates
    |-- sequence 2
    ...
```
    
### Training, evaluation end prediction
Make sure activate your virtual environment before running code:

    conda activate venv
    
Train model:

    python train.py --config config_kitti.json
    # all parameters of model are configured in this file
    # file config_kitti.json is used for the kitti dataset
    # file config_aerial.json is used for the aerial dataset

Evaluate model:

We need a pretrained model for evaluation. The pretrained model is put in a folder along with file config. 
See the `pretrains` in the project for illustration. The evaluated dataset is the data in the folder `val`

    python test.py --resume <path to pretrained model> --save_folder <folder to save the results>
    
Prediction:

Predict the depth map from the image and sparse depth input. There's no need ground truth depth (see the structure of `test` folder)

    python predict.py --resume <path to pretrained model> --save_folder <folder to save the results>
