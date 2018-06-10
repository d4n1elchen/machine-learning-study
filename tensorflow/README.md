# TensorFlow setup note

## Install CUDA and cuDNN

For enabling GPA support, we need to install CUDA and cuDNN (only for NVIDIA graphic card, you can skip this step if your computer doesn't have a graphic card or not a NVIDIA one)

Ref:
- [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [cuDNN Installation Guide](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
- [ubuntu 17.10 + CUDA 9.0 + cuDNN 7 + tensorflow源码编译](https://zhuanlan.zhihu.com/p/30781460)

The references above are for CUDA 9.0 and cuDNN 7, but the latest release of TensorFlow (1.4) only support CUDA 8.0 and cuDNN 6.

So we must install CUDA 8.0 and cuDNN 6 to make TensorFlow work.

### CUDA 8.0

Install lastest version of your graphic card first.

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-384 nvidia-384-dev
```

In ubuntu, you can directly install CUDA 8.0 from apt.

```bash
sudo apt install nvidia-cuda-toolkit
```

### cuDNN 6

Download cuDNN v6.0 for CUDA 8.0 on [NVIDIA cuDNN Home Page](https://developer.nvidia.com/cudnn). (Click "Download" and register an account or login to an exist account, then download "cuDNN v6.0 Library for Linux" in "Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0")

Install cuDNN

```bash
cd <Your download directory>
tar -zxvf cudnn-8.0-linux-x64-v6.0.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

### Setup Enviroment Variable

Add the following line to your `.zshrc` or `.bashrc`

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
```

## Install TensorFlow in Anaconda

Ref: [Installing TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux)

`virtualenv` is prefered in the official guide, but I choose to use anaconda.

### Download Anaconda and Install

Download latest Anaconda for Python 3.6 from their [official website](https://www.anaconda.com/download/#linux)

Run the installation

```bash
chmod +x Anaconda3-5.0.1-Linux-x86_64.sh
./Anaconda3-5.0.1-Linux-x86_64.sh
```

### Create an enviroment and activate it

```bash
conda create -n tensorflow python=3.6
source activate tensorflow
```

To deactivate conda enviroment

```bash
source deactivate
```

### Install TensorFlow

Make sure your conda enviroment activated first, then run following command to install TensorFlow

```bash
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl
```

### Test

Run following python code in python shell after installation finished

```python
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

The expected output should be

```
Hello, TensorFlow!
```

