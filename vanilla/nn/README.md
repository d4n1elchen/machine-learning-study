# Simple Neural Network Framework Implemented by Numpy

## Preparation

### Download and extract dataset
```shell
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1vCLi2JIKbaDbMIPYM1HJG7lPiyjKACCj" -O mnist.pkl.gz
python mnist_csv3.py
```

### Install dependency
```shell
pip install -r requirements.txt
```
or
```shell
pip install numpy==1.19.4 pandas==1.1.5
```

## Training and testing
```shell
python experiment.py
```