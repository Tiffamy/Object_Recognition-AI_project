# Object-Recognition-AI_project-
This is the final project of "Introduction to Aritificial Intelligence", and topic of this project is dicided as "Object-Recognition" temporally.

## Enviroment
GPU training on Ubuntu Serer; CUDA version 10.1; Python 3.6+.
> ###### torch==1.7.1, torchvision==0.8.2, torchaudio==0.7.2 

## Dataset
We use "CIFAR-10" as our dataset, which has 60000 32x32 images and 10 different classes (50000 for train, 10000 for test).

![image](https://github.com/Tiffamy/Object_Recognition-AI_project-/blob/main/image/cifar-10.png)
## Usage
You can simply train the model by execute train.py with some user-defined parameter.

```
# You can train the specific model with: 
python3 main.py --model=$(model_you_want_to_use)
```
There are some parameter which can be defined by user.
```
# You can specific some learning parameters by given following arguments:
--lr=$(LR)  # default is 0.1
--epochs=$(epochs)  # default is 200
```
You can use our pretrained models for quickly show the result.  
```
# You can use pretrained model provided by us:
python3 main.py --model=$(model_you_want_to_use) --pretrained=True
```
Download pretrained models [HERE](https://drive.google.com/drive/folders/18hrbUlK1-fwN2j3exj2JmIf_pVcZTL_U?usp=sharing) and put all the .pth files into pretrained directory.  
An example folder structure:
```
root
└── pretrained
    └── resnet18.pth
    └── vgg16.pth
    ...
└── main.py
...
```

## Results on different models
