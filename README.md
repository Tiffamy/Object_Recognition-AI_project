# AI_project:Object Recognition
This is the final project of "Introduction to Aritificial Intelligence", and topic of this project is dicided as "Object-Recognition".

## Enviroment
Multi-GPU training on Ubuntu Serer; CUDA version 10.1; Python 3.6+.
> ###### torch==1.7.1, torchvision==0.8.2, torchaudio==0.7.2, scipy==1.6.3, opencv-python==4.5.2.54

## Dataset
We use "CIFAR-10" as our dataset, which has 60000 32x32 images and 10 different classes (50000 for train, 10000 for test).

![image](https://github.com/Tiffamy/Object_Recognition-AI_project-/blob/main/image/cifar-10.png)
## Usage
You can simply train the model by execute train.py with some user-defined parameter.

```
# You can train the specific model with: 
python3 main.py --model=$(model_you_want_to_use)

# We provide following models for you:
--model=resnet18
--model=vgg16
--model=mobilenet
--model=mobilenet_v2
--model=densenet121
--model=googlenet
--model=resnext29
--model=our_model
--model=our_model_v2
--model=our_model_v3
--model=our_model_v4
```
There are some parameter which can be defined by user.
```
# You can specific some learning parameters by given following arguments:
--lr=$(LR)  # default is 0.1
--epoch=$(epochs)  # default is 200
--train_batch=$(train_batch_size)  # default is 128
```
You can use our pretrained models for quickly show the result.  
```
# You can use pretrained model provided by us:
python3 main.py --model=$(model_you_want_to_use) --pretrained
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
You can use your own image and model provided by us to predict which class this image belongs to.  
```
# If the image's resolution is far from 32x32, the result might be bad.
python3 img_clf.py --model=$(model_you_want_to_use) $(image_path)
```
* Note that you need to download pretrained model first or prepare your own pretrained model(.pth) which should be stored as above folder structure.
## Structure of our models
our_model:

![image](https://github.com/Tiffamy/Object_Recognition-AI_project-/blob/main/image/v1.jpg)

our_model_v2:

![image](https://github.com/Tiffamy/Object_Recognition-AI_project-/blob/main/image/v2.jpg)
## Results on different models
