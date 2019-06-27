# AlexNet (Work in progress)

## Paper 

This paper was particularly important to popularize deep learning and to push further the research interest in this field. 
The set of achievements described in the paper goes much beyond a simple performance improvement on the ImageNet dataset
and respective competition. The paper also proved that it is possible to go deeper, to build more complex models and 
increase the performance of previous models or architectures. To do that, for the first time, some new techniques were 
used together to improve the capability of the model. 

This is not only a nice paper with an even nicer architecture and results, but it also is a landmark in the history of 
deep learning.

### Introduction 

Until a few years before this paper, the size of the dataset was not considerably big, therefore some simple model 
conjugated with data augmentation techniques were enough to achieve a considerable performance and avoid overfitting. 
With the release of ImageNet dataset and others of the same kind, previous models became rather insufficient to learn 
all the complexities in the data distribution. Therefore other, more uncommon, models started to spark the interest of 
researchers like neural networks. 

However, using a neural network was not enough to achieve the desired performance with the new datasets. Neural networks 
were particularly challenging to train, with a substantial computational cost and prone to overfitting. Therefore the 
previous research papers on new activations like 
[ReLU](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.6419&rep=rep1&type=pdf), regularization methods like 
[Dropout](https://arxiv.org/pdf/1207.0580.pdf) and the presented 
implementation of efficient convolutions on a GPU were positively significant to the architecture described here. 


### Dataset 

ImageNet was one of the introductory datasets to gather millions of examples, with over 15 millions of images 
distributed across 22 thousand categories. The existence of such a large dataset was one key element to the emergence of 
deep learning and the popularity of neural networks as models with a significant capability to learn complex functions. 
The dataset is in its [2010 version](http://www.image-net.org/challenges/LSVRC/2010/) already split between test, 
validation and train data. 

In the dataset, to evaluate each model, two errors are measured. The first, top-1, refers to situations where the 
correct label is the label with the highest probability accordingly to the model. The second, top-5, refers to 
situations where the true label is in the five predictions with larger probabilities.

All the images are further scaled to be possible to use them as the input of the same neural network. 
The chosen scale is 256x256. 

### Architecture 

The architecture of this network surprised by the number of layers (8!) and some uncommon features at that time like 
ReLU and Dropout. 

#### Relu Nonlinearity 

Activation used on previous paper were particularly difficult to train. Some examples of those activations, 
commonly mentioned as 
saturating nonlinearities, are the sigmoid and tanh functions. ReLU, also called non-saturating nonlinearity, is 
considerably faster and easier to train. 
 
- ReLU -> max(0,x)

- Tanh -> tanh(x)

- Sigmoid -> ![sigmoid](https://latex.codecogs.com/gif.latex?%281%20&plus;%20e%5E%7B-x%7D%29%5E%7B-1%7D) 

Previous papers already tried some different activations, e.g. f(x) = |tanh(x)| by [Jarrett et al.](https://ieeexplore.ieee.org/document/5459469). 
The ability of networks using ReLU to learn faster are crucial to performance on large datasets. 

#### Local Response Normalization 

The use of ReLU removes the necessity no have input normalization. Nevertheless, the authors found that it still 
remains beneficial to network performance to apply some normalization after the nonlinearity in some layers. 

![equation](https://latex.codecogs.com/gif.latex?b_%7Bx%2Cy%7D%5Ei%20%3D%20a_%7Bx%2Cy%7D%5Ei%20%28k%20&plus;%20%5Calpha%20%5Csum_%7Bj%3Dmax%280%2Ci-n/2%29%7D%5E%7Bmin%28N-1%2Ci&plus;n/2%29%7D%20%28a_%7Bx%2Cy%7D%5Ej%29%5E2%29%5E%5Cbeta)

In the equation above, used to normalize the data between layers, 'b' represents the normalized data, 'a' is the 
activity of the neurons belonging to the layer. The remaining letters are hyperparameters, thus calculated by 
comparing the results of the network in the validation set. 

#### Overlapping pooling 


#### Overall review of the architecture


### Addressing Overfitting

#### Data Augmentation 


#### Dropout 


### Training and Learning 

Details of training and learning are further explained in the implementation section below.


## Implementation/Architecture Reproduction 

