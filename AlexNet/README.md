# AlexNet 

## Paper 

This paper was particularly important to popularize deep learning and to push further the research interest in this field. 
The set of achievements described in the paper goes much beyond a simple performance improvement on the ImageNet dataset
and respective competition. The paper also proved that it is possible to go deeper, to build more complex models and 
increase the performance of previous models or architectures. To do that, for the first time, some new techniques were 
used together to improve the capability of the model. 

This is not only a nice paper with an even nicer architecture and results, but it also is a landmark in the history of 
deep learning.

### Introduction 

Until a few years before this paper, the size of datasets was not considerably big, therefore some simple model 
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

Pooling layers are a popular way to resume the information in some input information, by moving a sliding window by some
 value (stride!) and capturing values in a square of a specific size (window size). The information in the window is 
 resumed either by summing all the elements, selecting the max or just averaging them. 

Previous implementations of pooling layers, either max or sum pooling generally consider the stride to be equal to the 
size of the window. Although, the authors found that it improves the error rate and reduces the probability of 
overfitting when the pooling layer has a stride lower than the window size. This creates overlapping information since 
some positions on the input are going to be used more than once to calculate the output. 

#### Overall review of the architecture 

The architecture for a single GPU or CPU (paper shows the architecture for 2 GPUs) is the following: 

- Convolutional Layer (Input = 224x224x3, Kernel = 11x11x96, stride = 4)
- ReLU nonlinearity
- Response-normalization Layer
- Max Pooling Layer (stride = 2, window size = 3)
- Convolutional Layer (Input = 27x27x96 , Kernel = 5x5x256, stride = 1)
- ReLU nonlinearity
- Response-normalization Layer
- Max Pooling Layer (stride = 2, window size = 3)
- Convolutional Layer (Input = 13x13x256 , Kernel = 3x3x384, stride = 1)
- ReLU nonlinearity
- Convolutional Layer (Input = 13x13x384 , Kernel = 3x3x384, stride = 1)
- ReLU nonlinearity
- Convolutional Layer (Input = 13x13x384 , Kernel = 3x3x256, stride = 1)
- ReLU nonlinearity
- Max Pooling Layer (stride = 2, window size = 3)
- Fully-connected Layer (Input = 6x6x256, Output = 4096, Neurons = 4096)
- ReLU nonlinearity
- Fully-connected Layer (Input = 4096, Output = 4096, Neurons = 4096)
- ReLU nonlinearity
- Fully-connected Layer (Input = 4096, Output = 1000, Neurons = 4096)
- ReLU nonlinearity



### Addressing Overfitting

Overfitting was a nightmare of most machine learning researchers during years, and despite still being a recurrent 
problem to several problems and models, recent developments lead to new methods to handle and reduce the probability of 
overfitting. On this section, the authors discussed some of those methods that helped to improve the stability of this 
network and further explain why that happens. Not only that but the authors also tried to use methods that do not 
increase too much the computation time needed. 

#### Data Augmentation 

The paper describes two techniques to artificially increase the amount of data available, that are in practice, 
computationally free, since they have a minimal cost to be computed, and are computed on the CPU while the previous 
batch is training on the GPU. 

The first approach generates horizontal reflexions of the images and translations. This is done by extracting random 
224x224 patches from the 256x256 images and use them to train the network. This leads to 2048 times higher training set 
but with a lot of interdependent instances.  At test time, the four corners and the center are extracted as 224x224 
patches and their reflexions. To obtain the final prediction an average of the predictions on the 10 patches is taken 
into account. 

The second approach is related to the changes on the RGB channels, it is said by the authors that they perform the 
principal component analysis on the RGB pixel values through the dataset. Afterward, they add multiples of the 
principal components found to the original image. These multiples have magnitudes derivated from the corresponding 
eigenvalues times a random value drawn from a Gaussian (mean 0, std 0.1). 
Each value is drawn only once per image on one iteration, and there is 1 value per RGB channel. This second approach 
helps the neural network to understand that the object does not change based on the color or ilumination. 

#### Dropout 

Dropout was a recent technique, that implements a very efficient and doable model combination technique, that remains 
feasible when applied to big neural networks. Nevertheless, it does increase the training time by a factor of 2. 

The basic idea of dropout is for each neuron of a neural network the output of the neuron will be 0 with a probability 
of 0.5 (this probability can change). Those neurons will not be part of the forward and backward pass of a neural 
network. The final result is that at each training iteration the network architecture will be different, however, it 
will be sharing weights with previous architectures. 

At the test time, there are no changes to the architecture, but the output of each neuron is multiplied by 0.5, which 
the authors say to be a reasonable approximation to the geometric approximation to many dropout networks-

### Training and Learning 

Details of training and learning are further explained in the implementation section below.


## Implementation/Architecture Reproduction 

