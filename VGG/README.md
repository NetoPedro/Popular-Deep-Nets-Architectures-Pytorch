# VGG (Work in progress)

## Paper 

This paper, authored by Oxford researchers, presents an architecture, not good enough to win the ImageNet competition, 
although with extremely interesting details and scientific considerations that inspired future work. Therefore it is 
one of the most popular and influential papers in Deep Learning growth as a research field.

It is shown also a major performance improvement when compared to the 2-years older AlexNet, the network that 
popularized Deep Learning.

### Introduction 

The main concern of this paper is to address the depth of the convolutional neural network. It is done with parameters tuning combined with small iterative increases in the number of layers. The essential element that allows this process to become feasible is the size of the filters ( 3x3 on all Conv layers). The paper further explains how such small filters can capture the same information of others with window sizes of 7. 


### ConvNet Configurations 

This section explains with detail the configuration of the network that will be studied, as well as the configuration of the evaluation process. 

#### Architecture

The network requires an input with 3 channels (RGB) and a resolution of 224x224. For the convolutions, all layers have a filter size of 3x3. To argue with the choice of this filter size, authors mention that this is the smallest size to include a notion of left/right, up/down and center. With a stride of 1, the chosen padding of 1, is selected to keep the input and the output sizes equal, in terms of resolution. The resolution reduction is achieved with 5 max-pooling layers after specific convolutions, these reductions are not overlapping, meaning that the stride is not lower than the window size (2x2 for the window and 2 for the stride).  

Finally, all the networks share the same structure of fully connected layers, with 3 layers of 4096, 4096 and 1000 hidden units respectively.

All the above-mentioned layers are followed by reLU nonlinearity. 

#### Configurations and Discussion 

### Classification Framework 

#### Training

#### Testing 

### Classification Experiments 

#### Single Scale Evaluation 

#### Multi-Scale Evaluation

#### Multi-Crop Evaluation 


## Implementation/Architecture Reproduction 

