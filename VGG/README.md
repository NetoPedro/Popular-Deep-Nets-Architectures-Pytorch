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



#### Configurations and Discussion 

### Classification Framework 

#### Training

#### Testing 

### Classification Experiments 

#### Single Scale Evaluation 

#### Multi-Scale Evaluation

#### Multi-Crop Evaluation 


## Implementation/Architecture Reproduction 

