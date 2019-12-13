# AlexNet 

## Paper 


### Introduction 


### Deep Residual Learning 

#### Residual Learning 

#### Identity Mapping by Shortcuts


### Architecture 

#### Residual Block

#### Bottleneck

#### Overall review of the architecture 



### Training and Learning 

Details of training and learning are further explained in the implementation section.


## Implementation/Architecture Reproduction 

Dataset| Training Epochs      | Cross-Entropy Training Loss    |  Accuracy |
:-------------: | :-------------: |:-------------:| :-------------:|
CIFAR10 | 15 | 0.329 | 72.2% |
CIFAR100 | 30 |  0.872 | 28.6%  |
Fashion_MNIST | 5 |  0.272|  89.4%  |

### Discussion 

The network failed most predictions on the CIFAR100 dataset, nevertheless I do believe that if instead of considering Top-1 error, we considered Top-5 error rate it would be presenting good results (this assumption is made considering the drop in the cross-entropy loss, that may not be fully captured by the accuracy on the test set). On the other 2, simpler dataset it was easier to achieve good results, requiring also, much fewer epochs to reach them. It is worth noting that the network does not contain local-response normalization layers, nor those were substituted by batch normalization ones. Furthermore, since this project was to study the architecture of the model and not the preprocessing, no data augmentation techniques were used. 

