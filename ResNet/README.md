# AlexNet 

## Paper 


### Introduction 

Deep Learning research have been showing that the performance of a network can be increased by increasing its depth. 
But it is not as simple as just adding more and more layers. Gradients are crucial to the learning process and with deeper
networks they usually explode or vanish. However, these problems have already been also solved, so it is possible to stack 
a greater number of layers.

Despite being able to stack more layers, the performance usually starts to degrade, it can be seen with a non improving
accuracy that after some iterations start to decay quickly. The authors also mention that this degradation does not happen 
due to being overfitted.

It is then claimed, that if we stack new identity layers to an already working model, it will result in the same result.
Hence, there is at least a deeper solution with equal training error, leading the authors to say that "Deeper models 
should produce no higher training error than its shallower counterpart".

The paper intents to solve this degradation problem by creating and alternative, easier to achieve solution when the 
identity mapping or something close to it is needed. 

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


### Discussion 

The network failed most predictions on the CIFAR100 dataset, nevertheless I do believe that if instead of considering Top-1 error, we considered Top-5 error rate it would be presenting good results (this assumption is made considering the drop in the cross-entropy loss, that may not be fully captured by the accuracy on the test set). On the other 2, simpler dataset it was easier to achieve good results, requiring also, much fewer epochs to reach them. It is worth noting that the network does not contain local-response normalization layers, nor those were substituted by batch normalization ones. Furthermore, since this project was to study the architecture of the model and not the preprocessing, no data augmentation techniques were used. 

