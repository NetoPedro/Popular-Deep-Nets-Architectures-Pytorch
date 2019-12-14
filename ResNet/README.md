# ResNet 

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
identity mapping or something close to it is needed. It is then stated that not only the presented solution can increase 
the depth while still increasing the accuracy, but it is also the deepest network presented until that moment, despite 
having lower complexity than shallower networks. 

### Deep Residual Learning 

The paper presented a solution to deep networks throughout residual building blocks, this section describes in detail 
how these blocks work. 

#### Residual Learning 

The authors start considering a mapping of a part of the network called H(x), then they state that it is possible to
 approximate the residual function H(x) - x. Thus they choose to explicitly approximate the residual function, with 
 F(x) = H(x) - x => F(x) + x = H(x). 
 
It is considered that despite the network being able to learn both forms, it makes it easier to learn.

So, how does this formulation tackles the degradation problem? If the optimal solution is in a extreme case, the identity mapping,
then instead of approximating the original H(x) to an identity mapping (which is difficult due to the non linear layers),
it just needs to push F(x), the non linear part, to 0. In non extreme cases, if the optimal solution is still closer to 
the identity mapping than to a zero mapping, it is easier to learn F(x) as perturbations in the identity mapping. 


#### Identity Mapping by Shortcuts


### Architecture 

#### Residual Block

#### Bottleneck

#### Overall review of the architecture 



### Training and Learning 

Details of training and learning are further explained in the implementation section.


## Implementation/Architecture Reproduction 


### Discussion 


