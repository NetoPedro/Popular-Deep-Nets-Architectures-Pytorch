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

The authors start considering a mapping of a part of the network called H(x), then they hypothesize that it is possible to
 approximate the residual function H(x) - x. They then talk about explicitly approximate the residual function, with 
 F(x) = H(x) - x => F(x) + x = H(x). 
 
It is considered that despite the network being able to learn both forms, it makes it easier to learn.

So, how does this formulation tackles the degradation problem? If the optimal solution is in a extreme case, the identity mapping,
then instead of approximating the original H(x) to an identity mapping (which is difficult due to the non linear layers),
it just needs to push F(x), the non linear part, to 0. In non extreme cases, if the optimal solution is still closer to 
the identity mapping than to a zero mapping, it is easier to learn F(x) as perturbations in the identity mapping. 


#### Identity Mapping by Shortcuts

To be able to implement the residual training in several stacked layers, the authors defined a building block as: 

    y = F(x,{W_i}) + x

Where F(x,{W_i}) represents the residual mapping that needs to be learned. 


One of the main advantages of this approach is that it does not add any extra computational cost beside the addition operation
 which can be ignored. Furthermore, this allows to compare with other non residual (Plain) networks in a more straightforward manner. 

Although, there are cases where the residual mapping does not preserve the dimensions of the input. For this cases the 
authors defined: 

    y = F(x,{W_i}) + W_x * x
    
To be the function to learn, where W_s is used to linearly project the inputs to match the size of the residual 
mapping output.


![Residual Building Block](https://raw.githubusercontent.com/NetoPedro/Popular-Deep-Nets-Architectures-Pytorch/master/ResNet/images/residual_building_block.png) 


The paper experimented with two or three layers in the residual building block, with W_s being only used if dimension 
matching is necessary, thus proving that the identity mapping is sufficient to avoid the degradation problem. 


#### Bottleneck

![Residual Building Block](https://raw.githubusercontent.com/NetoPedro/Popular-Deep-Nets-Architectures-Pytorch/master/ResNet/images/bottleneck_residual.png) 

[Image Source](https://blobscdn.gitbook.com/v0/b/gitbook-28427.appspot.com/o/assets%2F-LRrOFNeUGLZef_2NLZ0%2F-LeEJi2MCK6d2wToNmIy%2F-LeEMR0uW50GA3ny07_C%2Fbottleneck.png?alt=media&token=9e6a700d-aa96-4381-8922-b544462ba101) 

For deeper networks the authors have defined the bottleneck building block, with the same complexity of the previous residual building block. The main change is that it now includes 3 convolutions, with 2 of them having a kernel size of 1x1, the idea of this 2 (1x1) convolutions is to decrease and then increase the dimensions of the input. This leads to having the 3x3 layer with smaller input size.

### Architecture 

In order to compare the residual training, the authors tested a plain and a residual network with the same configuration 
aside from the shortcut connections. It is also important to note that despite being deeper, it only has 18% of the computational
cost of the VGG-19. 

The configuration is the following 

    - Initial 7x7 convolution with 64 channels and stride 2
    - 2x2 max pooling with stride 2 
    - 6 3x3 convolutions with 64 channels 
    - 1 3x3 convolution with 128 channels and stride 2
    - 7 3x3 convolution with 128 channels
    - 1 3x3 convolution with 256 channels and stride 2
    - 11 3x3 convolution with 256 channels
    - 1 3x3 convolution with 512 channels and stride 2
    - 5 3x3 convolution with 512 channels
    - Average pooling layer 
    - Fully connected layer with 1000 (ImageNet classes) as output 

The network number of trainable layers is 34 (notice that it is much deeper than the VGG-19).

In this configuration the downsampling of the images happens in the convolutions with stride of 2, and on these layers 
the number of channels is doubled. Every convolutional layer is also followed by a batch normalization layer and a ReLU non linearity. 

The architecture is rather similar with the residual version of this network, being the only difference the identity 
shortcut connections that go after each 2 layers. If between the starting and the ending point of the shortcut there is 
a downsampling layer, it is necessary to downsample or add a padding of zeros to the identity mapping. The second is cost
 free. 


### Training and Learning 

Training hyperparameters: 

    - Stochastic Gradient Descent 
    - Mini-Batch size: 256
    - Initial Learning Rate: 0.1
    - Learning Rate Divided by 10 when error stops improving
    - 600000 iterations 
    - Weight Decay: 0.0001
    - Momentum: 0.9
    - Dropout is not used


#### Experiments 


Experiments with both networks either with the 34 layers configuration shown above, or a shallower 18 layers configuration, has shown that despite both networks having similar errors in the shallower configuration, when we increase the size, the plain network increases the error. On the other hand, not showing signs of degradation, the residual network lowers the validation error rate at a 34 layers configuration. The authors hypothesize that the convergence rates on the plain nets is exponentially low, making it extremely difficult to reduce the error.  

Further experiments were conducted with networks of sizes 50,101 and 152 (!!!) layers. 
To construct a 50 layers network, the 2 layers residual building blocks on the 34 layers network were replaced by 3 layers bottleneck building blocks with identity mapping. 

It has been shown that this 3 deeper architectures further improve the error when compared with the previous, hence it shows that this residual networks do not suffer from the degradation problem. 

## Implementation/Architecture Reproduction 


### Discussion 


