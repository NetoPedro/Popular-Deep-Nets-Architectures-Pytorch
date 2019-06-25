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
previous research papers on new activations like ReLU, regularization methods like Dropout and the presented 
implementation of efficient convolutions on a GPU were positively significant to the architecture described here. 


### Dataset 

ImageNet was one of the introductory datasets to gather millions of examples, with over 15 millions of images 
distributed across 22 thousand categories. The existence of such a large dataset was one key element to the emergence of 
deep learning and the popularity of neural networks as models with a significant capability to learn complex functions. 
The dataset is in its 2010 version already split between test, validation and train data. 

In the dataset, to evaluate each model, two errors are measured. The first, top-1, refers to situations where the 
correct label is the label with the highest probability accordingly to the model. The second, top-5, refers to 
situations where the true label is in the five predictions with larger probabilities.

All the images are further scaled to be possible to use them as the input of the same neural network. 
The chosen scale is 256x256. 
