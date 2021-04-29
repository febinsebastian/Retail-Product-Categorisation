# Retail-Product-Categorisation
This notebook gives an introduction to the data and a first baseline solution.
### Context

In the past 10 years, the applications of artificial neural networks have developed outstandingly from image segmentation to speech recognition. One notably successful application of deep learning is deep embedding, a technique to transform the input data into a more useful representation, a list of real numbers called vectors. During the training on a supervised machine learning prediction task, the parameters of the neural network – the weights – are the embeddings that will modify to minimize the loss. These resulting embedding vectors have closer representation in the embedding space for the inputs from a similar category. Deep embedding is widely used to make recommendations by finding the nearest neighbors in the embedding space.

###### **Problem Description**
Every eCommerce application or retailer has millions of products. Identifying similar products can be used for recommendation and search. Our task is to build a product recommendation system from the product image and text description.
You can find more information about the dataset [in our Arxiv paper](https://arxiv.org/abs/2103.13864).
You can download the dataset from [kaggle](https://www.kaggle.com/c/retail-products-classification).

###### **Our Approach**
We build a deep learning model by concatenating a Convolutional Neural Network (ConvNet) and Long Short-Term Networks (LSTM). ConvNet will classify the product images and the LSTM network with an embedding layer will classify the description text. We have used a supervised learning process to train the model by labeling the samples by their category. The neural network models learn while training by a feedback process called backpropagation. This involves comparing the output produced by the network with the actual output and using the difference between them to modify the weights of the connections between the units in the neural network. We assume that once the model successfully trained the output or features from the final layer are embedded. Being more specific, the output layer will generate a similar dense vector for the samples from similar categories. We can create the embedding space by using these vectors and similarities.

### Content

This dataset consists of more than 42000 product information such as images and short descriptions belonging to 21 categories. 
The train folder contains all the product images of size 100X100 used to train the model.
The test folder contains images to evaluate model performance.
data.csv file contains the following information:
- ImgId: unique id of the product. Also, all the images are saved by this name.
- title: Name of the product
- description: a short description of the product
- category: name of the category the product belongs to

### Acknowledgements

We would like to thank Jianmo Ni, UCSD for their large public [dataset](https://nijianmo.github.io/amazon/index.html) created from amazon.
