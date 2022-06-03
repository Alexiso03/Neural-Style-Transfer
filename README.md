# Neural-Style-Transfer

## This repository contains Deep Learning & Art mixture with the help of Neural Style Transfer.

1. Implemented the neural style transfer algorithm
2. Generated novel artistic images using your algorithm
3. Defined the style cost function for Neural Style Transfer
4. Defined the content cost function for Neural Style Transfer

<a name='2'></a>
## Problem Statement

Neural Style Transfer (NST) is one of the most fun and interesting optimization techniques in deep learning. It merges two images, namely: a <strong>"content" image (C)</strong> and a <strong>"style" image (S)</strong>, to create a <strong>"generated" image (G)</strong>. The generated image G combines the "content" of the image C with the "style" of image S. 

In this project, we are going to combine the Louvre museum in Paris (content image C) with the impressionist style of Claude Monet (content image S) to generate the following image:

![image](https://user-images.githubusercontent.com/86974424/171830538-b728903e-25f1-4f68-bd76-9d1e085efe88.png)

## Transfer Learning

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning. 

I will be using the the epynomously named VGG network from the [original NST paper](https://arxiv.org/abs/1508.06576) published by the Visual Geometry Group at University of Oxford in 2014. Specifically, you'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers). 

#### Make Generated Image G Match the Content of Image C

One goal you should aim for when performing NST is for the content in generated image G to match the content of image C. To do so, you'll need an understanding of <b> shallow versus deep layers </b>:

* The shallower layers of a ConvNet tend to detect lower-level features such as <i>edges and simple textures</i>.
* The deeper layers tend to detect higher-level features such as more <i> complex textures and object classes</i>. 

#### To choose a "middle" activation layer:
You need the "generated" image G to have similar content as the input image C. Suppose you have chosen some layer's activations to represent the content of an image. 
* In practice, you'll get the most visually pleasing results if you choose a layer in the <b>middle</b> of the network--neither too shallow nor too deep. This ensures that the network detects both higher-level and lower-level features.
* After you have finished this exercise, feel free to come back and experiment with using different layers to see how the results vary!

#### To forward propagate image "C:"
* Set the image C as the input to the pretrained VGG network, and run forward propagation.  
* Let $a^{(C)}$ be the hidden layer activations in the layer you had chosen. (In lecture, this was written as $a^{[l](C)}$, but here the superscript $[l]$ is dropped to simplify the notation.) This will be an $n_H \times n_W \times n_C$ tensor.

#### To forward propagate image "G":
* Repeat this process with the image G: Set G as the input, and run forward progation. 
* Let $a^{(G)}$ be the corresponding hidden layer activation. 

The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.

![image](https://user-images.githubusercontent.com/86974424/171831111-cf50bd51-dadd-491d-a047-c19dbd764ca6.png)

Content Cost Function  𝐽𝑐𝑜𝑛𝑡𝑒𝑛𝑡(𝐶,𝐺) 
One goal you should aim for when performing NST is for the content in generated image G to match the content of image C. A method to achieve this is to calculate the content cost function, which will be defined as:

### 𝐽𝑐𝑜𝑛𝑡𝑒𝑛𝑡(𝐶,𝐺)=14×𝑛𝐻×𝑛𝑊×𝑛𝐶∑all entries(𝑎(𝐶)−𝑎(𝐺))2(1)

Here,  𝑛𝐻,𝑛𝑊  and  𝑛𝐶  are the height, width and number of channels of the hidden layer you have chosen, and appear in a normalization term in the cost.
For clarity, note that  𝑎(𝐶)  and  𝑎(𝐺)  are the 3D volumes corresponding to a hidden layer's activations.
In order to compute the cost  𝐽𝑐𝑜𝑛𝑡𝑒𝑛𝑡(𝐶,𝐺) , it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below.
Technically this unrolling step isn't needed to compute  𝐽𝑐𝑜𝑛𝑡𝑒𝑛𝑡 , but it will be good practice for when you do need to carry out a similar operation later for computing the style cost  𝐽𝑠𝑡𝑦𝑙𝑒 .

![image](https://user-images.githubusercontent.com/86974424/171831248-2a0bdf5f-2b5b-4e4f-a481-a28e36af781d.png)

Style image is shown below:

![image](https://user-images.githubusercontent.com/86974424/171831375-04fa12f3-b887-4b74-9a96-eeaee3852a97.png)

This was painted in the style of impressionism.

Style Matrix:
Gram matrix:
The style matrix is also called a "Gram matrix."
In linear algebra, the Gram matrix G of a set of vectors  (𝑣1,…,𝑣𝑛)  is the matrix of dot products, whose entries are  𝐺𝑖𝑗=𝑣𝑇𝑖𝑣𝑗=𝑛𝑝.𝑑𝑜𝑡(𝑣𝑖,𝑣𝑗) .
In other words,  𝐺𝑖𝑗  compares how similar  𝑣𝑖  is to  𝑣𝑗 : If they are highly similar, you would expect them to have a large dot product, and thus for  𝐺𝑖𝑗  to be large.

![image](https://user-images.githubusercontent.com/86974424/171831880-54b772eb-a039-478c-90af-70bc5cdd0106.png)

# Next Computations are:
1. compute_layer_style_cost
2. compute_style_cost
3. Defining the Total Cost to Optimize
4. Solving the Optimization Problem
5. Randomly Initialize the Image to be Generated
6. Load Pre-trained VGG19 Model
7. Compute the Content image Encoding (a_C)
8. Compute the Style image Encoding (a_S)
9. train_step
10. Use the Adam optimizer to minimize the total cost J
11. Train the Model
12. Testing With the image loaded.


# The end result is obtained as:
![image](https://user-images.githubusercontent.com/86974424/171831567-9e8985bf-ed42-4291-b760-ae33992f0cde.png)

