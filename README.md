# 2-D data analysis with Neural-Network.
This is a Two Dimensional data which has 100 2-D data points from each class[Two Classes]. The plot of the data is as follows which gives a clear idea about the decision boundary required for classification. This data is generated using scikit learn datasets [make_circles].

</br>
<p align="center">
  <img src="/Plots/Two-Dimensional-data.png" alt="One dimensional data with two classes" height="300" width="400" />
</p>

# Network Details
If we use a neural network with 2 layers (input layer is not counted) i.e. Hidden layer and Output layer and where hidden layer has only 2 units then we will not be able to classify this data. This is because the data in 2-D space needs a non-linear boundary and with our configuration of neural network, where output layer has 1 unit and hidden layer has 2 unit, will not be able to create a non-linear boundary. 

This data in 2 dimensional space is not linearly separable so in this case we will need at least 3 units in the hidden layer. These 3 units in the hidden layer will project our 2-Dimensional input into 3-D space where the data will be linearly separable and after which our neural network will start classifying the data without any problem.

For the analysis we will just concentrate on the three output from the three hidden units, weights and bias vectors of the 2nd layer. We are just simply using a neural network show below in the image.

### Mathematical formula for this neural net
For one particular data sample *x*<sup>(*i*)</sup>:
</br>
*z*<sup>*[1] (i)*</sup> =  *W*<sup>*[1]*</sup> *x*<sup>*(i)*</sup> + *b*<sup>*[1] (i)*</sup>
</br>
*a*<sup>*[1] (i)*</sup> = *tanh*(*z*<sup>*[1] (i)*</sup>)
</br>
*z*<sup>*[2] (i)*</sup> = *W*<sup>*[2]*</sup> *a*<sup>*[1] (i)*</sup> + *b*<sup>*[2] (i)*</sup>
</br>
*y*<sup>*(i)*</sup> = *a*<sup>*[2] (i)*</sup> = σ(*z*<sup>*[2] (i)*</sup>)
</br>

<p align="center">
  <img src="/Plots/2d-NN.PNG" alt="One dimensional data with two classes" height="200" width="300" />
</p>
