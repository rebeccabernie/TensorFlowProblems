# Problem Sheet 4 - TensorFlow  

> Module: Emerging Technologies / 4th Year  
> Lecturer: Dr Ian McLoughlin  
> [Original Problem Sheet](https://github.com/emerging-technologies/emerging-technologies.github.io/blob/master/problems/tensorflow.md)  

This problem sheet relates to the python library [Tensorflow](https://www.tensorflow.org/) and the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris). Additional libraries used include [Keras](https://keras.io/), [SciKit](http://scikit-learn.org/stable/) and [NumPy](http://www.numpy.org/).  

### Instructions  
1. Use Tensorflow to create a model to predict the species of Iris from a flower’s sepal width, sepal length, petal width, and petal length.  
I used SciKit's [dataset module](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) to get the Iris data, simply because it has an inbuilt `load_iris()` function to handle the Iris dataset. The function returns the sepal/petal data, as well as the target data (iris species) in two separate arrays.  

2. Split the data set into a training set and a testing set. You should investigate the best way to do this, and list any online references used in your notebook. If you wish to, you can write some code to randomly separate the data on the fly.  
I decided to use SciKit's [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function as it splits a given array or matrix into randomised train and test subsets in one operation. The test set is configured to be half of the total data set. This can be increased or decreased by changing the `test_size` parameter - a float between 0 and 1 indicates a percentage of the original set, any whole number indicates number of entries to be used. Each set consists of original sepal/petal data and the One Hot Encoded data, indicating the species of Iris. Below is an example of the first five rows in generated subsets -  

![twosets](https://user-images.githubusercontent.com/14957616/33153895-8d01241c-cfdc-11e7-9e5a-80e6dff3482a.JPG "Training Set and Test Set")  

**One Hot Encoding**  
One Hot Encoding is a method of transforming a classification or type based data set into a format more suitable for machine learning and neural networks. OHE takes the classification / type, in this case plant species, and creates a new matrix with a column for each species. Each row in the OHE set maps to an entry in the original data. OHE works in a  binary/boolean way - a 1 indicates true, a 0 indicates false. In this situation, if a particular entry was of species setosa, a 1 would appear in the first column and 0s in the following two columns, indicating not versicolor or virginica.  

3. Use the training set to train your model.

4. Use the testing set to test your model, clearly calculating and displaying the error rate.

The [IrisTF.ipynb](https://github.com/rebeccabernie/TensorFlowProblems/blob/master/IrisTF.ipynb) notebook uses TensorFlow to create a model and splits the data into training and testing sets. The model is then trained and tested using the relevant sets.  