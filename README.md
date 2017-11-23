# Problem Sheet 4 - TensorFlow  

> Module: Emerging Technologies / 4th Year  
> Lecturer: Dr Ian McLoughlin  
> [Original Problem Sheet](https://github.com/emerging-technologies/emerging-technologies.github.io/blob/master/problems/tensorflow.md)  

This problem sheet relates to the python library [Tensorflow](https://www.tensorflow.org/) and the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris). Additional libraries used include [Keras](https://keras.io/), [SciKit](http://scikit-learn.org/stable/) and [NumPy](http://www.numpy.org/).  

The [IrisTF.ipynb](https://github.com/rebeccabernie/TensorFlowProblems/blob/master/IrisTF.ipynb) notebook uses TensorFlow to create a model and splits the data into training and testing sets. The model is then trained and tested using the relevant sets.  

## Instructions  
### 1. Use Tensorflow to create a model.  
*Use Tensorflow to create a model to predict the species of Iris from a flower’s sepal width, sepal length, petal width, and petal length.*  
I used SciKit's [dataset module](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) to get the Iris data, simply because it has an inbuilt `load_iris()` function to handle the Iris dataset. The function returns the sepal/petal data, as well as the target data (iris species) in two separate arrays.  

### 2. Split the data into training and testing sets.  
*Split the data set into a training set and a testing set. You should investigate the best way to do this, and list any online references used in your notebook. If you wish to, you can write some code to randomly separate the data on the fly.*  
I decided to use SciKit's [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function as it splits a given array or matrix into randomised train and test subsets in one operation. The test set is configured to be half of the total data set. This can be increased or decreased by changing the `test_size` parameter - a float between 0 and 1 indicates a percentage of the original set, any whole number indicates number of entries to be used. Each set consists of original sepal/petal data and the One Hot Encoded data, indicating the species of Iris. Below is an example of the first five rows in generated subsets -  

![twosets](https://user-images.githubusercontent.com/14957616/33153895-8d01241c-cfdc-11e7-9e5a-80e6dff3482a.JPG "Training Set and Test Set")  

#### One Hot Encoding  
One Hot Encoding is a method of transforming a classification or type based data set into a format more suitable for machine learning and neural networks. OHE takes the classification / type, in this case plant species, and creates a new matrix with a column for each species. Each row in the OHE set maps to an entry in the original data. OHE works in a  binary/boolean way - a 1 indicates true, a 0 indicates false. In this situation, if a particular entry was of species setosa, a 1 would appear in the first column and 0s in the following two columns, indicating not versicolor or virginica.  

### 3. Train the model.  
 *Use the training set to train your model.*  
 I used the `model.fit` method to train the model. This method takes in both training sets (the original data set and OHE set), a `verbose` parameter (progress output), a batch_size and an epoch limit. In this instance, I set a batch size of 25 and an epoch number of 500. Verbose was set to 0 (no output) for the sake of tidiness.

### 4. Test the model.  
 *Use the testing set to test your model, clearly calculating and displaying the error rate.*  
 Firstly, the program will make a species prediction for each element in the test set, outputting any incorrect predictions with the corresponding actual species.  
 The program then uses Keras' `model.evaluate()` function to calculate the loss and accuracy values. The function returns the results in an array of length 2, with the loss result stored in position `0`, and the accuracy stored in position `1` as a float between 0 and 1. This float is multiplied by 100 to get the percentage, and the error rate is calculated by taking this new percentage from 100. The program then outputs the loss, accuracy, and error rates. From my experience testing the program, the accuracy tends to fall between 94 and 98%.  
 See below for an example output.  

 ![results](https://user-images.githubusercontent.com/14957616/33153896-8d446ba0-cfdc-11e7-9579-e3f4de9a4174.JPG "Finished results and percentages")  

 #### Known Bugs
In some instances, the program appears to be unable to predict a species and returns `[0, 0, 0]` as a prediction and some runtime errors, pointing to the following line:  
`prediction = np.around(model.predict(np.expand_dims(test_x[i], axis=0))).astype(np.int)[0]`  
I suspected the problem was to do with rounding down to 0 instead of 1 and tried wrapping the line in the `math.ceil` function, to force the prediction to round up to 1. This did not work, as the math.ceil function kept producing compile errors regarding rounding an array element. I tried replacing `np.around()[0]` with `math.ceil` and removing the `axis=0` parameter, but this produced compile errors regarding expected array sizes.  
Failing to fix the issue, I decided the next best option was to simply output "Error" in place of a species when the prediction was [0, 0, 0]. While this didn't fix the error, at least the program now outputs without compile errors.
