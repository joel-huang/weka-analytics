# weka-analytics: Supervised Learning Basics

Implementation of weka k-NN and SVM classifiers in Java.

# Training and Testing in Classification

* Training: A training dataset is used as a set of examples, each example containing input vectors that live in the feature space, and a target vector or scalar (attributes). The relation between the data points and attributes are learned or fitted (sometimes overfitted).
* Testing: The model is tested using the testing dataset in order to determine the accuracy of the model. The model outputs predicted attributes from the testing dataset and we compare these predictions with the actual attributes of the training dataset.
* Choosing informative, discriminating and independent features is a crucial step for effective algorithms in pattern recognition, classification and regression. 

# The k-Nearest Neighbours algorithm for classification

* An object is classified by a majority vote of its k nearest neighbours.
* Distance metrics: Euclidean distance (continuous), Hamming distance (discrete).
* Larger k values reduce noise, but make boundaries between classes less distinct.
* Pros: k-NN is 'lazy learning', where the target function is approximated locally. This allows faster training on large datasets.
* Cons: If one of the classes has a abnormally large frequency, it will dominate the prediction of the new example. One way to remedy this is to multiply the inverse of the distance from each neighbour with a weight.
* For evaluation, k-fold cross validation: Splitting the input data into k equal sized sets (folds). 
* For example, given 200 labelled data, and k=10, it is split into 10 folds of 20 data. Pick one of those folds (e.g. k0) as a testing set, then use the rest (k1, k2, ... k9) as a training set. Then pick (k1) as a testing set, and use (k0, k2, ... k9) as a training set. Repeat until all are used, and average the ten different performance values.
* You've used all of your data for testing, and all of your data for training!

# Support Vector Machines (SVMs)
* An SVM constructs one or more lines or hyperplane(s) in a high- or infinite-dimensional space, which can be used for classification, regression, etc.
* The boundary achieves maximum separation between two points, one from each class, called the support vectors. The boundary is the line bisecting the line between two support vectors.
* If the classes are not linearly separable (a linear line cannot divide them) use the Kernel trick to obtain non-linear boundaries.
* Pros: Resilient to overfitting (Boundaries only depend on support vectors)
