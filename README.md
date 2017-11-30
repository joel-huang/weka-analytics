# weka-analytics

Implementation of weka k-NN and SVM classifiers in Java.

No Android, no UDOO.

# The k-Nearest Neighbours algorithm for classification

* An object is classified by a majority vote of its k nearest neighbours.
* Distance metrics: Euclidean distance (continuous), Hamming distance (discrete).
* Larger k values reduce noise, but make boundaries between classes less distinct.
* Pros: k-NN is 'lazy learning', where the target function is approximated locally. This allows faster training on large datasets.
* Cons: If one of the classes has a abnormally large frequency, it will dominate the prediction of the new example. One way to remedy this is to multiply the inverse of the distance from each neighbour with a weight.


* For evaluation, k-fold cross validation: Splitting the input data into k equal sized sets (folds). 
* For example, given 200 labelled data, and k=10, it is split into 10 folds of 20 data. Pick one of those folds (e.g. k0) as a testing set, then use the rest (k1, k2, ... k9) as a training set. Then pick (k1) as a testing set, and use (k0, k2, ... k9) as a training set. Repeat until all are used, and average the ten different performance values.
* You've used all of your data for testing, and all of your data for training!