# PCA-KNN-for-mnist-classification
This project uses principal component analysis to compute eigenvalues and eigendigits, and then uses k-nearest neighbors to perform classification on testing dataset. The PCA and KNN algorithm are constructed from scratch using NumPy to allow more flexibility and individualization. Specifically, when the training size is less than 784 (28\*28), we can use the tranpose tricks in matrix algebra so that we do not need to compute all the 784 eigenvalues.
## Files
- `main.py` is the main python program for the task
- `extended.py` is the extended python program which has slight modifications inside so that we can also handle training sizes larger than 784
