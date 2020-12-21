import idx2numpy
from matplotlib import pyplot as plt
import numpy as np

# using idx2numpy to import the MNIST dataset
train_image = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_label = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
test_image = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_label = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# print(train_image.shape)
# print(train_label.shape)
# print(test_image.shape)
# print(test_label.shape)

# set training size
trainsize = 500

matrix = []
for i in range(0, trainsize):
    matrix.append(train_image[i].flatten())

matrix = np.array(matrix).T

print(matrix.shape)  # should be (784, trainsize)


# print(matrix)

# take an (x by k) matrix A (where x is the total number of pixels in an image
# and k is the number of training images) and return a vector m of length x
# containing the mean column vector of A and an (x by k) matrix V that
# contains k eigenvectors of the covariance matrix of A
# (after the mean has been subtracted). These should be sorted in descending
# order by eigenvalue (i.e., V(:,1) is the eigenvector with the largest associated
# eigenvalue) and normalized (i.e., norm(V(:,1)) = 1). You can reshape a vector
# and display it as an image.
def hw1FindEigendigits(Mat):
    mean = np.array(np.mean(Mat, axis=1), ndmin=2).T

    mean_Matrix = np.broadcast_to(mean, (784, trainsize))
    A = (Mat - mean_Matrix) / np.sqrt(trainsize)  # A should be a (784,trainsize) matrix

    if trainsize < 784:
        # compute A^tA
        small_Mat = np.matmul(A.T, A)
        # find the eigenvalues and eigenvectors of A^tA
        mu, v = np.linalg.eig(small_Mat)
    else:
        small_Mat = np.matmul(A, A.T)
        mu, v = np.linalg.eig(small_Mat)

    # print(mu)
    print(v[0].shape)  # should be (trainsize,)
    # print(v[0])

    # sort by eigenvalues in decreasing order
    ls = []
    for i in range(min(trainsize, 784)):
        ls.append((mu[i], v[i]))
    ls = sorted(ls, key=lambda ls: ls[0], reverse=True)
    # print(ls)

    # print(np.matmul(A,ls[0][1]).shape)  # should be (784,)

    V = []
    for i in range(min(trainsize, 784)):
        if trainsize < 784:
            vec = np.matmul(A, ls[i][1])
            norm = np.linalg.norm(vec)
            vec = vec / norm
            # print(np.linalg.norm(vec))  # check the norm, should be 1
            V.append(vec)
        else:
            V.append(ls[i][1])

    V = np.array(V).T
    print(V.shape)  # should be (784,trainsize)

    return mean, V


mean, V = hw1FindEigendigits(matrix)
# print(mean.shape)  # should be (784,1)
# print(mean)

# # code for displaying the first eight eigendigits
# for i in range(8):
#     plt.subplot(2, 4, i + 1)
#     img = (V[:, i]).reshape(28, 28)
#     plt.title('#{}'.format(i + 1))
#     plt.imshow(img)
# plt.show()

# print(V.shape)  # should be (784,8)

# create eigenvector space
Omega = []
for i in range(trainsize):
    Omega_i = []
    X_i = matrix[:, i] - mean[:, 0]
    for j in range(min(trainsize, 784)):
        Omega_i.append(np.dot(X_i, V[:, j]))
    Omega.append(Omega_i)

Omega = np.array(Omega)
print(Omega.shape)


# Define our prediction function, which uses KNN to make predictions
def predict(testsize, k, eigen):
    predict_label = []
    for i in range(testsize):
        # Form the eigenvector space of the predicting image
        Omega_test = []
        Y_i = test_image[i].flatten() - mean[:, 0]
        for j in range(min(trainsize, 784, eigen)):
            Omega_test.append(np.dot(Y_i, V[:, j]))

        # reconstruct some test images with Matrix V and the Omega_test above
        # if i < 3 and trainsize < 600:
        #     recon_img = np.matmul(V, Omega_test).reshape(28, 28)
        #     print(recon_img.shape)
        #     plt.imshow(recon_img)
        #     plt.show()

        # Calculate L2-norm distances
        distance = []
        for j in range(min(trainsize,eigen)):
            L2 = np.linalg.norm(np.array(Omega_test) - Omega[j, 0:eigen])
            distance.append((L2, train_label[j]))
        # print(distance)
        # Sort the distance in increasing order
        distance = sorted(distance, key=lambda distance: distance[0])
        # print(distance)

        # Using KNN to choose the k nearest ones and pick the label that appears the most
        KNN_arr = []
        for i in range(k):
            KNN_arr.append(distance[i][1])
        # print(KNN_arr)  # should only contain the k-nearest labels

        # create "bins" to count the occurrences of different labels and create "order"
        # to count the appearing order of them.
        bin = np.zeros((10,), dtype=int)
        order = bin + 10
        count = 0
        for i in range(k):
            bin[KNN_arr[i]] += 1
            if order[KNN_arr[i]] == 10:
                order[KNN_arr[i]] = count
                count += 1

        # print(bin, order*(-0.1))

        # We subtract the bin count by 0.1 times the appearing order,
        # so that the digit with highest counts and appears first will have
        # the largest value.
        # We use 0.1 here so that the digits appear n times will always
        # have higher "rank" values than those appear (n-1) or fewer times.
        rank = bin - 0.1 * order
        # print(rank)

        # using argmax to find the index of the largest value, which becomes
        # the digit we predict, and assign it to label
        label = np.argmax(rank)
        predict_label.append(label)

    return predict_label


# set our testsize and k (number of nearest neighbors) to make predictions
testsize = 1000
k = 4

# # display the first few actual test labels
# if trainsize < 600:
#     for i in range(3):
#         plt.imshow(test_image[i])
#         plt.show()

# generate predicting labels

# set the number of eigenvectors we consider, the default value is trainsize,
# namely we use all of the eigenvectors
eigen = trainsize

# generate predicting labels
predict_label = np.array(predict(testsize, k, eigen))
print("Our Predictions: {}".format(predict_label))

# assign actual labels to test_ans
test_ans = test_label[0:testsize]
print("Actual Labels:   {}".format(test_ans))

# Using "==" and sum to compute the number of correct predictions
right_ones = sum(predict_label == test_ans)
# Calculate accuracy
accuracy = right_ones / testsize
# Print accuracy with corresponding trainsize, testsize, and number of eigenvectors
print('Accuracy ({} trainsize, {} testsize, {} eigenvectors): {}'.format(trainsize, testsize, eigen, accuracy))
