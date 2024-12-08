# Handwritten Digit Recognition
# Importing libraries
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Fetching the mnist dataset
mnist = fetch_openml('mnist_784')

#specifying the predictor and target varibles as x, y
x, y = mnist['data'], mnist['target']

#taking any digit from the dataset and storing it in the variable some_digit
some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  #reshaping the size of image to plot it
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')                   #plotting the image of  the given digit 
plt.axis("off")
plt.show()

#splitting the dataset into training and testing
x_train, x_test = x[0:60000], x[60000:]
y_train, y_test = y[0:60000], y[60000:]

#Shuffling the training dataset
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

# Train a logistic regression classifier
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train_2)
example = clf.predict([some_digit])
print(example)

# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(f"The accuracy obtained from the above model is :{a.mean()}")
