# -*- coding: utf-8 -*-

# -- 1 --

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Creating dataframe
df = pd.read_csv('data_assignment2.csv')
df

#Creating a new dataframe for each of the variables
Selling_price = df['Selling_price'].values
Living_area = df['Living_area'].values

#Compute linear regression for living_area and selling_price
LIVING_AREA = Living_area[:, np.newaxis]
model = LinearRegression().fit(LIVING_AREA, Selling_price)

#Plotting the linear regression line together with the individual data points
plt.figure(figsize=(20,10))
plt.scatter(Living_area, Selling_price)
xfit = np.linspace(0, 250, 1000)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit) 
plt.plot(xfit, yfit)
plt.title('House price vs Living Area')
plt.xlabel('Living area in m^2')
plt.ylabel('House price in MSEK')
plt.grid(color='black', linestyle='-', linewidth=0.1)
plt.savefig('House price vs Living Area.png', dpi=400)
plt.show()

#Calculate coefficient and intercept of the linear regression
(model.coef_,model.intercept_)

#Predicting the price of a 100m2 living area
model.predict([[100]])

#Predicting the price of a 150m2 living area
model.predict([[150]])

#Predicting the price of a 200m2 living area
model.predict([[200]])

from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot

#Create a new model for the and plot the residuals for the regression
model = Ridge()
visualizer = ResidualsPlot(model)

#Fit the training data to the visualizer
visualizer.fit(LIVING_AREA, Selling_price) 
visualizer.show()

#Creating a multiple variable regression
df1 = df[df['Land_size'].notna()]
df1 = df1[df1['Rooms'].notna()]
X = df1[['Living_area','Rooms','Land_size','Age']]
y = df1['Selling_price']
regr = LinearRegression()
regr.fit(X, y)

#The intercept and coefficients of the new model
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

#Plot the residuals for the model based on multiple variables
model = Ridge()
visualizer = ResidualsPlot(regr)

#Fit the training data to the visualizer
visualizer.fit(X, y)  
visualizer.show()

# -- 2a --

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

#Load the dataset and set X to the data and y to the labels
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

#Split the data into training set and test set. Split 25% with a random state.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#Compute confusion matrix
titles_options = [("k-NN Distance-weighted, Confusion matrix", None)]
for title, normalize in titles_options:
    plt.savefig('Confusion matrix', dpi=400)
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
                                 
    
    plt.grid(False) 
    disp.ax_.set_title(title)

plt.grid(False)
plt.show()

from sklearn import metrics

y_pred = classifier.predict(X_test) 

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=15)
scores.mean()

# -- 2b --

from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier

#Loading the dataset
X, y = datasets.load_iris(return_X_y=True)

#Separating training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 7)

#Creating a k-NN classifier with 14 neighbors and euclidean distance
knn = KNeighborsClassifier(n_neighbors=14, metric='euclidean')

knn.fit(X_train, y_train)

#Make predictions for the test dataset
y_pred = knn.predict(X_test)

#Proportion of correct predictions on the test set:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Cross-validating the accuracy of the model
scores = cross_val_score(knn, X, y, cv=15)
scores.mean()

import matplotlib.pyplot as plt

#Creating evenly spaced values in the interval 1-100
neighbors = np.arange(1, 100)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

#Iteratively creating a classifier with the different k's in 'neighbor' and adding the cross-validated accuracies to a list.
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    knn.fit(X_train, y_train)

    test_accuracy[i] = cross_val_score(knn, X, y, cv=15).mean()

#Plotting the different accuracies dependent on k's 
plt.title('Uniform k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.savefig('KNN.png', dpi=400)
plt.show()

#Create knn classifier weighted by distance, n=14
knnW = KNeighborsClassifier(n_neighbors=14, metric='euclidean', weights= 'distance')

#Train the model
knnW.fit(X_train, y_train)

#Predict y in the test data
y_pred = knnW.predict(X_test)

#Proportion of correct predictions on the test set:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Cross validate the accuracy of the model 
scores = cross_val_score(knnW, X, y, cv=15)
scores.mean()

import matplotlib.pyplot as plt

#Creating evenly spaced values in the interval 1-100
neighbors = np.arange(1, 100)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
#Iteratively creating a classifier with the different k's in 'neighbor' and adding the cross-validated accuracies to a list.
for i, k in enumerate(neighbors):
    knnW = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights = 'distance')

    knnW.fit(X_train, y_train)

    test_accuracy[i] = cross_val_score(knnW, X, y, cv=15).mean()
    
#Plotting the different accuracies dependent on k's 
plt.title('Distance-weighted k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.savefig('KNNW.png', dpi=400)
plt.show()

# -- 2c --

#Load dataset and set X to data and y to target
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
#Split into training data and test data. 75%=training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

#k-nn classifier with n=7 uniform distance
knn2c = KNeighborsClassifier(n_neighbors=14, metric='euclidean')

#Train the model
knn2c.fit(X_train, y_train)

#Compute confusion matrix k-NN uniform
titles_options = [("k-NN Uniform, Confusion matrix", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
                                 
    
    plt.grid(False) 
    disp.ax_.set_title(title)

plt.grid(False)
plt.savefig('UKNNCM.png', dpi=400)
plt.show()


#k-nn classifier with n=7 uniform distance

knnW2c = KNeighborsClassifier(n_neighbors=14, metric='euclidean', weights='distance')

#Train the model
knnW2c.fit(X_train, y_train)

#Compute confusion matrix k-NN weighted
titles_options = [("k-NN Distance-weighted, Confusion matrix", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
                                 
    
    plt.grid(False) 
    disp.ax_.set_title(title)

plt.grid(False)
plt.savefig('DKNNCM.png', dpi=400)
plt.show()

