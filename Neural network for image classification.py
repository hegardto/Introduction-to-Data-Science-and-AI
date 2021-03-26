# -*- coding: utf-8 -*-

# -- Sheet --

# Assignment 7: Neural Networks using Keras and Tensorflow Please see the associated document for questions
# 
# If you have problems with Keras and Tensorflow on your local installation please make sure they are updated. On Google Colab this notebook runs.


# ### Due to long computational times we were not able to re-run all cells in the assignment before handing it in.


# imports
from __future__ import print_function
import keras
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt

#Hyper-parameters data-loading and formatting
batch_size = 128
num_classes = 10
epochs = 10

#Definera antalet rader och kolumner i en bild
img_rows, img_cols = 28, 28

#Laddar 60,000 28x28 bilder på siffror som en tupels. Label 0-9 bereoende på vilken siffra det är
(x_train, lbl_train), (x_test, lbl_test) = mnist.load_data()

#Undersöker om channel (antalet dimensioner i färgern) dyker upp först eller sist och omformaterar datan utifrån detta, lägger till en dimension.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# **Preprocessing**


#Converting to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Scaling data to range (0,1)
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(lbl_train, num_classes)
y_test = keras.utils.to_categorical(lbl_test, num_classes)

#First model
model = Sequential()

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.SGD(lr = 0.1),
                        metrics=['accuracy'],)

fit_info = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=10,
                        verbose=1,
                        validation_data=(x_test, y_test))
model.summary()

#Create bar chart with train and val accuracies
train_accuracy = fit_info.history['accuracy']
test_accuracy = fit_info.history['val_accuracy']
df = pd.DataFrame(columns = ['train_accuracy'], index = range (1,epochs+1), data=train_accuracy)
df['test_accuracy'] = test_accuracy

plt = df.plot.bar(rot = 1, figsize = (12,10))
plt.grid(axis = 'y')
plt.set_ylim(0.8, 1)

fig = plt.get_figure()
fig.savefig("accuracy_plot.png")

print(df['test_accuracy'].max())

## Looping through different regularization factors and creating replicates to find the optimal parameter value (based on accuracy)
## New amount of units in layers: 300 & 500
from tensorflow.keras import regularizers
model = Sequential()
epochs = 40

regularization_factor = [0.000001,0.00025,0.00050,0.00075,0.0010]

max_accuracies = []

factor_count = 0
for factor in regularization_factor:
        
        for i in range (0,3):

                model.add(Flatten())
                model.add(Dense(500, activation = 'relu', kernel_regularizer=regularizers.l2(factor)))
                model.add(Dense(300, activation = 'relu'))
                model.add(Dense(num_classes, activation='softmax'))

                model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.SGD(lr = 0.1),
                        metrics=['accuracy'],)

                fit_info = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

                print(i)

                if i == 0:
                        max_accuracies.append([max(fit_info.history['val_accuracy'])])
                else: max_accuracies[factor_count].append(max(fit_info.history['val_accuracy']))

        factor_count += 1

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))

#Plotting the mean val accuracies and standard devs for the five factors 
import numpy as np

#Accuracies retrieved from previous cell (saved here due to unable to re-run previous cell in an OK amount of time)
max_accuracies = np.array([[0.9833999872207642, 0.9846000075340271, 0.98439712944031], [0.9845999848365784, 0.9846000143051147, 0.9837999939918518], [0.984400029438257, 0.98420002149021, 0.9845999173568726], [0.9839000105857849, 0.9836999773979187, 0.98389999904632568], [0.982199999592194, 0.9819999124124457, 0.982412489178255]])

val_accuracy = []
std = []

print(np.mean(max_accuracies[2]))

for factor in max_accuracies:
    val_accuracy.append(np.mean(factor))
    std.append(np.std(factor))
    
df1 = pd.DataFrame(columns = ['mean_val_accuracy'], index = regularization_factor, data=val_accuracy)
df2 = pd.DataFrame(columns = ['std'], index= regularization_factor, data=std)

plt1 = df1.plot.bar(rot = 1, figsize = (12,10), ylim = (0.97,0.99), grid = True, ylabel = "Mean validation accuracy", xlabel = 'L2 regularization factor')
plt2 = df2.plot.bar(rot = 1, figsize = (12,10), grid = True, ylabel = "Standard deviation", xlabel = 'L2 regularization factor')
#plt.grid(axis = 'y')
#plt.set_ylim(0.95,1)

fig = plt1.get_figure()
fig.savefig("mean_val_accuracy.png")

fig = plt1.get_figure()
fig.savefig("mean_std_accuracy.png")

# ## 3. Convulational layers


# New model with convolutional layers added
epochs = 40
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ### Question 4) Auto-Encoder for denoising


# salt and pepper method to add binomial random noise to image data
import numpy as np
noise_level = 0.5
def salt_and_pepper(input, noise_level=noise_level):
    """
    This applies salt and pepper noise to the input tensor - randomly setting bits to 1 or 0.
    Parameters
    ----------
    input : tensor
        The tensor to apply salt and pepper noise to.
    noise_level : float
        The amount of salt and pepper noise to add.
    Returns
    -------
    tensor
        Tensor with salt and pepper noise applied.
    """
    # salt and pepper noise
    a = np.random.binomial(size=input.shape, n=1, p=(1 - noise_level))
    b = np.random.binomial(size=input.shape, n=1, p=0.5)
    c = (a==0) * b
    return input * a + c

#data preparation
flattened_x_train = x_train.reshape(-1,784)
flattened_x_train_seasoned = salt_and_pepper(flattened_x_train, noise_level=noise_level)

flattened_x_test = x_test.reshape(-1,784)
flattened_x_test_seasoneed = salt_and_pepper(flattened_x_test, noise_level=noise_level)

# Creation of autoencoder and decoder models
latent_dim = 98

input_image = keras.Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_image)
encoded = Dense(latent_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_image, decoded)
encoder_only = keras.Model(input_image, encoded)

encoded_input = keras.Input(shape=(latent_dim,))
decoder_layer = Sequential(autoencoder.layers[-2:])
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

# training the autoencoder model
fit_info_AE = autoencoder.fit(flattened_x_train_seasoned, flattened_x_train,
                epochs=32,
                batch_size=64,
                shuffle=True,
                validation_data=(flattened_x_test_seasoneed, flattened_x_test))

# Plotting the images before and after encode/decode
num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(flattened_x_test_seasoneed.shape[0], size=num_images)

encoded_imgs = encoder_only.predict(flattened_x_test_seasoneed)
decoded_imgs = autoencoder.predict(flattened_x_test_seasoneed)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(flattened_x_test_seasoneed[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(14, 7))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.savefig("SnP=" +str(noise_level)+".png")
plt.show()

#Finding accuracies for different noise levels by combining the previous model with denoising
epochs = 32

noise_levels = [0.1,0.3,0.6,0.9]

accuracies = []

for noise_level in noise_levels:
    flattened_x_train = x_train.reshape(-1,784)
    flattened_x_train_seasoned = salt_and_pepper(flattened_x_train, noise_level=0.4)

    flattened_x_test = x_test.reshape(-1,784)
    flattened_x_test_seasoneed = salt_and_pepper(flattened_x_test, noise_level=0.4)

    latent_dim = 98

    input_image = keras.Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_image)
    encoded = Dense(latent_dim, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = keras.Model(input_image, decoded)
    encoder_only = keras.Model(input_image, encoded)

    encoded_input = keras.Input(shape=(latent_dim,))
    decoder_layer = Sequential(autoencoder.layers[-2:])
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    fit_info_AE = autoencoder.fit(flattened_x_train_seasoned, flattened_x_train,
                epochs=32,
                batch_size=64,
                shuffle=True,
                validation_data=(flattened_x_test_seasoneed, flattened_x_test))

    num_images = 10
    np.random.seed(42)
    random_test_images = np.random.randint(flattened_x_test_seasoneed.shape[0], size=num_images)

    encoded_imgs = encoder_only.predict(flattened_x_test_seasoneed)
    decoded_imgs = autoencoder.predict(flattened_x_test_seasoneed)

    plt.figure(figsize=(18, 4))

    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(flattened_x_test_seasoneed[image_idx].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encoded_imgs[image_idx].reshape(14, 7))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2*num_images + i + 1)
        plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.savefig("2:SnP=" + str(noise_level)+".png")
    plt.show()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(decoded_imgs.reshape(10000,28,28,1), y_test))
    score = model.evaluate(decoded_imgs.reshape(10000,28,28,1), y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    accuracies.append(score[1])

#Plotting the accuracies for the different noise levels
df = pd.DataFrame(columns = ['accuracies'], index = noise_levels, data=accuracies)

plt = df.plot.bar(rot = 1, figsize = (12,10), ylabel = "Accuracy", xlabel = 'Noise level')
plt.grid(axis = 'y')
plt.set_ylim(0.75, 0.85)

fig = plt.get_figure()
fig.savefig("noise_accuracy.png")

