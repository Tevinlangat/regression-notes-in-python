#  Intro to Regression with Neural Networks in TensorFlow

# There are many definitions for a regression problem but in our case, we can siplify it by saying: predicting a numerical variable based on some other combination of variables or.... predicting a number

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

### Creating data to view and fit

# Create features
X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])

# Create labels
y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

# Visualize it
plt.scatter(X,y)



## Input and Output shape

# Create a demo tensor for our housing price prediction problem
house_info = tf.constant(['bedroom','bathroom','garage'])
house_price = tf.constant([939700])
house_info, house_price

input_shape = X.shape
output_shape = y.shape
input_shape,output_shape

X= tf.constant(X)
y= tf.constant(y)
X,y

## Steps in modelling with tensorfolw

# 1. **Creating a model** - define the input and output layers, as well as the hidden layers of a deep learning model.
# 2. **Compiling a model** - define the loss function (in other words, the function that tells our model how wrong it is) and the optimizers (tells our model how to improve the patterns its learning) and evaluation metrics(what we can use to interprete the performance of our model).
# 3. **Fitting a model** - letting the model try to find patterns between X & y (features and labels)

# Set random seed
tf.random.set_seed(42)

#1. Create a model using the sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#2. Compile the model
model.compile(
    loss=tf.keras.losses.mae, # mae is short for mean absolute error
    optimizer = tf.keras.optimizers.SGD(), # sgd is short for stochastic gradient descent
    metrics= ['mae']
)

#3. Fit the model
model.fit(tf.expand_dims(X,axis=-1),tf.expand_dims(y,axis=-1), epochs=5)

model.predict([17])

## Improving the model

# We can improve a model by altering the steps we took to create a mode:
# 1. **Creating a model** - here we might add more layers, increase the number of hidden units (or called neurons) within each of the hidden layers, change the activation function of each layer
# 2. **Compiling a model** - here we might change the optimization function or perhaps the **Learning Rate** of the optimization function.
# 3. **Fitting a model** - here we might fit a model for more **epochs** (leave it to train for longer) or more data

#1. Create a model using the sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#2. Compile the model
model.compile(
    loss=tf.keras.losses.mae, # mae is short for mean absolute error
    optimizer = tf.keras.optimizers.SGD(), # sgd is short for stochastic gradient descent
    metrics= ['mae']
)

#3. Fit the model (this time we will train for longer)
model.fit(tf.expand_dims(X,axis=-1),tf.expand_dims(y,axis=-1), epochs=100)

model.predict([17])

#1. Create a model using the sequential API (this time with an extra hidden layer with 100 hidden units)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

#2. Compile the model
model.compile(
    loss=tf.keras.losses.mae, # mae is short for mean absolute error
    optimizer = tf.keras.optimizers.SGD(), # sgd is short for stochastic gradient descent
    metrics= ['mae']
)

#3. Fit the model
model.fit(tf.expand_dims(X,axis=-1),tf.expand_dims(y,axis=-1), epochs=100)

model.predict([17])

# Here the model did worse compared to the previous even though we added another layer, this could be over fitting

# **Parameters** - these are parameters the model learn by it\`s self and we cannot change

# **Hyper-parameters** - these are parameters which we can tweak to improve our model

# **NB: The model does not neccessarily do well by increasing hyperparameters to improve the model sometimes decreasing them would help ie tweak to suit your model**


## Common ways to improve a deep model:

# 1. Adding layers
# 2. Increase the number of hidden units
# 3. Change the activation function
# 4. Change the optimaization function
# 5. Change the learning rate
# 6. Fitting more data
# 7. Fitting for longer

# **Tip: the Learning rate is one of the most important parameters to change 😜✌️**

#1. Create a model using the sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

#2. Compile the model
model.compile(
    loss=tf.keras.losses.mae, # mae is short for mean absolute error
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), # sgd is short for stochastic gradient descent
    metrics= ['mae']
)

#3. Fit the model
model.fit(tf.expand_dims(X,axis=-1),tf.expand_dims(y,axis=-1), epochs=100)

## Evaluating a model

# In practise, a typical workflow you will go through when building neural networks is:

# Build a mode => fit it => evaluate it => tweak a model => fit it => evaluate it => tweak a model => fit it => evaluate it ...

# Make a bigger data set
X = tf.range(-100,100,4)
X

# Make labels for the data set
y = X + 10
y

#visualize the data
import matplotlib.pyplot as plt
plt.scatter(X,y)

### The 3 sets

# 1. Training set - The model learns from this data, typically 70 - 80%
# 2. Validation set - the model gets tunned on this data, typically 10 -15%
# 3. Test set - the model gets evaluated on this set typically 10 -15%

len(X)

# Split the data into train and test set
X_train= X[:40]
y_train= y[:40]

X_test= X[40:]
y_test= y[40:]

# Visualize the data
plt.figure(figsize = (10,7))

#Plot training data in blue
plt.scatter(X_train, y_train, c='b', label = 'Training data')

#Plot test data in green
plt.scatter(X_test, y_test, c='g', label = 'Testing data')

#Show legend
plt.legend()

#1. Create a model using the sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#2. Compile the model
model.compile(
    loss=tf.keras.losses.mae, # mae is short for mean absolute error
    optimizer = tf.keras.optimizers.SGD(), # sgd is short for stochastic gradient descent
    metrics= ['mae']
)

# #3. Fit the model
# model.fit(tf.expand_dims(X_train,axis=-1),tf.expand_dims(y_train,axis=-1), epochs=100)

### Visualize the model

#Creating a model which builds automatically by defining the input_shape argument
tf.random.set_seed(42)

# Create a model
#1. Create a model using the sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name = 'input_layer'),
    tf.keras.layers.Dense(1, name= 'output_layer')
], name='model_1')

#2. Compile the model
model.compile(
    loss=tf.keras.losses.mae, # mae is short for mean absolute error
    optimizer = tf.keras.optimizers.SGD(), # sgd is short for stochastic gradient descent
    metrics= ['mae']
)

model.summary()

#3. Fit the model
model.fit(tf.expand_dims(X_train,axis=-1),tf.expand_dims(y_train,axis=-1), epochs=100)

from tensorflow.keras.utils import plot_model
plot_model(model=model, show_shapes=True)

## Visualizing our model`s predictions

# To visualize predictions, it`s a good idea to plot them against the ground truth labels.

# Often you\`ll see this in the form of `y_test` or `y_true` versus `y_pred` (ground truth versus your model`s predictions).

# Make some predictions
y_pred = model.predict(X_test)
y_pred

# 🛎️ **Note:** if you feel like you are going to reuse some kind of functionality in the future, it`s a good idea to turn it into a function

# Let1s create a plotting function

def plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=y_pred):
  '''
  Plots training data, test data and compares predictions to ground truth labels.
   '''
  plt.figure(figsize=(10,7))
  # plot training data in blue
  plt.scatter(train_data,train_labels, c = 'b', label = 'Training data')

  #plot test data in green
  plt.scatter(test_data, test_labels, c = 'g', label = 'Testing data')

  #plot model`s predictions in red
  plt.scatter(test_data,predictions, c = 'r', label='Predictions')

  # Show legend
  plt.legend()

plot_predictions()

## Evaluating our model`s predictions with regression Evaluation Metrics

# Depending on the problem you\`re working on there will be different evaluation metrics to evaluate your model\`s performance

# since we are working on a regression model, the two main metrics are:
# * MAE- mean absolute error, on average how wrong is each of my models predictions
# * MSE - mean square error, square the avarage errors

# Evaluate the model on the test set
model.evaluate(X_test, y_test)

# Calculate the mean absolute error
mae = tf.metrics.mean_absolute_error(y_true=y_test,y_pred=tf.constant(y_pred))
mae

y_test, tf.constant(y_pred)

# The y_test and tf.constant(y_pred) are not of the same rank hence need to be squeezed so that they can be of the same dimension

tf.squeeze(y_pred)

# Calculate the mean absolute error, with tensors of same ranks

mae = tf.metrics.mean_absolute_error(y_true = y_test, y_pred = tf.squeeze(y_pred))
mae

# Calculating mean square error MSE
mse = tf.metrics.mean_squared_error(y_test, tf.squeeze(y_pred))
mse

### make reusable MAE and MSE functions

def mae(y_true, y_pred):
  return tf.metrics.mean_absolute_error(y_true,tf.squeeze(y_pred))

def mse(y_true,y_pred):
  return tf.metrics.mean_squared_error(y_true,tf.squeeze(y_pred))



## Running experiments to improve our model

# lets do 3 model experiments:
# 1. model_1 - same as original, 1 layer trained for 100 epochs
# 2. model_2 - 2 layers trained for 100 epochs
# 3. model_3 - 2 layers trained for 500 epochs

### model_1

tf.random.set_seed(42)

# Build the model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Compile the model
model_1.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

# Fit the model
model_1.fit(tf.expand_dims(X_train,axis=-1),tf.expand_dims(y_train, axis=-1), epochs = 100)

# Make and plot predictions for our model
y_pred_1 = model_1.predict(X_test)

plot_predictions(predictions=y_pred_1)

# Calculate model_1 evaluation metrics

mae_1 = mae(y_test,y_pred_1)
mse_1 = mse(y_test,y_pred_1)

mae_1, mse_1

### Model 2
# 2 dense layers trainied for 100 epochs

tf.random.set_seed(42)

# Build the model

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_2.compile(loss = tf.keras.losses.MAE,
                optimizer = tf.keras.optimizers.SGD(),
                metrics = ['mae'])

# Fit the model
model_2.fit(tf.expand_dims(X_test, axis=-1), tf.expand_dims(y_test, axis=-1), epochs = 100)

# Make predictions

y_pred_2 = model_2.predict(X_test)

# Plot predictions against true values
plot_predictions(predictions=y_pred_2)

model_2.predict(X_test), model_1.predict(X_test)


# Calculate model_2 evaluation metrics
mae_2 = mae(y_test,y_pred_2)
mse_2 = mse(y_test,y_pred_2)

mae_2,mse_2

### Building model_3

# 2 hidden layers, trained for 500 epochs

tf.random.set_seed(42)

# Build the model
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_3.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mse']
)

# fitting the model
model_3.fit(tf.expand_dims(X_train, axis=-1), tf.expand_dims(y_train, axis=-1) ,epochs=500)

# Make and plot predictions
y_pred_3 = model_3.predict(X_test)
plot_predictions(predictions=y_pred_3)

# Calculate the evaluation metrics
mae_3 = mae(y_test,y_pred_3)
mse_3 = mse(y_test,y_pred_3)
mae_3,mse_3

### Comparing the results of our model

# Lets compare our model`s predictions using pandas dataframe

import pandas as pd

model_results= [['model_1',mae_1.numpy(),mse_1.numpy()],
                ['model_2',mae_2.numpy(),mse_2.numpy()],
                ['model_3',mae_3.numpy(),mse_2.numpy()]]

all_results = pd.DataFrame(model_results, columns = ['model','mae','mse'])
all_results

model_1.summary()

## Tracking your experiments

# One good  habit of machine learning modeling is to track the results of your experiments.

# And when doing so it can be tedious when running lots of experiments

# Luckily there are tools to help us:

# * TensorBoard - a component of the tensorflow library to help tracking modelling experiments

# * Weights and Biases - a tool for tracking all kinds of experiments (plugs strait into tensorboard)

## saving our models

# There are 2 main ways in which we can save a model:

# 1. The SavedModel format
# 2. The HDF5 format

# save model using the SavedModel format
model_1.save('simple model')

# save model using the HDF5 format
model_1.save('simple model.h5')

### Loading  in a saved model

# Load in the SavedModel fomart model

load_SavedModel_format = tf.keras.models.load_model('simple model')
load_SavedModel_format.summary()

# Load in the H5 format model

load_H5_format = tf.keras.models.load_model('simple model.h5')
load_H5_format.summary()