from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

tf.enable_eager_execution()

dir_path = os.path.dirname(os.path.realpath(__file__))
train_filename = "C:/Users/Peter/Documents/primeBinary.csv"
test_filename = "C:/Users/Peter/Documents/testPrimeBinary.csv"

column_names = ['bit0', 'bit1','bit2' ,'bit3' ,'bit4' ,'bit5' , 'bit6', 'bit7','bit8','bit9','bit10','bit11','bit12','bit13','bit14','bit15','bit16', 'bit17','bit18' ,'bit19' ,'bit20' ,'bit21' , 'bit22', 'bit23','bit24','bit25','bit26','bit27','bit28','bit29','bit30','bit31', 'label']
class_names = ['composite', 'prime']

feature_names = column_names[:-1]
label_name = column_names[-1]
batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_filename,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

test_dataset = tf.contrib.data.make_csv_dataset(
    test_filename,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

train_dataset = train_dataset.map(pack_features_vector)
test_dataset = test_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(32,)),  # input shape required
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.train.get_or_create_global_step()
loss_value, grads = grad(model, features, labels)

optimizer.apply_gradients(zip(grads, model.variables), global_step)

train_loss_results = []
train_accuracy_results = []

num_epochs = 8000

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 1000 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

#Test the trained model
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

#create single test input loop
inputToTest = input("Please enter data to test:")
input = inputToTest.split(',')
input = list(map(float, iter(input)))
print(input)
predict_dataset2 = tf.convert_to_tensor([input])
predict_dataset = tf.convert_to_tensor([
    
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
