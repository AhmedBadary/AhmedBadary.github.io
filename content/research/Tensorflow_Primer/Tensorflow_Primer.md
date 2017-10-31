---
layout: NotesPage
title: Tensorflow <br /> Primer
permalink: /work_files/research/Tensorflow_Primer/tf_primer
prevLink: /work_files/research/conv_opt.html
---



```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Multiplication


```python
# Creating the Computation Graph
x = tf.placeholder("float")
y = tf.placeholder("float")

xy = tf.multiply(x,y)
```


```python
# Creating the Session
with tf.Session() as sess:
    print("%f x %f = %f"%(2, 3, sess.run(xy, feed_dict = {x:2, y:3})))
```

    2.000000 x 3.000000 = 6.000000


## Linear Regression


```python
# Setting up the data
trX = np.linspace(-1, 1, 500)
trY = 2 * trX + np.random.randn(*trX.shape)*.35 + 2
plt.scatter(trX,trY);
```


![png](/content/research/Tensorflow_Primer/output_5_0.png)



```python
# Setting up the variables and the graph
X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0,name="weights")
b = tf.Variable(0.0, name="bias")
y_hat = tf.add(tf.multiply(X,w),b)
```


```python
# Defining the objective function and the optimizer
cost = tf.reduce_mean(tf.square(Y - y_hat))
train_operation = tf.train.GradientDescentOptimizer(.01).minimize(cost)
```


```python
# Running the Session (Computation)
numEpochs = 200
costs = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for i in range(numEpochs):
        sess.run(train_operation,feed_dict={X:trX,Y: trY})
        costs.append(sess.run(cost,feed_dict={X:trX,Y: trY}))
        
    print("Final Error is %f"%costs[-1])
    wfinal,bfinal = sess.run(w),sess.run(b)
    print("Predicting  y = %.02f x + %.02f"%(wfinal,bfinal))
    print("Actually is y = %.02f x + %.02f"%(2,2))
```

    Final Error is 0.195300
    Predicting  y = 1.45 x + 1.95
    Actually is y = 2.00 x + 2.00



```python
# Plotting the cost
plt.plot(costs)
plt.ylabel("Mean Squared Error")
plt.xlabel("Epoch");
```


![png](/content/research/Tensorflow_Primer/output_9_0.png)


## (Multivariable) Linear Regression


```python
# Setting up the data
m = 8
n = 5
NUM_EXAMPLES = 100

W = np.random.rand(n,m)

trX = np.random.rand(100,n)
# trY = tf.multiply(X,W) + np.random.randn(NUM_EXAMPLES,m)
trY = trX.dot(W) + np.random.randn(NUM_EXAMPLES,m)
```


```python
# Setting up the variables and the graph
x = tf.placeholder("float",shape=[None, n])
y = tf.placeholder("float",shape=[None, m])

w = tf.Variable(tf.zeros([n,m]))
y_hat = tf.matmul(x,w)
```


```python
# Defining the objective function and the optimizer
cost = tf.reduce_mean(tf.square(y - y_hat))
train_operation = tf.train.GradientDescentOptimizer(.01).minimize(cost)
```


```python
# Running the Session (Computation)
numEpochs = 1000
costs = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for i in range(numEpochs):
        sess.run(train_operation,feed_dict={x:trX,y: trY})
        costs.append(sess.run(cost,feed_dict={x:trX,y: trY}))
        
    print("Final Error is %f"%costs[-1])
```

    Final Error is 1.051898



```python
plt.plot(costs)
plt.ylabel("Mean Squared Error")
plt.xlabel("Epoch");
```


![png](/content/research/Tensorflow_Primer/output_15_0.png)


## Logistic Regression


```python
# Importing the data and initilizing the variables
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/",one_hot=True)
trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels
```

    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    Extracting MNIST/train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    Extracting MNIST/train-labels-idx1-ubyte.gz
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting MNIST/t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting MNIST/t10k-labels-idx1-ubyte.gz



```python
# Setting up the graph
X = tf.placeholder("float",shape=[None,784])
Y = tf.placeholder("float",shape=[None,10])

w = tf.Variable(tf.random_normal([784,10], stddev=0.01))

pred_logit = tf.matmul(X,w)
sample_cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit,labels=Y)
total_cost = tf.reduce_mean(sample_cost)

train_operation = tf.train.GradientDescentOptimizer(0.05).minimize(total_cost)
predict_operation = tf.argmax(pred_logit, 1)
accuracy_operation = tf.reduce_mean(
                        tf.cast(tf.equal(predict_operation,tf.argmax(Y,1)),tf.float32)
                        )
```


```python
NUM_EPOCHS = 30
BATCH_SIZE = 200

accuracies = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for epoch in range(NUM_EPOCHS):
        for start in range(0,len(trX),BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_operation, \
                    feed_dict = {X: trX[start:end],Y: trY[start:end]})
        accuracies.append(sess.run(accuracy_operation,feed_dict= {X: teX,Y: teY}))
```


```python
plt.plot(accuracies)
```




    [<matplotlib.lines.Line2D at 0x12c3becc0>]




![png](/content/research/Tensorflow_Primer/output_20_1.png)


## Neural Networks


```python
NUM_HIDDEN = 620

X = tf.placeholder("float",shape=[None,784])
Y = tf.placeholder("float",shape=[None,10])

def init_weights(shape): # We define this out of convenience
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

W_h = init_weights([784,NUM_HIDDEN]) # Weights entering the hidden layer
W_o = init_weights([NUM_HIDDEN,10]) # Weights entering the output layer
entering_hidden = tf.matmul(X,W_h)
exiting_hidden = tf.nn.sigmoid(entering_hidden)
model = tf.matmul(exiting_hidden,W_o)

sample_cost = tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y)
total_cost = tf.reduce_mean(sample_cost)

train_operation = tf.train.GradientDescentOptimizer(0.2).minimize(total_cost)
predict_operation = tf.argmax(model, 1)
accuracy_operation = tf.reduce_mean(
                        tf.cast(tf.equal(predict_operation,tf.argmax(Y,1)),tf.float32)
                        )
NUM_EPOCHS = 100
BATCH_SIZE = 200
import tqdm
accuracies = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for epoch in tqdm.trange(NUM_EPOCHS):
        for start in range(0,len(trX),BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_operation, \
                    feed_dict = {X: trX[start:end],Y: trY[start:end]})
        accuracies.append(sess.run(accuracy_operation,feed_dict= {X: teX,Y: teY}))
        print(accuracies[-1])
```

      1%|          | 1/100 [00:05<09:05,  5.51s/it]

    0.8196


      2%|▏         | 2/100 [00:11<09:03,  5.54s/it]

    0.8719


      5%|▌         | 5/100 [00:31<09:58,  6.30s/it]

    0.9078


     10%|█         | 10/100 [01:05<09:52,  6.58s/it]

    0.9186


     15%|█▌        | 15/100 [01:38<09:16,  6.54s/it]

    0.926


     20%|██        | 20/100 [02:03<08:14,  6.18s/it]

    0.9339


     25%|██▌       | 25/100 [02:28<07:25,  5.94s/it]

    0.9407


     30%|███       | 30/100 [02:53<06:44,  5.78s/it]

    0.9478


     35%|███▌      | 35/100 [03:20<06:11,  5.72s/it]

    0.9521


     36%|███▌      | 36/100 [03:25<06:05,  5.71s/it]

    0.953


     37%|███▋      | 37/100 [03:30<05:58,  5.68s/it]

    0.9532


     38%|███▊      | 38/100 [03:35<05:50,  5.66s/it]

    0.954


     39%|███▉      | 39/100 [03:40<05:44,  5.65s/it]

    0.9549


     40%|████      | 40/100 [03:45<05:38,  5.63s/it]

    0.9555


     45%|████▌     | 45/100 [04:10<05:06,  5.57s/it]

    0.9583


     50%|█████     | 50/100 [04:36<04:36,  5.53s/it]

    0.9613


     70%|███████   | 70/100 [06:17<02:41,  5.39s/it]

    0.97

     80%|████████  | 80/100 [07:06<01:46,  5.33s/it]

    0.9722


     85%|████████▌ | 85/100 [07:35<01:20,  5.36s/it]

    0.9731


     90%|█████████ | 90/100 [08:05<00:53,  5.39s/it]

    0.9739


     95%|█████████▌| 95/100 [08:31<00:26,  5.39s/it]

    0.9742


    100%|██████████| 100/100 [08:57<00:00,  5.37s/it]

    0.9746


    



```python
print("Final Accuracy was %.04f"%accuracies[-1])
plt.plot(accuracies);plt.ylim(.9,1);
```

    Final Accuracy was 0.9746



![png](/content/research/Tensorflow_Primer/output_23_1.png)


## Modern Neural Network 
### RELU | Dropout | RMSProp Optimization | More hidden Layers


```python
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def model_gen(X,w_h,w_h2, w_o,drop_rate_input,drop_rate_hidden):
    out_X = tf.nn.dropout(X, drop_rate_input)
    
    in_H = tf.matmul(X,w_h)
    out_H = tf.nn.dropout(tf.nn.relu(in_H),drop_rate_hidden)
    
    in_H2 = tf.matmul(out_H,w_h2)
    out_H2 = tf.nn.relu(in_H2)
    
    model = tf.matmul(out_H2,w_o)
    return model

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

drop_rate_input = tf.placeholder("float")
drop_rate_hidden = tf.placeholder("float")


model = model_gen(X,w_h,w_h2,w_o,drop_rate_input,drop_rate_hidden)

sample_cost = tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y)
total_cost = tf.reduce_mean(sample_cost)

train_operation = tf.train.RMSPropOptimizer(0.001,0.9).minimize(total_cost)
predict_operation = tf.argmax(model, 1)
accuracy_operation = tf.reduce_mean(
                        tf.cast(tf.equal(predict_operation,tf.argmax(Y,1)),tf.float32)
                        )
NUM_EPOCHS = 20
BATCH_SIZE = 200
import tqdm
accuracies = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for epoch in tqdm.trange(NUM_EPOCHS):
        for start in range(0,len(trX),BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_operation, \
                    feed_dict = {X: trX[start:end],Y: trY[start:end],
                    drop_rate_input: 0.8, drop_rate_hidden: 0.5})
        accuracies.append(sess.run(accuracy_operation,feed_dict= {X: teX,Y: teY,drop_rate_input: 1, drop_rate_hidden: 1}))
        print(accuracies[-1])
```

      5%|▌         | 1/20 [00:13<04:19, 13.65s/it]

    0.9028


     10%|█         | 2/20 [00:26<03:58, 13.23s/it]

    0.9595


     15%|█▌        | 3/20 [00:36<03:29, 12.33s/it]

    0.9632


     20%|██        | 4/20 [00:48<03:15, 12.22s/it]

    0.9742


     25%|██▌       | 5/20 [01:00<03:02, 12.20s/it]

    0.971


     30%|███       | 6/20 [01:12<02:49, 12.14s/it]

    0.9753


     35%|███▌      | 7/20 [01:25<02:38, 12.20s/it]

    0.9777


     40%|████      | 8/20 [01:37<02:26, 12.19s/it]

    0.979


     45%|████▌     | 9/20 [01:49<02:13, 12.15s/it]

    0.98


     50%|█████     | 10/20 [02:01<02:01, 12.19s/it]

    0.9789


     55%|█████▌    | 11/20 [02:14<01:49, 12.20s/it]

    0.9817


     60%|██████    | 12/20 [02:25<01:37, 12.14s/it]

    0.9809


     65%|██████▌   | 13/20 [02:36<01:24, 12.07s/it]

    0.9823


     70%|███████   | 14/20 [02:47<01:11, 11.96s/it]

    0.9824


     75%|███████▌  | 15/20 [02:58<00:59, 11.87s/it]

    0.9821


     80%|████████  | 16/20 [03:09<00:47, 11.82s/it]

    0.9817


     85%|████████▌ | 17/20 [03:21<00:35, 11.83s/it]

    0.984


     90%|█████████ | 18/20 [03:34<00:23, 11.91s/it]

    0.9824


     95%|█████████▌| 19/20 [03:46<00:11, 11.94s/it]

    0.9833


    100%|██████████| 20/20 [03:59<00:00, 11.97s/it]

    0.9826


    



```python
print("Final Accuracy was %.04f"%accuracies[-1])
plt.plot(accuracies);plt.ylim(.9,1);
```

    Final Accuracy was 0.9826



![png](/content/research/Tensorflow_Primer/output_26_1.png)



```python

```
