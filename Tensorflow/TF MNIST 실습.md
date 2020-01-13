### Multinomial Classification 2

예제 MNIST (from Kaggle/TF)

> MNIST는 이미지를 학습하고 prediction하는 예제
>
> 28*28크기의 이미지

1. tensorflow에서 제공하는 MNIST예제를 이용해서 작업

- module import

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
```

- Data Load

```python
mnist = input_data.read_data_sets("./mnist",one_hot = True) #train 2개 test2개, train_label = y 
mnist.train.images.shape  # (55000, 784)
mnist.train.images # array([[0., 0., 0., ..., 0., 0., 0.],
   #    [0., 0., 0., ..., 0., 0., 0.],
   #    [0., 0., 0., ..., 0., 0., 0.],
   #    ...,
   #    [0., 0., 0., ..., 0., 0., 0.],
   #    [0., 0., 0., ..., 0., 0., 0.],
   #    [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
```

- dataframe으로 만들기 

> train_x_data & train_y_data

```python
train_x_data_df = pd.DataFrame(mnist.train.images)
train_x_data_df.to_csv("./mnist/mnist_x_data.csv", index = False)
# x data = 0 or 1인 사람으로 각 픽셀값은 이미 scale 되어 있는 상태 ()

train_y_data_df = pd.DataFrame(mnist.train.labels)
train_y_data_df.to_csv("./mnist/mnist_y_data.csv", index = False)
```

- 학습

```python
X = tf.placeholder(shape = [None,784], dtype = tf.float32)
Y = tf.placeholder(shape = [None,10], dtype = tf.float32)

W = tf.Variable(tf.random_normal([784,10]), name = "weight")
b = tf.Variable(tf.random_normal([10]), name = "bias")

# Hypothesis
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

# Cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit,labels = Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
train_epoch = 300
for step in range(train_epoch):
    _, cost_val = sess.run([train,cost],
                          feed_dict = {X:mnist.train.images,
                                      Y:mnist.train.labels})
    if step % 30 == 0:
        print("cost값은: {}".format(cost_val))
```

```
cost값은: 3.0225110054016113
cost값은: 2.631190299987793
cost값은: 2.3517580032348633
cost값은: 2.142700433731079
cost값은: 1.980194091796875
cost값은: 1.8498964309692383
cost값은: 1.7428096532821655
cost값은: 1.6530085802078247
cost값은: 1.5764440298080444
cost값은: 1.510246753692627
```

- 학습 후, 정확도(Accuracy 측정)

```python
predict = tf.argmax(H,1)
correct = tf.equal(predict, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct,dtype = tf.float32))
print("정확도는: {}".format(sess.run(accuracy, feed_dict = {X:mnist.test.images,
                                                       Y:mnist.test.labels})))
# 정확도는: 0.7312999963760376
```

### 학습의 정확도를 높이기 위해 epoch와 batch_size 설정

```python
X = tf.placeholder(shape = [None,784], dtype = tf.float32)
Y = tf.placeholder(shape = [None,10], dtype = tf.float32)

W = tf.Variable(tf.random_normal([784,10]), name = "weight")
b = tf.Variable(tf.random_normal([10]), name = "bias")

# Hypothesis
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

# Cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit,labels = Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
train_epoch = 30
batch_size = 100
for step in range(train_epoch):
    num_of_iter = int(mnist.train.num_examples / batch_size)
    cost_val = 0
    for i in range(num_of_iter):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        _,cost_val = sess.run([train,cost], feed_dict = {X: batch_x,
                                                        Y: batch_y})
    if step % 10 == 0:
        print("cost값은: {}".format(cost_val))
        
        
predict = tf.argmax(H,1)
correct = tf.equal(predict, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct,dtype = tf.float32))
print("정확도는: {}".format(sess.run(accuracy, feed_dict = {X:mnist.test.images,
                                                       Y:mnist.test.labels})))
```

```python
cost값은: 0.5211901068687439
cost값은: 0.38103264570236206
cost값은: 0.07081672549247742
정확도는: 0.9180999994277954
```

#### 예측 적용해보기

```python
# Prediction
r = np.random.randint(0,mnist.test.num_examples)
# 난수가 의미하는 행의 label값을 먼저 구해봄
print("Label: {}".format(sess.run(tf.argmax(mnist.test.labels[r:r+1], axis=1))))

# 예측해보기

sess.run(tf.argmax(H,1),feed_dict = {X:mnist.test.images[r:r+1]})
print("예측: {}".format(sess.run(tf.argmax(H,1),feed_dict = {X:mnist.test.images[r:r+1]})))



result = mnist.test.images[r:r+1].reshape(28,28)
plt.imshow(result, cmap='gray')
# Label: [1]
# 예측: [1]
# <matplotlib.image.AxesImage at 0x1f41396f748>
```

![img](data:image/png;base64,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)

