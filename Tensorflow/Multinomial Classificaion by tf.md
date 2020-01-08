### Logistic Regression

> let's draw Logistic Regression
>
> You have to import modules needed to conduct

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings # 경고메시지 사전차단
from sklearn.linear_model import LogisticRegression
import mglearn

warnings.filterwarnings(action="ignore") # warning 출력배제
```

> mglearn패키지의 데이터 셋을 가져와서 산점도 그리기

```python
x, y = mglearn.datasets.make_forge() # x는 2컬럼, y는 1
# y값이 0인 x를 추출해서 x의 첫번째 컬럼을 x축, x의 두번째 컬럼을  y축으로
blue = y[y==0]
plt.scatter(blue[:,0],blue[:,1])
orange = x[y==1]
plt.scatter(orange[:,0],orange[:,1])
```

![img](data:image/png;base64,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)

> ##### Logistic Regression 진행하기

```python
# train data set (testet은 넘어간다)
train_x_data = x
train_y_data = y.reshape([-1,1]) # 형태 맞춰주기

# placeholder
X = tf.placeholder(shape=[None,2],dtype = tf.float32)
Y = tf.placeholder(shape=[None,1],dtype = tf.float32)

# Weight, bias
W = tf.Variable(tf.random_normal([2,1]), name = "weight")
b =  tf.Variable(tf.random_normal([1]), name ="bias")

# H
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# Cost Function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels = Y))

# train
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습진행
for step in range(3000):
    _,cost_val = sess.run([optimizer,cost],feed_dict = {X:train_x_data,
                                                   Y:train_y_data})
    if step % 300 ==0:
        print("cost 값은: {}".format(cost_val))
```

>  #### 정확도 측정 : 95%이상 나오면 괜찮은 모델

```python
# Prediction
plt.scatter(blue[:,0],blue[:,1])
orange = x[y==1]
plt.scatter(orange[:,0],orange[:,1])

sess.run(H, feed_dict={X:[[9,4]]})
plt.scatter(9,4)
blue = x[y==0]
```

> #### 산점도 그래프 그리기

```python
model = LogisticRegression()
myModel = model.fit(x,y) # 학습
print(myModel.predict([[9,4]]))
mglearn.plots.plot_2d_separator(myModel, x, fill=False,
                               eps=0.5, alpha = 0.7)
```

![img](data:image/png;base64,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)

## Multinomial Classification

```python
train_x_data = [[10,7,8,5],
               [8,8,9,4],
               [7,8,2,3],
               [7,5,7,4],
                [2,4,3,1],
               [6,3,9,3],
               [3,5,6,2]]
train_y_data = [[1,0,0],
                [1,0,0],
               [1,0,0],
               [0,1,0],
                [0,1,0],
                [0,0,1],
                [0,0,1]]
```

> 학습진행
>
> : Multinomial classification이기 때문에 
>
> 가설(Hypothesis)에서 적용 함수는 sigmoid->softmax

```python
X = tf.placeholder(shape=[None,4], dtype = tf.float32)
Y = tf.placeholder(shape=[None,3], dtype = tf.float32)
W = tf.Variable(tf.random_normal([4,3]),name = "weight")
b = tf.Varibale(tf.random_normal([3]),name = "bias")
logit = tf.matmul(X,W)+b
H = tf.nn.softmax(logit) ### not sigmoid but softmax
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,labels=Y)) # cost역시 softmax~v2 로 변경

train = tf. train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess= tf.Session()
sess.run(tf.global_variables_initializer())

# 학습을 진행
for step in range(3000):
    _, cost_val = sess.run([train,cost],feed_dict={X:train_x_data,
                                                  Y : train_y_data})
    if step % 300 == 0:
        print("cost값은: { }".format(cost_val))
```

```
cost 값은: 5.596889495849609
cost 값은: 0.14735092222690582
cost 값은: 0.08337773382663727
cost 값은: 0.06306320428848267
cost 값은: 0.05059315636754036
cost 값은: 0.042166050523519516
cost 값은: 0.03610163927078247
cost 값은: 0.03153557702898979
cost 값은: 0.027978012338280678
cost 값은: 0.02513059601187706
```

> 정확도(accuracy 측정)

```python
predict = tf.argmax(H, axis = 1) # 가장큰 값의 index 번호 리턴
correct = tf.equal(predict,tf.argmax(Y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
print("정확도: {}".format(sess.run(accuray, feed_dict = {X: train_x_data,
                                                     Y: train_y_data}))) # 정확도: 1.0
```

