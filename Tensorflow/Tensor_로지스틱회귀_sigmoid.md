## Logistic Regression(binary classification)

> 지도학습 데이터의 label이 0 혹은 1로 설정

###### 이런 데이터는 기존의 선형회귀방식으로는 학습하고 prediction 불가능

다중선형회귀에서 사용가능한 가설:직선, H = Wx + b

직선이 아닌 0에서 1 사이의 값을 가지는 함수로 가설을 표현

`sigmoid function`을 이용해 표현

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-10,10,0.1)
y = 1 / (1 + np.exp(-x))
plt.scatter(x,y)
plt.show()
```

![img](data:image/png;base64,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)

-----------------------

## 실습예제 1)

```python
# training data set 생성
x_data = [[1,1],
         [2,0],
         [5,1],
         [3,3],
         [8,1],
         [2,3],
         [10,0]]
y_data = [[0],[0],[0],[1],[1],[1],[1]]

# Placeholder
W = tf.Variable(tf.random_normal([2,1]),name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
logit = tf.matmul(X,W) + b # sigmoid를 이용한 곡선
H = tf.sigmoid(logit)

# cost Function (텐서플로우에서 제공하는 함수)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits =  logit, labels =Y))

# Train
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

# Session 초기화
sess = tf.Session()
sess.run(tf.global_variable_initializer())

# 학습
for step in range(3000):
    _, cost_val =  sess.run([train, cost], feed_dict =  {X:x_data,Y:y_data})
    if step % 300 == 0 :
        print("cost값은: {}".format(cost_val))
        
        
# predict
result = sess.run(H, feed_dict = {X: [[8,1]]})

if result > 0.5:
    print("시험에 통과: {}".format(result))
else:
    print("시험에떨어짐: {}".format(result))
```

```python
# RESULT
cost값은: 1.1718705892562866
cost값은: 0.3176365792751312
cost값은: 0.2790791988372803
cost값은: 0.26033347845077515
cost값은: 0.2490604668855667
cost값은: 0.24143807590007782
cost값은: 0.23589208722114563
cost값은: 0.23165060579776764
cost값은: 0.2282881885766983
cost값은: 0.2255493402481079
시험에 통과: [[0.96883774]]
```

## 실습예제 2) ---

#### 대학원 입학시험예측

```python
df = pd.read_csv("/admission.csv")
df.head()
```

![image-20191227154354891](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191227154354891.png)

```python
# 모듈 import
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
```

> **상관관계 확인을 통해 전처리**

```python
df.corr() # rank의 값을 조절해야 ; 음의 상관관계
```

![image-20191227164113239](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191227164113239.png)

```python
# 함수를 생성하여 rank의 값을 역순으로 변경
def change_rank(k):
    if  k == 4:
        return 1
    elif k == 3:
        return 2
    elif k == 2:
        return 3
    elif k == 1:
        return 4
df["rank"] = df["rank"].apply(lambda x: change_rank(x))
```

```python
# NA Pretreatment
df.isnull().sum() # 결측치 없음
```

> 이상치 처리

```python
# 이상치 처리

# 이상치 처리하는 함수 생성
def process_outlier(tmp_df,tmp):
    q1,q3 = np.percentile(tmp,[25,75])
    iqr = q3 - q1
    upper = q3 + iqr * 1.5 # 상위임계치
    lower = q1 - iqr * 1.5 # 하위임계치
    upper_mask = tmp > upper
    lower_mask = tmp < lower
    
    result_mask = upper_mask | lower_mask
    return tmp_df[~result_mask]
#############################################################
df = process_outlier(df,df["gre"])
df = process_outlier(df,df["gpa"])
df.shape
# (395, 4)
```

> 머신러닝

```python
x_data = df[["gre","gpa","rank"]].values
x_data = MinMaxScaler().fit_transform(x_data)
y_data = df["admit"].values.reshape(-1,1)

#플홀
X = tf.placeholder(shape = [None,3], dtype =tf.float32)
Y = tf.placeholder(shape = [None,1], dtype= tf.float32)

#
W = tf.Variable(tf.random_normal([3,1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)


cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits =  logit, labels =Y))

ptimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(30000):
    _, cost_val =  sess.run([train, cost], feed_dict =  {X:x_data,Y:y_data})
    if step % 300 == 0 :
        print("cost값은: {}".format(cost_val))

```

-------

> 정확도 측정

```python
## Test Data set 존재 X
# train dataset을 test data test로 사용

predict = tf.cast(H > 0.5, dtype =tf.float32)
correct = tf.equal(predict,Y) # TRUE, FALSE, TRUE, TRUE
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
print("정확도: {}".format(sess.run(accuracy, feed_dict={X:x_data,Y:y_data})))
```

```python
# 결과
# 정확도: 0.701265811920166
```

