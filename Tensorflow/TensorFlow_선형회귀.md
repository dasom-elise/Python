## TensorFlow

#### linear Regression을 활용해 머신러닝 구현하기

> 모듈 import 및 training data set 생성

```python
import tensorflow as tf
x_data = [1,2,3]
y_data = [1,2,3]
```

> placeholder에 값 임시 저장: tf 그래프에 입력을 주기위한 parameter 역할

```Python
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)
```

> Weight, bias 생성 및 가설 제시

```python
W = tf.Variable(tf.random_normal([1]),name = "weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

H = W * x + b
```

> cost function(loss function) 알아내자
>
> : 가설에서 y값을 빼서 그 값을 제곱해서 평균구하기
>
> tf의 `reduce_mean`함수 활용

```python
cost = tf.reduce_mean(tf.square(H-y))
```

> Train node 생성
>
> : cost 함수를 미분
>
> : 미분을 통해 인접한 직선을 구해서 기울기를 찾는다.

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) 
# learning rate =얼마나움직일지 알려주는 값, 경험을 통해 스스로 알아내야
# 보통은 0.01
train = optimizer.minimize(cost)
```

> 그래프를 실행시키기 위해서 runner(session)이 필요
>
> : Variable을 사용하면 학습하기 전에 반드시 초기화 필요!

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

> learning 학습

```python
for step in range(3000):
    empty,w_val, cost_val,b_val = sess.run([train,W,cost,b],
                                      feed_dict={
                                          x : x_data,
                                          y : y_data
                                      })
    if step % 300 ==0:
        print("w값 : {}, b값: {}, cost값: {}".format(w_val,
                                                 b_val,
                                                 cost_val))
```

