### Titanic 01.07.

```
1. kaggle의 타이타닉 문제 해결
2. Logistic Regression이 어떤 의미를 가지는지

* 로지스틱 회귀를 Tensorflow를 이용해서 구현
```

> Data Set 필요
>
> 정확도 측정을 위해 학습용 데이터와 평가용 데이터를 따로 분리할 것
>
> train_df를 분리해서 학습용 데이터와 평가용 데이털 ㄹ생성
>
> 상위 80%를 학습용, 20%를 평가용 데이터로 사용

```python
train_df.shape # tuple, 891,10
import tensorflow as tf # module import
# 데이터 drop (PassengerID는 필요하지 않음)
train_df = train_df.drop("PassengerId",axis=1,inplace=False)
# shape를 구하면 891,9
```

```python
train_num = int(train_df.shape[0] * 0.8) # 712 (0~711번까지)
# train datsets 구성
train_x_data = train_df.drop("Survived",axis = 1, inplace=False)[:train_num].values # inplace = False이면 원본을 변하게 하지 않음
test_x_data = train_df.drop("Survived",axis = 1, inplace=False)[train_num:].values # Values를 써서 값형태로 도출

### train_x_data = array([[3., 0., 1., ..., 0., 0., 0.],
#       [1., 1., 1., ..., 3., 2., 2.],
 #      [3., 1., 1., ..., 0., 0., 1.],
   #    ...,
 #      [3., 0., 1., ..., 0., 2., 3.],
  #     [1., 1., 1., ..., 2., 2., 3.],
    #   [1., 0., 1., ..., 1., 0., 0.]])
```

```python
# y label 및 Survived데이터 필요
train_y_data = train_df["Survived"][:train_num].values.reshape([-1,1])
test_y_data = train_df["Survived"][train_num:].values.reshape([-1,1])
```

#### data set 구성완료 후, tensor flow를 이용한 logistic regression code 적용

> Placeholder
>
> ```python
> X = tf.placeholder(shape=[None,8],dtype=tf.float32)
> Y = tf.placeholder(shape=[None,1],dtype=tf.float32)
> ```
>
> Weight, Bias
>
> ```
> W = tf.Variable(tf.random_normal([8,1]),name='weight')
> b = tf.Variable(tf.random_normal([1]),name="bias")
> ```
>
> Hypothesis
>
> ```python
> logit = tf.matmaul(X,W) + b
> H = tf.sigmoid(logit)
> ```
>
> Cost Function ( linear에서는 최소제곱법 사용)
>
> ```python
> cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=Y)) 
> optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
> # optimizer은 값을 줄이는 역할을 하는 주체가 된다
> train = optimizer.minimize(cost)
> ```
>
> Session 초기화 작업 필요
>
> ```python
> # session, 초기화작업 필요
> sess = tf.Session()
> sess.run(tf.global_variables_initializer())
> 
> # 학습
> for step in range(3000):
>     _, cost_val = sess.run([train, cost], feed_dict = {X : train_x_data,
>                                                       Y : train_y_data})
>     # => cost값을 3000번 줄여라 -> 이 과정 겪으면서 최적의 W,b 값이 도출된다
>     if step % 300 == 0:
>         print("cost 값은 : {}".format(cost_val))
>     
> # 우리가 원하는 W와 b를 구했어요! => model을 구성했어요!
> # 정확도를 측정
> # 테스트용 x입력데이터를(test_x_data) 넣어서 예측을 해요!
> # 이렇게 구한 예측값과 y 입력데이터를(test_y_data) 비교해요!
> 
> # 예측값과 실제데이터의 차이를 "비율"로 계산해보아요
> predict = tf.cast(H > 0.5, dtype = tf.float32)
> # boolean으로 나온 값을 0과 1과 같은 숫자형태로 나오게 해줌
> correct = tf.equal(predict, Y)
> accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
> print("정확도: {}".format(sess.run(accuracy,feed_dict = {X: test_x_data,
>                                                       Y: test_y_data})))
> 
> ```