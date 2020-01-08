### BMI 데이터를 학습한 후 자신의 키와 데이터를 넣어 확인해보기

```python
## 모듈 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# pandas, sklearn, tensorflow, numpy 등 
# data loading
bmi = pd.read_csv("./bmi.csv",skiprows=3,encoding="CP949")
# skiprows: bmi 데이터의 지표에 대한 설명
```

> 결측치 이상치 먼저 처리

```python
bmi.isnull.sum() # 결측치 처리
plt.boxplot(bmi["height"]) # 이상치 확인
plt.boxplot(bmi["weight"]) 
## 결측치 및 이상치 없음
```

> ### 정규화

```python
# 정규화
scaler = MinMaxScaler()
x_data = scaler.fit_transform(bmi[["height","weight"]])

# train/test dataset 나누기
split_num = int(bmi.shape[0] * 0.8)
train_x_data = x_data[:split_num]
test_x_data = x_data[split_num:]
```

> ### One_Hot_Encoding

```python
# one hot encoding 으로 전환 시킬 때
# pandas get_dummies()
# tensorflow one_hot()
sess = tf.Session()
train_y_data = sess.run(tf.one_hot(bmi.loc[:split_num-1,'label'],depth=3)) # -> 2차원이 됨
# df의 loc는 inclusive이기 때문
# np.array의 loc 는 exclusive 
test_y_data = sess.run(tf.one_hot(bmi.loc[split_num:,'label'],depth=3))
```

> ### 학습

```python
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
Y = tf.placeholder(shape=[None,3], dtype=tf.float32)

W = tf.Variable(tf.random_normal([2,3]), name = "weight")
b = tf.Variable(tf.random_normal([3]), name = "bias")
logit = tf.matmul(X,W)+b
H=tf.nn.softmax(logit)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels = Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
    _,cost_val = sess.run([train,cost], feed_dict={X:train_x_data,
                                                  Y:train_y_data})
    if step % 3000 == 0:
        print("cost 값은: {}".format(cost_val))
```

```
cost 값은: 1.8709076642990112
cost 값은: 0.7737191319465637
cost 값은: 0.6093964576721191
cost 값은: 0.5310370922088623
cost 값은: 0.4836674928665161
cost 값은: 0.45039644837379456
cost 값은: 0.4248724579811096
cost 값은: 0.40423455834388733
cost 값은: 0.3869752883911133
cost 값은: 0.3722051978111267
```

> ###  정확도 확인

```python
predict = tf.argmax(H,axis=1) 
correct = tf.equal(predict, tf.argmax(Y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
print("정확도: {}".format(sess.run(accuracy, feed_dict={X:train_x_data,
                                                    Y:train_y_data})))
```

**정확도: 0.9513750076293945**

> 자신의 키/몸무게 넣어서 값 확인

```python
# Prediction
prediction_data = scaler.transform([[152,43]])
prediction_data # 정규화 됨 array([[0.8375    , 0.95555556]])
result = sess.run(tf.argmax(H,1),feed_dict={X: prediction_data})

if result ==0:
    print( "THIN")
elif result == 1:
    print("NORMAL")
else:
    print("FAT")
```

```python
# THIN
```

