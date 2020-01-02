## TensorFlow

```python
import tensorflow as tf
node1 = tf.constant("Hello Wolrd")       # 상수 node를 만드는 함수 constant
print(node1)
```

> ```python
> Tensor("Const_1:0", shape=(), dtype=string)
> ```

- 다른 package, module에 비해서 살짝 특이한 형태의 library
- tensorflow는 3가지 구성요소로 이해하면 쉬움.

1. Node: 수학적인 연산 담당, 데이터의 입출력

2. edge: 동적 데이터를 노드로 실어 나르는 역할

3. Tensor: 다차원배열형태의 동적 데이터

> > Tensorflow 그래프를 실행 시키기 위해서는 runner가 필요
> >
> > Session 생성(session이라고 불리는 runner)

```Python
sess = tf.Session()

sess.run(node1).decode()
```

> ```Python
> 'Hello Wolrd 123'
> ```

###### 2개의 값을 더하는 tensorflow graph를 생성하고 실행시켜서 값을 구해보기

```python
node2 =  tf.constant(10, dtype=tf.float32)
node3 = tf.constant(20, dtype=tf.float32)

node1 = node2 + node3

sess.run(node1)

### 30
```

```python
print(sess.run([node1,node2,node3]))
### 결과: [30.0, 10.0, 20.0]
```

###### 실행시키는 시점에 이미 각 노드의 작성길이가 정해져있음

###### constant는 상수 이기 때문

###### *만약 실행시키는 시점에 각 노드의 값을 결정해서 실행시키려면?*

```python
node1 = tf.placeholder(dtype=tf.float32)
node2 = tf.placeholder(dtype=tf.float32)
node3 = node1 + node2
sess = tf.Session()

result = sess.run(node3, feed_dict = {
    node1: input(),
    node2: input()
})

print("덧셈결과: {}".format(result))
```

