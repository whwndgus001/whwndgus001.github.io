손실 함수란 신경망이 얼마나 맞추지 못했냐 라는 지표이다.

* 즉 낮을 수록 좋은 값이다.
* 손실 함수에 영향을 미치는 것은 가중치(W)와 편향(B)
* 신경망의 학습이랑 손실 함수의 값이 낮은 가중치와 편향을 구하는 것이다.

# MSE (Mean Squared Error)

평균 제곱 오차


```python
import numpy as np
# 모델이 2로 예측 했을 확률이 0.6
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

# 실제 정답은 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```


```python
def mean_squared_error(y, t):
  return 0.5 * np.sum( (y - t) ** 2)
```


```python
print("MSE : {}".format(mean_squared_error(np.array(y), np.array(t))))
```

    MSE : 0.09750000000000003
    


```python
# 모델이 7로 예측 했을 확률이 0.6
y_error = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
```


```python
print("MSE : {}".format(mean_squared_error(np.array(y_error), np.array(t))))
```

    MSE : 0.5975
    
#
# CEE ( Cross Entropy Error)

* 교차 엔트로피 에러


```python
def cross_entropy_error(y, t):
  delta = 1e-7 # 아주 작은 값
  return -np.sum(t * np.log(y + delta)) # 델타를 더해주는 이유는?
```


```python
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) # 모델이 2로 예측 했을 확률이 0.6
y_error = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]) # 모델이 7로 예측 했을 확률이 0.6
# 실제 정답은 2 이다.
t = [0,0, 1, 0, 0, 0, 0, 0, 0, 0]

print("y의 CEE : {:.3f}".format(cross_entropy_error(y, t)))
print("y_error의 CEE : {:.3f}".format(cross_entropy_error(y_error, t)))
```

    y의 CEE : 0.511
    y_error의 CEE : 2.303
    

# 미니 배치


```python
from tensorflow.keras import datasets
mnist = datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    


```python
X_train[0].shape
```




    (28, 28)




```python
y_train[0].shape
```




    ()




```python
y_train.shape
```




    (60000,)



y_train을 OneHotEncoding 하세요
* 목표 : (60000, 10)


```python
y_train_step1 = y_train.reshape(-1, 1)
y_train_step1.shape
```




    (60000, 1)




```python
y_train_step1[:3]
```




    array([[5],
           [0],
           [4]], dtype=uint8)




```python
from sklearn.preprocessing import OneHotEncoder
y_train_dummy = OneHotEncoder().fit_transform(y_train_step1)
y_train_dummy = y_train_dummy.toarray()

y_train_dummy.shape
```




    (60000, 10)




```python
y_train_dummy[:3]
```




    array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])




```python
X_train = X_train.reshape(60000, -1)
X_train.shape
```




    (60000, 784)




```python
X_train.shape, y_train_dummy.shape
```




    ((60000, 784), (60000, 10))



미니 배치? - 랜덤하게 뽑은 인덱스로 배치를 만든 것


```python
# 60000장의 데이터 중에 무작위로 10개만 뽑아서 미니 배치를 만들자-
# choice(범위 숫자, 개수)

train_size = X_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 60000개 중에 10개를 무작위로 뽑는다.

print(batch_mask)
```

    [ 8351 49993 57815 40353 36245 38463 55762 22103 29121  8174]
    


```python
# 마스크에 의해서 선택된 데이터 뽑기
X_batch = X_train[batch_mask]
t_batch = y_train_dummy[batch_mask]
```

배치용 크로스 엔트로피 구성하기 ver.1


```python
def cross_entropy_error_v1(y, t):

  # batch를 쓰지 않는 경우, 즉 1장만 검사하는 경우,
  if y.ndim == 1 :
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(t * np.log(y)) / batch_size # OneHotEncoding된 레이블에만 사용 가능하다.  np.log(y)가 공식의 ynk까지를 포함해서 n으로나눈다 (batch_size)로 나눈다.
```


```python
def cross_entropy_error_v2(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = t.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
```


```python

```
