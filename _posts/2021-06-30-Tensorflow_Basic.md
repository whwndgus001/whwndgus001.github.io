## 2021-06-30
```python
import tensorflow as tf
import numpy as np
```


```python
# List -> tensor
tf.constant([1, 2, 3])
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>




```python
# Tuple -> tensor
tf.constant((1, 2, 3))
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>




```python
# Ndarray -> tensor
arr = np.array([1, 2, 3])
tf.constant(arr)
```




    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 2, 3])>



# Tensor 정보 확인하기


```python
tensor = tf.constant(np.array([1, 2, 3]))
tensor.shape
```




    TensorShape([3])




```python
tensor
```




    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 2, 3])>




```python
# 텐서를 만들 때 데이터 타입 지정해 주기
tensor = tf.constant([1, 2, 3], dtype = tf.float32)
tensor
```




    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>




```python
# 텐서를 numpy 배열화 하기
tensor.numpy()
```




    array([1., 2., 3.], dtype=float32)




```python
np.array(tensor)
```




    array([1., 2., 3.], dtype=float32)




```python
np.random.randn(9)
```




    array([ 0.14098203, -1.27555536, -1.14363966,  0.66870985,  1.58020165,
           -0.19326392, -0.9800438 ,  1.0589563 ,  0.67291172])




```python
tf.random.normal((3, 3))
```




    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[-1.3239969 ,  1.3226523 ,  0.01675554],
           [ 0.277809  ,  0.13703635,  0.12713248],
           [-0.41889596, -0.20001346,  0.17583163]], dtype=float32)>




```python
# 정규분포 만들기
tf.random.uniform((3, 3))
```




    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[0.8790958 , 0.02442348, 0.02669275],
           [0.47820878, 0.76478684, 0.7864615 ],
           [0.7643367 , 0.1469593 , 0.33033586]], dtype=float32)>



Tensor 형상관리


```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
image = X_train[0]
image.shape
```




    (28, 28)




```python
# matplotlib으로 이미지 시각화를 할 때 grayscale이면 채널 데이터가 필요 없음
# RGB면 채널 데이터 필요함
plt.imshow(image, 'gray')
plt.show()
```


![output_17_0](https://user-images.githubusercontent.com/69663368/123986597-68065500-da01-11eb-93c0-b295c6380f8f.png)
    



```python
# CNN 훈련시에는 채널 데이터가 필요하다..
X_train.shape
```




    (60000, 28, 28)




```python
X_train[..., tf.newaxis].shape
```




    (60000, 28, 28, 1)




```python
X_train.reshape((60000, 28 , 28, 1)).shape
```




    (60000, 28, 28, 1)




```python
X_train_reshaped = X_train[..., tf.newaxis]
X_train_reshaped.shape
```




    (60000, 28, 28, 1)




```python
plt.title(y_train[0])
plt.imshow(X_train_reshaped[0, ..., 0], 'gray')
plt.show()
```


    
![output_22_0](https://user-images.githubusercontent.com/69663368/123986636-705e9000-da01-11eb-8c36-c092949dcfe8.png)
    


# One Hot Encoding


```python
from tensorflow.keras.utils import to_categorical
```


```python
to_categorical(1, 5) # 1을 5개로 OHE
```




    array([0., 1., 0., 0., 0.], dtype=float32)




```python
label = y_train[0]
label
```




    5




```python
label_onehot = to_categorical(label, num_classes = 10)
label_onehot
```




    array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)




```python

```
