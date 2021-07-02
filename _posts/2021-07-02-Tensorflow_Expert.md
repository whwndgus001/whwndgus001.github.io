# Tensorflow로 모델링 하는 2가지 방법
- Sequential 모델을 활용하는 방법
  - Keras에서 제공하는 레이어들을 이어 붙여 가면서 모델링 하는 방법
  - 쉽고 빠르게 모델링 가능
  - 커스터마이징이 조금 힘들다.
- Functional API를 활용하는 방법
  - 개발자가 직접 레이어를 정의해서 keras의 레이어 처럼 사용하게 할 수 있다.(tf.keras.layers 상속)
  - 각종 Loss, Optimizer 등을 직접 만들어서 사용할 수 있다.

# tf.data 사용하기
  - 일반적인 배열이 아닌, 병렬 처리된 배열을 이용하여 매우빠른 속도로 데이터의 입출력이 가능
  - dataset 이라는 개념을 이용해 feature, label을 손쉽게 관리 가능


```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets
```


```python
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

X_train, X_test = X_train / 255.0, X_test / 255.0
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    

## tf.data의 함수들
  - from_tensor_slices()
    - feature, label을 합친 데이터세트(ds)를 생성
  - shuffle()
    - 랜덤하게 섞기
  - batch()
    - 배치 만들기


```python
# 제너레이트할 데이터를 넣어준다. 데이터를 병렬로 처리할 수 있다. ( 속도가 매우 빨라짐 )
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# 데이터 섞기. 매개변수 숫자는 랜덤 시드가 아닙니다. 버퍼사이즈이고, 보통 1kb 정도로 설정한다.
train_ds = train_ds.shuffle(1000)

# 배치 만들기
train_ds = train_ds.batch(32) # iteration 할 때마다 batch_size 만큼 데이터가 나온다.


```


```python
# 테스트 ds 만들기
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# 테스트 세트는 셔플이 필요없다.
test_ds = test_ds.batch(32)
```

## tf.data로 만든 데이터의 시각화


```python
import matplotlib.pyplot as plt
%matplotlib inline
```

take() 함수를 이용하면 데이터를 배치 사이즈만큼 가지고 올 수 있다.( 반복문 에서)


```python
# take는 2개의 결과를 리턴 ( feature_batch, label_batch )
for images, labels in train_ds.take(2):
  print(images.shape)
```

    (32, 28, 28, 1)
    (32, 28, 28, 1)
    


```python
for images, labels in train_ds.take(2):
  image = images[0, ..., 0]
  label = labels.numpy()[0]
  plt.title(label)
  plt.imshow(image, 'gray')
  plt.show()
```


    
![output_11_0](https://user-images.githubusercontent.com/69663368/124225557-9d668c00-db42-11eb-978b-1e9b7be51538.png)
    



    
![output_11_1](https://user-images.githubusercontent.com/69663368/124225563-a0617c80-db42-11eb-9f1d-2f9512db25fc.png)
    


# Conv2D 모델링 하기


```python
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D
from tensorflow.keras.layers import Dropout, Flatten, Dense

```


```python
input_shape = ( 28, 28, 1)
num_classes = 10

inputs = layers.Input(shape = input_shape)

# Feature Extraction
net = Conv2D(32, 3, padding = 'SAME')(inputs)
net = Activation("relu")(net)
net = Conv2D(32, 3, padding = 'SAME')(net)
net = Activation("relu")(net)
net = MaxPool2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

net = Conv2D(64, 3, padding = 'SAME')(net)
net = Activation("relu")(net)
net = Conv2D(64, 3, padding = 'SAME')(net)
net = Activation("relu")(net)
net = MaxPool2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

# Fully Connected
net = Flatten()(net)
net = Dense(512)(net)
net = Activation("relu")(net)
net = Dropout(0.25)(net)

# Output Layer
net = Dense(num_classes)(net)
net = Activation("softmax")(net)

model = tf.keras.Model(inputs = inputs, outputs = net, name = 'Basic_CNN')
```


```python
# 지난 시간의 컴파일 과정
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")
```


```python
# tf.data의 훈련 과정
model.fit(train_ds, epochs = 1) # ds에는 feature, label, batch 정보가 모두 들어있다.
```

    1875/1875 [==============================] - 7s 4ms/step - loss: 0.1260
    




    <tensorflow.python.keras.callbacks.History at 0x7f3af00dd940>



# Functional API 사용해 보기
Loss Function 이나 Optimizer등을 Tensorflow 및
Keras에서 제공하는 것이 아닌 사용자가 직접 함수를 만들어서 사용할 때 커스터마이징 하기 위한 방법


```python
# Loss Function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# Optimizer
optimizer = tf.keras.optimizers.Adam()
```


```python
# 평가 방법도 커스터마이징 가능
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')
```

# @tf.function 이란?


```python
@tf.function
def train_step(images, labels):
  # 자동 미분 수행하기( 오차 역전파 )
  with tf.GradientTape() as tape:
    prediction = model(images) # 모델이 train모드가 된다.
    loss = loss_object(labels, prediction)
    
  # 오차 역전파의 수행
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # loss, accuracy 기록 하기
  train_loss(loss)
  train_accuracy(labels, prediction)
  
```


```python
@tf.function
def test_step(images, labels):
  prediction = model(images)
  t_loss = loss_object(labels, prediction)

  test_loss(t_loss)
  test_accuracy(labels, prediction)
```


```python
# MNIST 추가 학습을 방지하기 위한 리모델링

input_shape = ( 28, 28, 1)
num_classes = 10

inputs = layers.Input(shape = input_shape)

# Feature Extraction
net = Conv2D(32, 3, padding = 'SAME')(inputs)
net = Activation("relu")(net)
net = Conv2D(32, 3, padding = 'SAME')(net)
net = Activation("relu")(net)
net = MaxPool2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

net = Conv2D(64, 3, padding = 'SAME')(net)
net = Activation("relu")(net)
net = Conv2D(64, 3, padding = 'SAME')(net)
net = Activation("relu")(net)
net = MaxPool2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

# Fully Connected
net = Flatten()(net)
net = Dense(512)(net)
net = Activation("relu")(net)
net = Dropout(0.25)(net)

# Output Layer
net = Dense(num_classes)(net)
net = Activation("softmax")(net)

model = tf.keras.Model(inputs = inputs, outputs = net, name = 'Basic_CNN')
```


```python
# Functional API 방식으로 훈련 시키기

epochs = 10
for epoch in range(epochs):

  # 한 에폭에서 모든 데이터에 대한 학습을 마친다.
  for images, labels in train_ds:
    train_step(images, labels)

  # 테스트 수행
  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  
  template = 'Epoch {}, Loss : {:.3f}, Accuracy: {:.3f}, Test Loss : {:.3f}, Test Accuracy: {:.3f}'
  print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))

```

    Epoch 1, Loss : 0.017, Accuracy: 99.462, Test Loss : 0.047, Test Accuracy: 99.026
    Epoch 2, Loss : 0.017, Accuracy: 99.476, Test Loss : 0.047, Test Accuracy: 99.031
    Epoch 3, Loss : 0.017, Accuracy: 99.490, Test Loss : 0.048, Test Accuracy: 99.038
    Epoch 4, Loss : 0.016, Accuracy: 99.502, Test Loss : 0.048, Test Accuracy: 99.048
    Epoch 5, Loss : 0.016, Accuracy: 99.513, Test Loss : 0.048, Test Accuracy: 99.053
    Epoch 6, Loss : 0.016, Accuracy: 99.524, Test Loss : 0.049, Test Accuracy: 99.057
    Epoch 7, Loss : 0.015, Accuracy: 99.534, Test Loss : 0.049, Test Accuracy: 99.062
    Epoch 8, Loss : 0.015, Accuracy: 99.543, Test Loss : 0.050, Test Accuracy: 99.065
    Epoch 9, Loss : 0.015, Accuracy: 99.552, Test Loss : 0.050, Test Accuracy: 99.070
    Epoch 10, Loss : 0.014, Accuracy: 99.561, Test Loss : 0.052, Test Accuracy: 99.072
    


```python

```
