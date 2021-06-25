Fashion MNIST는 keras 데이터셋에서 불러온다.


```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
```


```python
# 데이터 불러오기
fashion_mnist = tf.keras.datasets.fashion_mnist
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-5-2c6c2b7c2014> in <module>()
          1 # 데이터 불러오기
          2 fashion_mnist = tf.keras.datasets.fashion_mnist
    ----> 3 fashion_mnist.shape()
    

    AttributeError: module 'tensorflow.keras.datasets.fashion_mnist' has no attribute 'shape'



```python
from tensorflow.keras import datasets
mnist = datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    32768/29515 [=================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    8192/5148 [===============================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 0s 0us/step
    


```python
X_train.shape, y_train.shape
```




    ((60000, 28, 28), (60000,))




```python
image = X_train[0]
image.shape
```




    (28, 28)




```python
plt.imshow(image, 'gray')
plt.show()
```


    
![output_6_0](https://user-images.githubusercontent.com/69663368/123373399-4865cb80-d5c0-11eb-8f1d-b006106e7388.png)

    



```python
# 정규화
X_train = X_train / 255.0
X_test = X_test / 255.0
```


```python
model = Sequential()
model.add(Flatten())
model.add(Dense(512, activation = 'relu', input_shape = (784, )))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

modelcheckpoint =    ModelCheckpoint('best_model.h5', 
                     monitor='val_acc', 
                     mode='max', 
                     verbose=1,
                     save_best_only=True)

# 모델 컴파일
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

# 훈련
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=30,
    callbacks=[modelcheckpoint],
    validation_data=(X_test, y_test)
)
```

    Epoch 1/30
    468/469 [============================>.] - ETA: 0s - loss: 0.5130 - acc: 0.8163
    Epoch 00001: val_acc improved from -inf to 0.84600, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.5127 - acc: 0.8164 - val_loss: 0.4288 - val_acc: 0.8460
    Epoch 2/30
    469/469 [==============================] - ETA: 0s - loss: 0.3620 - acc: 0.8667
    Epoch 00002: val_acc improved from 0.84600 to 0.85430, saving model to best_model.h5
    469/469 [==============================] - 6s 12ms/step - loss: 0.3620 - acc: 0.8667 - val_loss: 0.4015 - val_acc: 0.8543
    Epoch 3/30
    468/469 [============================>.] - ETA: 0s - loss: 0.3257 - acc: 0.8799
    Epoch 00003: val_acc improved from 0.85430 to 0.87240, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.3256 - acc: 0.8799 - val_loss: 0.3557 - val_acc: 0.8724
    Epoch 4/30
    468/469 [============================>.] - ETA: 0s - loss: 0.3037 - acc: 0.8868
    Epoch 00004: val_acc did not improve from 0.87240
    469/469 [==============================] - 6s 13ms/step - loss: 0.3036 - acc: 0.8868 - val_loss: 0.3707 - val_acc: 0.8645
    Epoch 5/30
    468/469 [============================>.] - ETA: 0s - loss: 0.2843 - acc: 0.8940
    Epoch 00005: val_acc improved from 0.87240 to 0.87760, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.2843 - acc: 0.8940 - val_loss: 0.3426 - val_acc: 0.8776
    Epoch 6/30
    465/469 [============================>.] - ETA: 0s - loss: 0.2697 - acc: 0.8977
    Epoch 00006: val_acc improved from 0.87760 to 0.88080, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.2696 - acc: 0.8978 - val_loss: 0.3381 - val_acc: 0.8808
    Epoch 7/30
    469/469 [==============================] - ETA: 0s - loss: 0.2559 - acc: 0.9041
    Epoch 00007: val_acc did not improve from 0.88080
    469/469 [==============================] - 6s 13ms/step - loss: 0.2559 - acc: 0.9041 - val_loss: 0.3299 - val_acc: 0.8808
    Epoch 8/30
    465/469 [============================>.] - ETA: 0s - loss: 0.2459 - acc: 0.9080
    Epoch 00008: val_acc did not improve from 0.88080
    469/469 [==============================] - 6s 13ms/step - loss: 0.2457 - acc: 0.9080 - val_loss: 0.3393 - val_acc: 0.8790
    Epoch 9/30
    465/469 [============================>.] - ETA: 0s - loss: 0.2349 - acc: 0.9106
    Epoch 00009: val_acc improved from 0.88080 to 0.88260, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.2348 - acc: 0.9106 - val_loss: 0.3469 - val_acc: 0.8826
    Epoch 10/30
    469/469 [==============================] - ETA: 0s - loss: 0.2258 - acc: 0.9144
    Epoch 00010: val_acc did not improve from 0.88260
    469/469 [==============================] - 6s 13ms/step - loss: 0.2258 - acc: 0.9144 - val_loss: 0.3559 - val_acc: 0.8777
    Epoch 11/30
    465/469 [============================>.] - ETA: 0s - loss: 0.2163 - acc: 0.9173
    Epoch 00011: val_acc did not improve from 0.88260
    469/469 [==============================] - 6s 12ms/step - loss: 0.2162 - acc: 0.9173 - val_loss: 0.3593 - val_acc: 0.8806
    Epoch 12/30
    466/469 [============================>.] - ETA: 0s - loss: 0.2094 - acc: 0.9199
    Epoch 00012: val_acc improved from 0.88260 to 0.88770, saving model to best_model.h5
    469/469 [==============================] - 6s 12ms/step - loss: 0.2097 - acc: 0.9197 - val_loss: 0.3284 - val_acc: 0.8877
    Epoch 13/30
    468/469 [============================>.] - ETA: 0s - loss: 0.2022 - acc: 0.9236
    Epoch 00013: val_acc did not improve from 0.88770
    469/469 [==============================] - 6s 12ms/step - loss: 0.2021 - acc: 0.9237 - val_loss: 0.3590 - val_acc: 0.8755
    Epoch 14/30
    469/469 [==============================] - ETA: 0s - loss: 0.1958 - acc: 0.9249
    Epoch 00014: val_acc improved from 0.88770 to 0.88940, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.1958 - acc: 0.9249 - val_loss: 0.3363 - val_acc: 0.8894
    Epoch 15/30
    469/469 [==============================] - ETA: 0s - loss: 0.1877 - acc: 0.9275
    Epoch 00015: val_acc improved from 0.88940 to 0.89100, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.1877 - acc: 0.9275 - val_loss: 0.3551 - val_acc: 0.8910
    Epoch 16/30
    465/469 [============================>.] - ETA: 0s - loss: 0.1827 - acc: 0.9290
    Epoch 00016: val_acc did not improve from 0.89100
    469/469 [==============================] - 6s 12ms/step - loss: 0.1824 - acc: 0.9291 - val_loss: 0.3445 - val_acc: 0.8892
    Epoch 17/30
    468/469 [============================>.] - ETA: 0s - loss: 0.1723 - acc: 0.9347
    Epoch 00017: val_acc improved from 0.89100 to 0.89270, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.1722 - acc: 0.9348 - val_loss: 0.3626 - val_acc: 0.8927
    Epoch 18/30
    468/469 [============================>.] - ETA: 0s - loss: 0.1694 - acc: 0.9347
    Epoch 00018: val_acc did not improve from 0.89270
    469/469 [==============================] - 6s 13ms/step - loss: 0.1695 - acc: 0.9346 - val_loss: 0.3553 - val_acc: 0.8902
    Epoch 19/30
    468/469 [============================>.] - ETA: 0s - loss: 0.1643 - acc: 0.9365
    Epoch 00019: val_acc improved from 0.89270 to 0.89700, saving model to best_model.h5
    469/469 [==============================] - 6s 13ms/step - loss: 0.1644 - acc: 0.9365 - val_loss: 0.3292 - val_acc: 0.8970
    Epoch 20/30
    469/469 [==============================] - ETA: 0s - loss: 0.1555 - acc: 0.9402
    Epoch 00020: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 13ms/step - loss: 0.1555 - acc: 0.9402 - val_loss: 0.3701 - val_acc: 0.8896
    Epoch 21/30
    465/469 [============================>.] - ETA: 0s - loss: 0.1529 - acc: 0.9412
    Epoch 00021: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1530 - acc: 0.9412 - val_loss: 0.3416 - val_acc: 0.8934
    Epoch 22/30
    468/469 [============================>.] - ETA: 0s - loss: 0.1486 - acc: 0.9425
    Epoch 00022: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1485 - acc: 0.9425 - val_loss: 0.3755 - val_acc: 0.8934
    Epoch 23/30
    469/469 [==============================] - ETA: 0s - loss: 0.1437 - acc: 0.9448
    Epoch 00023: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1437 - acc: 0.9448 - val_loss: 0.3729 - val_acc: 0.8929
    Epoch 24/30
    469/469 [==============================] - ETA: 0s - loss: 0.1370 - acc: 0.9465
    Epoch 00024: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1370 - acc: 0.9465 - val_loss: 0.3760 - val_acc: 0.8940
    Epoch 25/30
    467/469 [============================>.] - ETA: 0s - loss: 0.1363 - acc: 0.9472
    Epoch 00025: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1361 - acc: 0.9473 - val_loss: 0.3934 - val_acc: 0.8910
    Epoch 26/30
    467/469 [============================>.] - ETA: 0s - loss: 0.1321 - acc: 0.9485
    Epoch 00026: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1320 - acc: 0.9486 - val_loss: 0.4142 - val_acc: 0.8935
    Epoch 27/30
    466/469 [============================>.] - ETA: 0s - loss: 0.1268 - acc: 0.9506
    Epoch 00027: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1271 - acc: 0.9506 - val_loss: 0.4410 - val_acc: 0.8886
    Epoch 28/30
    468/469 [============================>.] - ETA: 0s - loss: 0.1261 - acc: 0.9522
    Epoch 00028: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1261 - acc: 0.9522 - val_loss: 0.4092 - val_acc: 0.8909
    Epoch 29/30
    466/469 [============================>.] - ETA: 0s - loss: 0.1182 - acc: 0.9545
    Epoch 00029: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1185 - acc: 0.9544 - val_loss: 0.3903 - val_acc: 0.8908
    Epoch 30/30
    465/469 [============================>.] - ETA: 0s - loss: 0.1153 - acc: 0.9554
    Epoch 00030: val_acc did not improve from 0.89700
    469/469 [==============================] - 6s 12ms/step - loss: 0.1153 - acc: 0.9554 - val_loss: 0.4489 - val_acc: 0.8885
    


```python
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_3 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    dense_6 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    dense_7 (Dense)              (None, 128)               32896     
    _________________________________________________________________
    dense_8 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dense_9 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 575,050
    Trainable params: 575,050
    Non-trainable params: 0
    _________________________________________________________________
    
