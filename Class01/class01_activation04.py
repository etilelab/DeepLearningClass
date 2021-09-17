
import tensorflow as tf
from tensorflow.keras.layers import Dense


# input data, 8 x 10 8행 10열, 8개의 row vector 10개의 칼럼(특징값)
# W 는 10개, B 는 1개 생성 된다.
N, n_feature = 8, 10

x = tf.random.normal(shape=(N, n_feature)) # generate minibatch
print(x.shape)

dense = Dense(units=1, activation='relu')
y = dense(x)

W, B = dense.get_weights()

print("Shape of x : ", x.shape)
print("Shape of W : ", W.shape)
print("Shape of B : ", B.shape)

