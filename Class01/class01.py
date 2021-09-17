
import tensorflow as tf
from tensorflow.keras.layers import Dense

x = tf.constant([[10.]]) # column vector 로 만들어 줘야함, input setting
print(x)
dense = Dense(units=1, activation='linear') # imp. an affine function, linear x와 y값이 똑같이 해준다.

# dense 통과전 W, B출력시 오류가 나타나게 된다.
# W, B = dense.get_weights()
# print(W,B)

y_tf = dense(x)  # forward propagation + params init, X값이 통과할때 W와 B가 만들어진다.
print(y_tf)

W, B = dense.get_weights()  # get weight, bias
print(W,B)

