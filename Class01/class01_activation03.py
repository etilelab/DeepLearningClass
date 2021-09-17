
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow._api.v2.math import exp, maximum

activation = 'sigmoid'

x = tf.random.uniform(shape=(1,10))

dense = Dense(units=1, activation=activation)

y_tf = dense(x)
W, B = dense.get_weights()

# calculation activation value manually
y_man = tf.linalg.matmul(x, W) + B
if activation == "sigmoid":
    y_man = 1/(1 + exp(-y_man))
elif activation == "tanh":
    y_man = (exp(y_man) - exp(-y_man))/(exp(y_man) + exp(-y_man))
elif activation == "relu":
    y_man = maximum(x, 0)