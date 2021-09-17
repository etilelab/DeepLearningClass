import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant

x = tf.constant([[10.]]) # row vector, column 형태가 되어야함으로 !
# weight/bias setting
w,b = tf.constant(10.), tf.constant(20.) # .써주는거 습관
w_init, b_init = Constant(w), Constant(b)

print(w_init, b_init)  # 실제 텐서 값을 가지고있는것 아님, w와 b를 초기화시켜주는 object 만 생성

# imp. an affine function
# W와 B를 우리가 원하는 값으로 세팅
dense = Dense(units=1, activation='linear',
              kernel_initializer=w_init,
              bias_initializer=b_init)

y_tf = dense(x)
print(y_tf) # 10 * 10 + 20 = 120

W, B = dense.get_weights()

print("W: {} {}\n".format(W.shape, W))  # 우리가 원하는 값으로 초기화 됨을 알 수 있음
print("B: {} {}\n".format(B.shape, B))
