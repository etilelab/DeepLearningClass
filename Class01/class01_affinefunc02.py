import tensorflow as tf
from tensorflow.keras.layers import Dense

x = tf.random.uniform(shape=(1,10), minval=0, maxval=10) # [ 공부량, 수면량, 휴대폰사용량 ... 10개]
print(x.shape,'\n',x)

dense = Dense(units=1)  # W를 몇개 만들어야 되는지 모르는 상태

y_tf = dense(x) # W 10개, B1개 초기화해야되는구나 !
W, B = dense.get_weights()  # 초기화
y_man = tf.linalg.matmul(x,W) + B

print(W)  # W 10개
print(y_tf)
print(y_man)
