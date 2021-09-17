
# Dense Layer 에서 어떤 일이 일어나는가 ?

import tensorflow as tf
from tensorflow.keras.layers import Dense

x = tf.constant([[10.]]) # column vector 로 만들어 줘야함, input setting -> matrix, X값 설정
print(x)
dense = Dense(units=1, activation='linear') # imp. an affine function, linear x와 y값이 똑같이 해준다.

# Dense(8, input_dim=4, init='uniform', activation='relu'))
# 첫번째 인자 : 출력 뉴런의 수를 설정합니다.
# input_dim : 입력 뉴런의 수를 설정합니다.
# init : 가중치 초기화 방법 설정합니다.
# ‘uniform’ : 균일 분포
# ‘normal’ : 가우시안 분포
# activation : 활성화 함수 설정합니다.
# ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
# ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
# ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
# ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

# dense 통과전 W, B 출력시 오류가 나타나게 된다.
# W, B = dense.get_weights()
# print(W,B)

y_tf = dense(x)  # forward propagation + params init, X값이 통과할때 W와 B가 임의로 만들어진다.
print(y_tf)

W, B = dense.get_weights()  # get weight, bias
print(W,B)

y_man = tf.linalg.matmul(x,W) + B  # Forward propagation(manual)

print(y_tf.shape, y_tf.numpy())
print(y_man.shape, y_man.numpy())  # dense 를 통과한 값과 직접 계산한 Wx + b 의 값과 동일함을 알 수 있다.

