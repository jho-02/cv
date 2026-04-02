import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 크기 출력
print("훈련 데이터 크기:", x_train.shape)
print("훈련 라벨 크기:", y_train.shape)
print("테스트 데이터 크기:", x_test.shape)
print("테스트 라벨 크기:", y_test.shape)

# 픽셀 값을 0~1 범위로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 간단한 신경망 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
model.fit(x_train, y_train, epochs=5)

# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)

# 결과 출력
print("테스트 손실값:", loss)
print("테스트 정확도:", accuracy)