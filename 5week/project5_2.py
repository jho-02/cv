import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# CIFAR-10 클래스 이름
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 데이터 전처리: 픽셀 값을 0~1 범위로 정규화
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CIFAR-10 샘플 이미지 출력
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i][0])])
    plt.axis("off")
plt.suptitle("CIFAR-10 Sample Images")
plt.tight_layout()
plt.show()

# CNN 모델 구성
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

# 모델 성능 평가
loss, accuracy = model.evaluate(x_test, y_test)
print(f"테스트 손실값: {loss:.4f}")
print(f"테스트 정확도: {accuracy * 100:.2f}%")

# 학습 정확도 그래프 출력
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 학습 손실 그래프 출력
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# dog.jpg 불러오기
img_path = "dog.jpg"
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img)

# dog.jpg 전처리
img_array = img_array.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)

# dog.jpg 예측
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
predicted_label = class_names[predicted_class]

print("dog.jpg 예측 결과:", predicted_label)
print("각 클래스 확률:", prediction)

# dog.jpg 출력
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(f"Prediction: {predicted_label}")
plt.axis("off")
plt.show()