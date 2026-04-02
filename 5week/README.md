1번 과제: 간단한 이미지 분류기 구현
1. 과제 개요

이번 과제는 손글씨 숫자 이미지 데이터셋인 MNIST를 이용하여 간단한 이미지 분류기를 구현하는 것이다.
MNIST는 0부터 9까지의 손글씨 숫자 이미지로 이루어진 대표적인 이미지 분류 데이터셋이며, 컴퓨터 비전과 딥러닝 입문에서 가장 많이 사용되는 기초 데이터셋 중 하나이다.

이번 실습에서는 MNIST 데이터를 불러온 뒤, 훈련 데이터와 테스트 데이터를 사용하여 간단한 신경망 모델을 학습시키고, 최종적으로 테스트 정확도를 확인하였다. 과제 요구사항 역시 MNIST 데이터셋을 로드하고, 간단한 신경망을 구성한 뒤 모델을 훈련시키고 정확도를 평가하는 것이다.

2. 배경지식
2-1. MNIST 데이터셋

MNIST는 손글씨 숫자 이미지 데이터셋으로, 각 이미지는 28×28 크기의 흑백 이미지이다.
각 이미지에는 0~9 사이의 숫자 하나가 들어 있으며, 이미지 분류 문제의 가장 기본적인 예제로 자주 사용된다. 과제 자료에서도 손글씨 숫자 이미지는 28×28 픽셀 크기의 흑백 이미지라고 제시되어 있다.

2-2. 이미지 정규화

원본 이미지의 픽셀 값은 보통 0-255 범위를 가진다.
이를 그대로 사용하면 학습이 비효율적일 수 있기 때문에, 보통 255로 나누어 0-1 범위로 정규화한 뒤 학습에 사용한다.
이 과정은 학습 안정성을 높이고, 신경망이 더 빠르고 안정적으로 수렴하도록 돕는다.

2-3. 인공신경망(ANN)

인공신경망은 입력값을 받아 여러 층을 거치며 특징을 학습하고, 마지막에 어떤 클래스인지 예측하는 모델이다.
이번 과제에서는 CNN처럼 복잡한 구조가 아니라, 입력층 → 은닉층 → 출력층으로 이루어진 간단한 완전연결 신경망을 사용하였다.

2-4. 활성화 함수
ReLU: 은닉층에서 사용되며, 음수는 0으로 만들고 양수는 그대로 전달한다. 학습 속도가 빠르고 자주 사용된다.
Softmax: 출력층에서 사용되며, 10개의 숫자 클래스 각각에 대한 확률을 계산한다.
2-5. 손실 함수와 정확도
Loss(손실값): 모델의 예측이 실제 정답과 얼마나 차이가 나는지를 수치로 나타낸 값이다.
Accuracy(정확도): 전체 데이터 중 맞춘 비율을 의미한다.

이번 과제에서는 손실 함수로 sparse_categorical_crossentropy를 사용했는데, 이는 정답 라벨이 0~9 같은 숫자 형태일 때 다중 분류 문제에서 자주 사용된다.

3. 사용한 주요 알고리즘 및 구조 설명
3-1. Sequential 모델

Sequential은 층을 순서대로 쌓아 올리는 가장 기본적인 모델 구성 방식이다.
이번 과제처럼 구조가 단순한 신경망을 만들 때 사용하기 적합하다. 과제 힌트에서도 Sequential 모델을 활용하여 신경망을 구성하라고 제시되어 있다.

3-2. Flatten

MNIST 이미지는 28×28의 2차원 배열 형태이다.
하지만 완전연결층(Dense)은 1차원 벡터를 입력으로 받기 때문에, 먼저 Flatten을 사용하여 28×28 이미지를 784차원 벡터로 펼쳐 준다.

3-3. Dense

Dense는 완전연결층을 의미한다.
이번 코드에서는

첫 번째 Dense 층: 128개의 뉴런, ReLU 활성화 함수 사용
두 번째 Dense 층: 10개의 뉴런, Softmax 활성화 함수 사용

으로 구성하였다.

즉, 입력 이미지를 펼친 뒤 128개의 특징 표현으로 변환하고, 마지막에 10개의 숫자 클래스 중 하나로 분류하는 구조이다.

3-4. Adam Optimizer

adam은 신경망 학습에서 가장 널리 사용되는 최적화 알고리즘 중 하나이다.
가중치를 자동으로 효율적으로 조정해 주기 때문에, 초보자가 사용하기에도 편리하고 성능도 안정적이다.

4. 주요 코드 설명
4-1. 데이터셋 불러오기
```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```
MNIST 데이터셋을 불러오는 부분이다.
훈련용 데이터와 테스트용 데이터를 각각 나누어 가져온다.

4-2. 데이터 크기 확인
```
print("훈련 데이터 크기:", x_train.shape)
print("훈련 라벨 크기:", y_train.shape)
print("테스트 데이터 크기:", x_test.shape)
print("테스트 라벨 크기:", y_test.shape)
```
불러온 데이터의 크기를 확인하는 부분이다.
일반적으로 MNIST는 훈련 이미지 60000장, 테스트 이미지 10000장으로 구성된다.

4-3. 정규화
```
x_train = x_train / 255.0
x_test = x_test / 255.0
```
이미지 픽셀 값을 0-1 범위로 변환하는 과정이다.
학습 효율을 높이기 위해 수행한다.

4-4. 모델 구성
```
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```
간단한 신경망 모델을 정의한 부분이다.

Flatten(input_shape=(28, 28)): 28×28 이미지를 1차원으로 펼침
Dense(128, activation='relu'): 은닉층
Dense(10, activation='softmax'): 10개 숫자 클래스로 분류하는 출력층

4-5. 모델 컴파일
```
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
학습 방법을 설정하는 부분이다.

optimizer='adam': 가중치 업데이트 방식
loss='sparse_categorical_crossentropy': 다중 분류용 손실 함수
metrics=['accuracy']: 학습 중 정확도 출력

4-6. 모델 학습
```
model.fit(x_train, y_train, epochs=5)
```
훈련 데이터를 사용하여 모델을 학습하는 부분이다.
여기서는 전체 훈련 데이터를 5번 반복 학습하였다.

4-7. 모델 평가
```
loss, accuracy = model.evaluate(x_test, y_test)
```
학습이 끝난 뒤 테스트 데이터로 모델 성능을 평가하는 부분이다.
손실값과 정확도를 함께 확인할 수 있다.

실습이미지
<img width="561" height="217" alt="image" src="https://github.com/user-attachments/assets/7df73f19-64fc-4313-a1d2-10ccf5c9dfae" />

전체 코드
```
import tensorflow as tf  # TensorFlow 라이브러리를 불러옴
from tensorflow.keras.models import Sequential  # 층을 순서대로 쌓는 Sequential 모델을 불러옴
from tensorflow.keras.layers import Flatten, Dense  # Flatten 층과 Dense 층을 불러옴

# MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # MNIST 훈련 데이터와 테스트 데이터를 불러옴

# 데이터 크기 출력
print("훈련 데이터 크기:", x_train.shape)  # 훈련 이미지 데이터의 크기를 출력함
print("훈련 라벨 크기:", y_train.shape)  # 훈련 정답 라벨의 크기를 출력함
print("테스트 데이터 크기:", x_test.shape)  # 테스트 이미지 데이터의 크기를 출력함
print("테스트 라벨 크기:", y_test.shape)  # 테스트 정답 라벨의 크기를 출력함

# 픽셀 값을 0-1 범위로 정규화
x_train = x_train / 255.0  # 훈련 이미지의 픽셀 값을 255로 나누어 0-1 범위로 변환함
x_test = x_test / 255.0  # 테스트 이미지의 픽셀 값을 255로 나누어 0-1 범위로 변환함

# 간단한 신경망 모델 구성
model = Sequential([  # Sequential 방식으로 신경망 모델을 생성함
    Flatten(input_shape=(28, 28)),  # 28x28 형태의 이미지를 1차원 벡터로 펼침
    Dense(128, activation='relu'),  # 128개의 뉴런을 가진 은닉층을 만들고 활성화 함수로 ReLU를 사용함
    Dense(10, activation='softmax')  # 10개의 숫자 클래스를 분류하기 위한 출력층을 만들고 Softmax를 사용함
])

# 모델 컴파일
model.compile(  # 모델의 학습 방식을 설정함
    optimizer='adam',  # 최적화 알고리즘으로 Adam을 사용함
    loss='sparse_categorical_crossentropy',  # 다중 클래스 분류 문제에 맞는 손실 함수를 사용함
    metrics=['accuracy']  # 학습 과정에서 정확도를 함께 출력하도록 설정함
)

# 모델 학습
model.fit(x_train, y_train, epochs=5)  # 훈련 데이터를 이용해 모델을 5번 반복 학습시킴

# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)  # 테스트 데이터를 사용해 손실값과 정확도를 평가함

# 결과 출력
print("테스트 손실값:", loss)  # 테스트 데이터에서의 손실값을 출력함
print("테스트 정확도:", accuracy)  # 테스트 데이터에서의 정확도를 출력함
```


2번 과제: CIFAR-10 데이터셋을 활용한 CNN 모델 구축

1. 과제 개요

이번 과제는 CIFAR-10 데이터셋을 이용하여 합성곱 신경망(CNN, Convolutional Neural Network)을 구성하고, 이미지 분류를 수행하는 실습이다.
CIFAR-10은 10개의 클래스(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)로 이루어진 대표적인 컬러 이미지 데이터셋이다. 각 이미지는 32×32 크기의 RGB 컬러 이미지이며, 이미지 분류 문제에서 기본적인 CNN 구조를 실습할 때 자주 사용된다.

이번 실습에서는 CIFAR-10 데이터를 불러와 정규화 전처리를 수행하고, CNN 모델을 설계하여 학습한 뒤, 테스트 데이터셋으로 성능을 평가하였다. 또한 추가로 dog.jpg 파일을 입력하여 학습된 모델이 새로운 이미지에 대해 어떤 클래스로 예측하는지도 확인하였다. 과제 자료에서도 CIFAR-10 로드, 전처리, CNN 설계 및 훈련, 성능 평가, dog.jpg 예측 수행을 요구하고 있다

2. 배경지식
   
2-1. CIFAR-10 데이터셋

CIFAR-10은 총 10개의 클래스에 대해 학습하는 이미지 데이터셋이다.
MNIST가 흑백 손글씨 숫자 데이터셋이었다면, CIFAR-10은 컬러 이미지 분류 문제라는 점에서 더 복잡하고 어려운 데이터셋이다. 이미지 크기가 작고(32×32), 배경과 물체가 함께 들어 있어 단순한 완전연결 신경망보다는 CNN 구조가 더 적합하다.

2-2. 정규화(Normalization)

원본 이미지의 픽셀 값은 0-255 범위를 가진다. 
학습 전에 이를 255로 나누어 0-1 범위로 변환하면 모델이 더 안정적으로 학습할 수 있다. 과제 힌트에서도 데이터 전처리 시 픽셀 값을 0-1 범위로 정규화하면 모델 수렴이 빨라질 수 있다고 제시되어 있다.

2-3. CNN(합성곱 신경망)

CNN은 이미지 처리에 특화된 신경망 구조이다.
일반적인 완전연결 신경망(Dense)과 달리, CNN은 이미지의 공간적 구조를 유지하면서 특징을 추출할 수 있다.
주요 구성 요소는 다음과 같다.
    - Conv2D: 이미지에서 중요한 특징(에지, 모서리, 질감, 패턴 등)을 추출
    - MaxPooling2D: 특징 맵의 크기를 줄여 계산량을 감소시키고 중요한 특징만 유지
    - Flatten: 다차원 특징 맵을 1차원 벡터로 변환
    - Dense: 최종적으로 분류를 수행하는 완전연결층

2-4. 활성화 함수

- ReLU: 합성곱층과 은닉층에서 사용되며, 음수는 0으로 만들고 양수는 그대로 전달한다.
- Softmax: 출력층에서 사용되며, 10개 클래스 각각에 대한 확률을 계산한다.

2-5. 손실 함수와 정확도
- Loss(손실값): 모델의 예측이 실제 정답과 얼마나 차이가 나는지를 수치로 나타낸 값
- Accuracy(정확도): 전체 데이터 중 맞춘 비율

이번 코드에서는 손실 함수로 sparse_categorical_crossentropy를 사용했다. 이는 정답 라벨이 원-핫 인코딩이 아니라 정수 형태일 때 다중 클래스 분류에서 자주 사용되는 손실 함수이다.

3. 사용한 주요 알고리즘 및 모델 구조 설명

3-1. CNN 구조를 사용한 이유

이미지는 단순한 숫자 벡터가 아니라, 가로·세로 위치 관계가 중요한 데이터이다.
CNN은 이 공간 정보를 유지한 상태로 특징을 뽑아낼 수 있기 때문에 CIFAR-10 같은 이미지 데이터셋에 적합하다.

3-2. Conv2D

Conv2D는 입력 이미지 위를 작은 필터(kernel)가 이동하며 특징을 추출하는 층이다.
처음 층에서는 단순한 선이나 에지를 배우고, 층이 깊어질수록 더 복잡한 형태와 패턴을 학습한다.

3-3. MaxPooling2D

MaxPooling2D는 특징 맵의 크기를 줄여 주는 역할을 한다.
중요한 특징만 유지하면서 데이터 크기를 줄여 계산량을 감소시키고, 과적합을 어느 정도 방지하는 데도 도움이 된다.

3-4. Flatten

합성곱층과 풀링층을 거친 결과는 다차원 특징 맵이다.
이를 최종 분류층(Dense)에 넣기 위해 Flatten을 사용하여 1차원 벡터로 펼친다.

3-5. Dense

Dense는 완전연결층이다.
Flatten으로 펼쳐진 특징 정보를 바탕으로 최종 분류를 수행한다.

3-6. Adam Optimizer

adam은 학습 시 가중치를 효율적으로 업데이트해 주는 대표적인 최적화 알고리즘이다.
성능이 안정적이고 사용이 간편하여 기본 실습에서 자주 사용된다.

4. 주요 코드 설명
   
4-1. 라이브러리 불러오기
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
```
CNN 모델 구성, 이미지 전처리, 결과 출력 및 그래프 시각화를 위해 필요한 라이브러리를 불러온다.

4-2. CIFAR-10 클래스 이름 지정
```
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
```

4-3. 데이터셋 불러오기
```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```
TensorFlow에서 제공하는 CIFAR-10 데이터셋을 불러온다.
과제 힌트에서도 tensorflow.keras.datasets에서 CIFAR-10을 불러올 수 있다고 제시되어 있다.
CIFAR-10의 10개 클래스 이름을 리스트로 저장한다.
예측 결과를 숫자가 아닌 실제 클래스 이름으로 보기 위해 사용한다.

4-4. 데이터 전처리
```
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```
훈련 데이터와 테스트 데이터의 픽셀 값을 0~1 범위로 정규화한다.
과제 힌트의 “픽셀 값을 0~1 범위로 정규화” 부분을 반영한 코드이다

4-6. CNN 모델 구성
```
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
```
이 모델은 다음과 같은 흐름으로 구성되어 있다.

1. 입력 이미지(32×32×3)를 받음
2. Conv2D + MaxPooling2D를 반복하여 특징을 추출하고 크기를 줄임
3. 마지막에 Flatten으로 펼침
4. Dense 층을 거쳐 최종적으로 10개 클래스 중 하나로 분류함

4-7. 모델 컴파일
```
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
학습 방식을 설정하는 부분이다.

- adam: 최적화 알고리즘
- sparse_categorical_crossentropy: 다중 분류용 손실 함수
- accuracy: 정확도 출력

4-8. 모델 학습
```
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)
```
4-9. dog.jpg 예측
```
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = img_array.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)
```
dog.jpg를 불러와 CIFAR-10 입력 크기와 동일한 32×32로 조정한 뒤, 예측에 맞게 배열 형태로 변환하고 정규화한다.
훈련 데이터를 사용해 모델을 10번 반복 학습한다.
또한 검증 데이터로 테스트 데이터를 함께 넣어, 학습 과정에서 정확도와 손실의 변화를 확인할 수 있도록 하였다.
```
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
predicted_label = class_names[predicted_class]
```
모델이 dog.jpg에 대해 예측한 확률 중 가장 큰 값을 가지는 클래스를 선택한다.
```
plt.imshow(img)
plt.title(f"Prediction: {predicted_label}")
```
예측한 결과를 이미지와 함께 시각적으로 확인할 수 있도록 출력한다.

추가로 10번 돌려본 결과 9번 : 개 1번 고양이로 나왔다

실습이미지 1
<img width="317" height="67" alt="image" src="https://github.com/user-attachments/assets/5e58baee-7f1c-4160-9a3b-028fcf69fcbe" />

실습이미지 2
<img width="367" height="384" alt="image" src="https://github.com/user-attachments/assets/192dfcbc-5dcb-4d3f-a6af-a2ec0c09a6e4" />

실습이미지 3
<img width="795" height="504" alt="image" src="https://github.com/user-attachments/assets/035c7634-9b56-4975-9472-b0469a904f9f" />

실습이미지 4
<img width="745" height="471" alt="image" src="https://github.com/user-attachments/assets/7a86cf42-6913-45bc-acaa-7197ebbeda30" />

전체코드
```
import tensorflow as tf  # TensorFlow 라이브러리를 불러옴
from tensorflow.keras.models import Sequential  # 층을 순서대로 쌓는 Sequential 모델을 불러옴
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input  # CNN 구성에 필요한 층들을 불러옴
import numpy as np  # 배열 처리와 차원 변환을 위해 NumPy를 불러옴
from tensorflow.keras.preprocessing import image  # 이미지 파일을 불러오고 배열로 바꾸기 위한 모듈을 불러옴
import matplotlib.pyplot as plt  # 이미지와 그래프를 출력하기 위해 matplotlib을 불러옴

# CIFAR-10 클래스 이름
class_names = [  # CIFAR-10 데이터셋의 10개 클래스 이름을 리스트로 저장함
    'airplane', 'automobile', 'bird', 'cat', 'deer',  # 비행기, 자동차, 새, 고양이, 사슴 클래스
    'dog', 'frog', 'horse', 'ship', 'truck'  # 개, 개구리, 말, 배, 트럭 클래스
]
plt.imshow(img)
plt.title(f"Prediction: {predicted_label}")

예측한 결과를 이미지와 함께 시각적으로 확인할 수 있도록 출력한다.

# CIFAR-10 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # CIFAR-10의 훈련 데이터와 테스트 데이터를 불러옴

# 데이터 전처리: 픽셀 값을 0~1 범위로 정규화
x_train = x_train.astype("float32") / 255.0  # 훈련 이미지의 픽셀 값을 실수형으로 바꾼 뒤 255로 나누어 0~1 범위로 정규화함
x_test = x_test.astype("float32") / 255.0  # 테스트 이미지의 픽셀 값을 실수형으로 바꾼 뒤 255로 나누어 0~1 범위로 정규화함

# CIFAR-10 샘플 이미지 출력
plt.figure(figsize=(10, 10))  # 샘플 이미지를 여러 장 보기 위해 10x10 크기의 출력 창을 생성함
for i in range(9):  # 처음 9장의 이미지를 반복해서 출력함
    plt.subplot(3, 3, i + 1)  # 3행 3열 형태의 서브플롯 중 하나를 선택함
    plt.imshow(x_train[i])  # 현재 훈련 이미지를 화면에 출력함
    plt.title(class_names[int(y_train[i][0])])  # 현재 이미지의 정답 라벨을 제목으로 표시함
    plt.axis("off")  # 축 눈금을 보이지 않게 설정함
plt.suptitle("CIFAR-10 Sample Images")  # 전체 샘플 이미지 출력 창의 제목을 설정함
plt.tight_layout()  # 그래프 간격이 겹치지 않도록 자동으로 정리함
plt.show()  # 샘플 이미지를 화면에 출력함

# CNN 모델 구성
model = Sequential([  # Sequential 방식으로 CNN 모델을 생성함
    Input(shape=(32, 32, 3)),  # 입력 이미지 크기가 32x32이고 RGB 3채널임을 지정함
    Conv2D(32, (3, 3), activation='relu'),  # 32개의 필터를 가진 3x3 합성곱 층을 추가하고 활성화 함수로 ReLU를 사용함
    MaxPooling2D((2, 2)),  # 2x2 최대 풀링을 적용하여 특징 맵의 크기를 줄임

    Conv2D(64, (3, 3), activation='relu'),  # 64개의 필터를 가진 두 번째 합성곱 층을 추가함
    MaxPooling2D((2, 2)),  # 다시 2x2 최대 풀링을 적용하여 크기를 줄임
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),  # 128개의 필터를 가진 세 번째 합성곱 층을 추가하고 패딩을 same으로 설정함
    MaxPooling2D((2, 2)),  # 세 번째 최대 풀링을 적용하여 특징 맵을 더 압축함

    Flatten(),  # 다차원 특징 맵을 1차원 벡터로 펼침
    Dense(64, activation='relu'),  # 64개의 뉴런을 가진 완전연결층을 추가하고 ReLU를 사용함
    Dense(10, activation='softmax')  # 10개 클래스 분류를 위한 출력층을 추가하고 Softmax를 사용함
])

# 모델 컴파일
model.compile(  # 모델의 학습 방식을 설정함
    optimizer='adam',  # 최적화 알고리즘으로 Adam을 사용함
    loss='sparse_categorical_crossentropy',  # 다중 클래스 분류 문제에 적합한 손실 함수를 사용함
    metrics=['accuracy']  # 학습 중 정확도를 함께 출력하도록 설정함
)

# 모델 학습
history = model.fit(  # 모델 학습 결과를 history 변수에 저장함
    x_train, y_train,  # 훈련 이미지와 훈련 라벨을 학습 데이터로 사용함
    epochs=10,  # 전체 데이터를 10번 반복 학습함
    validation_data=(x_test, y_test)  # 테스트 데이터를 검증 데이터로 사용하여 매 epoch마다 성능을 확인함
)

# 모델 성능 평가
loss, accuracy = model.evaluate(x_test, y_test)  # 테스트 데이터를 사용해 최종 손실값과 정확도를 계산함
print(f"테스트 손실값: {loss:.4f}")  # 계산된 테스트 손실값을 소수점 넷째 자리까지 출력함
print(f"테스트 정확도: {accuracy * 100:.2f}%")  # 계산된 테스트 정확도를 퍼센트 형태로 출력함

# 학습 정확도 그래프 출력
plt.figure(figsize=(8, 5))  # 정확도 그래프를 그리기 위한 출력 창을 생성함
plt.plot(history.history['accuracy'], label='Train Accuracy')  # 훈련 데이터 정확도 변화를 그래프로 그림
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # 검증 데이터 정확도 변화를 그래프로 그림
plt.title('Training and Validation Accuracy')  # 그래프 제목을 설정함
plt.xlabel('Epoch')  # x축 이름을 Epoch로 설정함
plt.ylabel('Accuracy')  # y축 이름을 Accuracy로 설정함
plt.legend()  # 그래프 범례를 표시함
plt.grid(True)  # 그래프에 격자선을 표시함
plt.show()  # 정확도 그래프를 화면에 출력함

# 학습 손실 그래프 출력
plt.figure(figsize=(8, 5))  # 손실 그래프를 그리기 위한 출력 창을 생성함
plt.plot(history.history['loss'], label='Train Loss')  # 훈련 데이터 손실값 변화를 그래프로 그림
plt.plot(history.history['val_loss'], label='Validation Loss')  # 검증 데이터 손실값 변화를 그래프로 그림
plt.title('Training and Validation Loss')  # 그래프 제목을 설정함
plt.xlabel('Epoch')  # x축 이름을 Epoch로 설정함
plt.ylabel('Loss')  # y축 이름을 Loss로 설정함
plt.legend()  # 그래프 범례를 표시함
plt.grid(True)  # 그래프에 격자선을 표시함
plt.show()  # 손실 그래프를 화면에 출력함

# dog.jpg 불러오기
img_path = "dog.jpg"  # 예측할 이미지 파일 이름을 변수에 저장함
img = image.load_img(img_path, target_size=(32, 32))  # dog.jpg를 불러오고 CIFAR-10 입력 크기에 맞게 32x32로 조정함
img_array = image.img_to_array(img)  # 불러온 이미지를 NumPy 배열 형태로 변환함

# dog.jpg 전처리
img_array = img_array.astype("float32") / 255.0  # 예측용 이미지의 픽셀 값을 실수형으로 바꾸고 0~1 범위로 정규화함
img_array = np.expand_dims(img_array, axis=0)  # 모델 입력 형태에 맞추기 위해 배치 차원을 추가함

# dog.jpg 예측
prediction = model.predict(img_array)  # 전처리한 dog.jpg를 모델에 넣어 예측 결과를 얻음
predicted_class = np.argmax(prediction)  # 가장 확률이 높은 클래스의 인덱스를 선택함
predicted_label = class_names[predicted_class]  # 선택된 인덱스를 실제 클래스 이름으로 변환함

print("dog.jpg 예측 결과:", predicted_label)  # dog.jpg가 어떤 클래스로 예측되었는지 출력함
print("각 클래스 확률:", prediction)  # 10개 클래스에 대한 예측 확률을 모두 출력함

# dog.jpg 출력
plt.figure(figsize=(4, 4))  # 예측한 dog.jpg를 보기 위한 출력 창을 생성함
plt.imshow(img)  # 불러온 dog.jpg 이미지를 화면에 출력함
plt.title(f"Prediction: {predicted_label}")  # 예측 결과를 이미지 제목으로 표시함
plt.axis("off")  # 축 눈금을 보이지 않게 설정함
plt.show()  # dog.jpg 이미지를 화면에 출력함
```


