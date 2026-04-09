SORT 알고리즘을 활용한 다중 객체 추적기 구현
1. 과제 개요

이번 과제는 비디오에서 객체를 검출한 뒤, SORT(Simple Online and Realtime Tracking) 알고리즘을 이용하여 각 객체를 프레임 간 연속적으로 추적하는 프로그램을 구현하는 것이다.
과제에서는 YOLOv3와 같은 사전 학습된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출하고, 검출된 경계 상자를 SORT 추적기에 입력하여 객체를 추적하며, 최종적으로 각 객체에 고유 ID와 경계 상자를 부여하여 영상에 실시간으로 시각화하도록 제시되어 있다.

본 구현에서는 OpenCV DNN 모듈을 이용하여 YOLOv3 기반 객체 검출을 수행하고, 칼만 필터와 Hungarian Algorithm을 이용한 SORT 추적기를 직접 구성하여 사람, 자동차, 버스, 트럭, 오토바이, 자전거를 추적하도록 구현하였다. 사용한 전체 코드는 제출 파일 project6_1.py에 포함되어 있다.

2. 배경지식
2-1. 객체 검출(Object Detection)

객체 검출은 이미지나 비디오에서 물체가 어디에 있는지 찾고, 해당 물체가 어떤 클래스에 속하는지를 예측하는 작업이다.
단순 분류가 이미지 전체가 어떤 대상인지 판단하는 것이라면, 객체 검출은 영상 안에서 여러 물체를 각각 구분하여 위치와 클래스를 함께 출력한다.
이번 과제에서는 각 프레임에서 사람, 자동차, 버스, 트럭 등 여러 객체를 찾아야 하므로 객체 검출 과정이 필수적이다.

2-2. YOLOv3

YOLO(You Only Look Once)는 대표적인 실시간 객체 검출 알고리즘이다.
이미지를 한 번의 forward 연산으로 처리하여 객체의 위치와 클래스를 동시에 예측하므로 속도가 빠르며, 실시간 영상 처리에 자주 사용된다.
이번 과제에서는 YOLOv3를 사용하여 각 프레임에서 객체를 검출하였으며, 과제 자료에서도 YOLOv3와 같은 사전 학습된 객체 검출 모델을 사용하라고 제시하고 있다.

2-3. 다중 객체 추적(Multiple Object Tracking)

다중 객체 추적은 비디오에서 여러 객체를 프레임 간 연결하여, 같은 객체가 이동하더라도 동일한 ID를 유지하도록 하는 작업이다.
객체 검출만 수행하면 매 프레임마다 객체를 새로 찾는 데 그치지만, 추적을 적용하면 “이전 프레임의 사람 1번”과 “현재 프레임의 사람 1번”이 같은 객체임을 연결할 수 있다.
이번 과제의 핵심은 검출된 객체를 단순히 표시하는 것에서 끝나는 것이 아니라, 각 객체에 고유 ID를 부여하여 지속적으로 추적하는 것이다.

2-4. SORT 알고리즘

SORT는 Simple Online and Realtime Tracking의 약자로, 비교적 단순한 구조이지만 빠르게 동작하는 다중 객체 추적 알고리즘이다.
SORT의 핵심은 다음 두 가지이다.

칼만 필터(Kalman Filter) : 이전 상태를 기반으로 다음 프레임에서 객체 위치를 예측
Hungarian Algorithm : 현재 프레임의 검출 결과와 기존 추적 객체를 최적으로 매칭

과제 자료에서도 SORT가 칼만 필터와 헝가리안 알고리즘을 사용한다고 제시되어 있다.

2-5. IoU(Intersection over Union)

IoU는 두 개의 경계 상자가 얼마나 겹치는지를 수치로 나타내는 값이다.
값이 1에 가까울수록 두 박스가 많이 겹친다는 뜻이고, 0에 가까울수록 거의 겹치지 않는다는 뜻이다.
이번 구현에서는 검출된 박스와 예측된 박스 간의 IoU를 계산하여 같은 객체인지 판단하는 기준으로 사용하였다.

2-6. NMS(Non-Maximum Suppression)

객체 검출 과정에서는 같은 물체에 대해 여러 개의 중복 박스가 생성될 수 있다.
NMS는 이 중에서 confidence가 가장 높은 박스만 남기고 나머지를 제거하는 기법이다.
이번 과제에서도 YOLOv3 출력 결과에 대해 NMS를 적용하여 중복 검출을 줄였다.

3. 사용한 주요 알고리즘 및 구성 요소
3-1. YOLOv3 객체 검출

YOLOv3는 하나의 신경망으로 이미지 전체를 한 번에 처리하여 객체의 위치와 클래스를 예측하는 객체 검출 알고리즘이다.
본 구현에서는 OpenCV의 cv.dnn.readNetFromDarknet() 함수를 이용하여 yolov3.cfg와 yolov3.weights 파일을 불러오고, 각 프레임에서 객체를 검출하였다.
이후 confidence threshold와 NMS(Non-Maximum Suppression)를 적용하여 중복 박스를 제거하였다. 코드에서도 OpenCV DNN을 통해 YOLOv3 네트워크를 불러오고, blobFromImage, forward, NMSBoxes를 사용하는 흐름으로 구성하였다.

3-2. SORT 추적기

SORT는 검출 기반 다중 객체 추적 알고리즘으로,
객체의 상태 예측에는 칼만 필터(Kalman Filter) 를 사용하고,
현재 프레임의 검출 결과와 이전 추적 객체 간의 매칭에는 Hungarian Algorithm 을 사용한다.
과제 자료에서도 SORT가 칼만 필터와 헝가리안 알고리즘을 사용한다고 제시되어 있다.

본 구현에서는 KalmanBoxTracker 클래스를 정의하여 각 객체의 상태를 관리하고,
associate_detections_to_trackers() 함수에서 IoU 기반 매칭을 수행하였다.
이후 Sort 클래스에서 tracker 생성, 업데이트, 삭제를 처리하도록 구성하였다. 전체 구조는 제출 코드에 포함되어 있다.

3-3. IoU 기반 데이터 연관

프레임 간 동일 객체 여부를 판단하기 위해 IoU(Intersection over Union)를 사용하였다.
검출 박스와 예측 박스의 겹침 정도를 계산한 뒤, Hungarian Algorithm으로 최적 매칭을 수행하였다.
IoU가 일정 threshold 이하인 경우에는 새로운 객체로 판단하거나 기존 tracker를 unmatched 상태로 처리하였다. 이 과정은 iou_batch()와 associate_detections_to_trackers() 함수로 구현하였다.

3-4. 객체 시각화

최종적으로 현재 프레임에서 유효한 tracker에 대해

경계 상자
객체 ID
객체 클래스 이름

을 영상에 표시하였다.
과제 요구사항에서 제시한 “고유 ID와 경계 상자를 프레임에 표시” 부분을 반영한 것이다.

4. 주요 코드 설명
4-1. 추적할 클래스 지정

COCO 클래스 전체 중에서 사람, 자동차, 버스, 트럭, 오토바이, 자전거만 추적 대상으로 설정하였다.
```
TRACK_CLASSES = {"person", "car", "bus", "truck", "motorbike", "bicycle"}
```
불필요한 클래스는 제외하여 과제 영상에서 필요한 객체만 추적하도록 구성하였다.

4-2. YOLOv3 로드

OpenCV DNN 모듈을 사용하여 YOLOv3 설정 파일과 가중치 파일을 불러왔다.
```
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
```
이 부분은 YOLOv3 객체 검출기를 초기화하는 단계이다.

4-3. 비디오 입력

입력 영상 파일을 불러와 프레임 단위로 처리할 수 있도록 설정하였다.
```
video_path = "slow_traffic_small.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("비디오를 열 수 없습니다.")
    exit()
```

4-4. 프레임별 객체 검출

각 프레임에 대해 blob을 생성하고 YOLOv3 forward 연산을 수행하여 객체를 검출하였다.
```
blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)
```
이후 confidence threshold를 적용하고, 클래스가 TRACK_CLASSES에 포함된 경우만 저장하였다. 또한 cv.dnn.NMSBoxes()를 이용해 중복 검출을 제거하였다.

4-5. SORT 추적기 업데이트

YOLOv3 검출 결과를 SORT 추적기에 입력하여 객체 상태를 갱신하였다.
```
tracks, all_trackers = tracker.update(detections, det_class_names)
```
이 과정에서

기존 tracker의 위치 예측
검출 결과와 tracker의 매칭
새로운 tracker 생성
오래된 tracker 제거

가 이루어진다. 구현은 KalmanBoxTracker, Sort, associate_detections_to_trackers()를 중심으로 구성하였다.

4-6. 결과 시각화

현재 프레임에서 유효한 tracker를 대상으로 경계 상자와 ID를 출력하였다.
```
cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.putText(
    frame,
    f"ID {obj_id}: {label}",
    (x1, max(0, y1 - 10)),
    cv.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)
```
이 부분을 통해 영상 내 각 객체가 어떤 ID로 추적되고 있는지 확인할 수 있다.
실습 이미지
<img width="640" height="392" alt="image" src="https://github.com/user-attachments/assets/cab22930-bae9-48a8-89ce-addce6bcfcbc" />


전체 코드
```
import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import numpy as np  # 배열 계산과 수치 연산을 위해 NumPy를 불러옴
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm 매칭을 위해 scipy의 선형 할당 함수를 불러옴
from filterpy.kalman import KalmanFilter  # SORT에서 사용할 칼만 필터 클래스를 불러옴


# COCO 데이터셋에 포함된 객체 클래스 이름 목록을 저장함
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# 실제로 추적할 객체 클래스만 따로 집합 형태로 저장함
TRACK_CLASSES = {"person", "car", "bus", "truck", "motorbike", "bicycle"}


def iou_batch(bb_test, bb_gt):
    """
    두 박스 집합 사이의 IoU를 한 번에 계산하는 함수
    bb_test: Nx4 형태의 박스 배열
    bb_gt: Mx4 형태의 박스 배열
    박스 형식: [x1, y1, x2, y2]
    """
    if len(bb_test) == 0 or len(bb_gt) == 0:  # 둘 중 하나라도 비어 있으면
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)  # 크기에 맞는 0 행렬을 반환함

    bb_test = np.expand_dims(bb_test, 1)  # Nx4 배열을 Nx1x4로 바꿔서 브로드캐스팅 가능하게 만듦
    bb_gt = np.expand_dims(bb_gt, 0)  # Mx4 배열을 1xMx4로 바꿔서 브로드캐스팅 가능하게 만듦

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])  # 교집합 박스의 왼쪽 위 x 좌표를 계산함
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])  # 교집합 박스의 왼쪽 위 y 좌표를 계산함
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])  # 교집합 박스의 오른쪽 아래 x 좌표를 계산함
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])  # 교집합 박스의 오른쪽 아래 y 좌표를 계산함

    w = np.maximum(0., xx2 - xx1)  # 교집합 너비를 계산하되 음수가 나오지 않도록 0과 비교함
    h = np.maximum(0., yy2 - yy1)  # 교집합 높이를 계산하되 음수가 나오지 않도록 0과 비교함
    inter = w * h  # 교집합 면적을 계산함

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])  # 첫 번째 박스들의 면적을 계산함
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])  # 두 번째 박스들의 면적을 계산함

    union = area_test + area_gt - inter  # 합집합 면적을 계산함
    return inter / (union + 1e-6)  # 0으로 나누는 상황을 피하기 위해 작은 값을 더한 뒤 IoU를 반환함


def convert_bbox_to_z(bbox):
    """
    [x1, y1, x2, y2] 형식의 박스를 SORT 상태 표현인 [x, y, s, r]로 변환하는 함수
    x, y: 중심 좌표
    s: 박스 면적
    r: 종횡비
    """
    w = bbox[2] - bbox[0]  # 박스의 너비를 계산함
    h = bbox[3] - bbox[1]  # 박스의 높이를 계산함
    x = bbox[0] + w / 2.  # 중심 x 좌표를 계산함
    y = bbox[1] + h / 2.  # 중심 y 좌표를 계산함
    s = w * h  # 박스 면적을 계산함
    r = w / (h + 1e-6)  # 높이가 0에 가까운 경우를 피하기 위해 작은 값을 더한 뒤 종횡비를 계산함
    return np.array([x, y, s, r]).reshape((4, 1))  # 4x1 벡터 형태로 반환함


def convert_x_to_bbox(x):
    """
    SORT 상태 표현 [x, y, s, r]를 다시 [x1, y1, x2, y2] 형식의 박스로 바꾸는 함수
    """
    w = np.sqrt(x[2] * x[3])  # 면적과 종횡비를 이용하여 박스 너비를 복원함
    h = x[2] / (w + 1e-6)  # 면적을 너비로 나누어 박스 높이를 복원함
    return np.array([  # 좌상단과 우하단 좌표 형태로 박스를 만들어 반환함
        x[0] - w / 2.,
        x[1] - h / 2.,
        x[0] + w / 2.,
        x[1] + h / 2.
    ]).reshape((1, 4))  # 1x4 형태로 반환함


class KalmanBoxTracker:
    count = 0  # 생성된 tracker 개수를 세기 위한 클래스 변수

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 상태 차원 7, 관측 차원 4인 칼만 필터를 생성함

        self.kf.F = np.array([  # 상태 전이 행렬을 설정함
            [1, 0, 0, 0, 1, 0, 0],  # x는 이전 x와 속도 성분의 영향을 받음
            [0, 1, 0, 0, 0, 1, 0],  # y는 이전 y와 속도 성분의 영향을 받음
            [0, 0, 1, 0, 0, 0, 1],  # s는 이전 s와 변화량의 영향을 받음
            [0, 0, 0, 1, 0, 0, 0],  # r은 그대로 유지됨
            [0, 0, 0, 0, 1, 0, 0],  # x 속도 성분
            [0, 0, 0, 0, 0, 1, 0],  # y 속도 성분
            [0, 0, 0, 0, 0, 0, 1]   # s 변화량 성분
        ], dtype=np.float32)

        self.kf.H = np.array([  # 관측 행렬을 설정함
            [1, 0, 0, 0, 0, 0, 0],  # 측정값은 x를 직접 관측함
            [0, 1, 0, 0, 0, 0, 0],  # 측정값은 y를 직접 관측함
            [0, 0, 1, 0, 0, 0, 0],  # 측정값은 s를 직접 관측함
            [0, 0, 0, 1, 0, 0, 0]   # 측정값은 r을 직접 관측함
        ], dtype=np.float32)

        self.kf.R[2:, 2:] *= 10.  # 면적과 종횡비 관측 잡음을 조금 더 크게 설정함
        self.kf.P[4:, 4:] *= 1000.  # 속도 성분의 초기 불확실성을 크게 설정함
        self.kf.P *= 10.  # 전체 오차 공분산을 한 번 더 키워 초기 추정을 느슨하게 잡음
        self.kf.Q[-1, -1] *= 0.01  # 마지막 상태 성분의 프로세스 잡음을 조정함
        self.kf.Q[4:, 4:] *= 0.01  # 속도 관련 프로세스 잡음을 작게 설정함

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # 초기 박스를 칼만 필터 상태 벡터에 넣어 초기화함

        self.time_since_update = 0  # 마지막 업데이트 후 지난 프레임 수를 기록함
        self.id = KalmanBoxTracker.count  # 현재 tracker의 고유 ID를 부여함
        KalmanBoxTracker.count += 1  # 다음 tracker를 위해 count를 증가시킴

        self.hits = 0  # 총 매칭 성공 횟수를 기록함
        self.hit_streak = 0  # 연속 매칭 성공 횟수를 기록함
        self.age = 0  # tracker가 생성된 이후 지난 총 프레임 수를 기록함
        self.cls_name = ""  # 현재 tracker에 연결된 클래스 이름을 저장할 문자열 변수임

    def update(self, bbox, cls_name=""):
        self.time_since_update = 0  # 업데이트가 되었으므로 경과 프레임 수를 0으로 초기화함
        self.hits += 1  # 총 매칭 성공 횟수를 1 증가시킴
        self.hit_streak += 1  # 연속 매칭 성공 횟수를 1 증가시킴
        self.kf.update(convert_bbox_to_z(bbox))  # 새로 검출된 박스로 칼만 필터를 보정함
        if cls_name:  # 클래스 이름이 전달된 경우
            self.cls_name = cls_name  # tracker의 클래스 이름을 갱신함

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:  # 예측 결과 면적이 0 이하가 될 가능성이 있으면
            self.kf.x[6] = 0  # 면적 변화량을 0으로 보정함
        self.kf.predict()  # 칼만 필터 예측 단계를 수행함
        self.age += 1  # tracker의 전체 나이를 1 증가시킴

        if self.time_since_update > 0:  # 이전 프레임에서 업데이트가 없었다면
            self.hit_streak = 0  # 연속 매칭 카운트를 끊어줌
        self.time_since_update += 1  # 마지막 업데이트 이후 지난 프레임 수를 증가시킴

        return convert_x_to_bbox(self.kf.x)  # 예측된 상태를 박스 좌표로 바꿔 반환함

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)  # 현재 상태 벡터를 박스 좌표 형식으로 반환함


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:  # 현재 tracker가 하나도 없으면
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)  # 모든 검출은 unmatched로 반환함

    iou_matrix = iou_batch(detections, trackers)  # 검출 박스와 tracker 박스 사이의 IoU 행렬을 계산함
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # IoU가 최대가 되도록 음수 부호를 붙여 Hungarian Algorithm을 적용함

    matched_indices = np.array(list(zip(row_ind, col_ind))) if len(row_ind) > 0 else np.empty((0, 2), dtype=int)  # 매칭된 인덱스를 배열로 저장함

    unmatched_detections = []  # 매칭되지 않은 detection 인덱스를 저장할 리스트임
    for d in range(len(detections)):  # 모든 detection에 대해 반복함
        if d not in matched_indices[:, 0] if len(matched_indices) > 0 else True:  # 어떤 tracker와도 매칭되지 않았으면
            unmatched_detections.append(d)  # unmatched detection 목록에 추가함

    unmatched_trackers = []  # 매칭되지 않은 tracker 인덱스를 저장할 리스트임
    for t in range(len(trackers)):  # 모든 tracker에 대해 반복함
        if t not in matched_indices[:, 1] if len(matched_indices) > 0 else True:  # 어떤 detection과도 매칭되지 않았으면
            unmatched_trackers.append(t)  # unmatched tracker 목록에 추가함

    matches = []  # 최종 유효 매칭 결과를 저장할 리스트임
    for m in matched_indices:  # Hungarian Algorithm이 만든 매칭 후보를 하나씩 확인함
        if iou_matrix[m[0], m[1]] < iou_threshold:  # IoU가 기준값보다 낮으면
            unmatched_detections.append(m[0])  # detection을 unmatched로 처리함
            unmatched_trackers.append(m[1])  # tracker도 unmatched로 처리함
        else:  # IoU가 기준값 이상이면
            matches.append(m.reshape(1, 2))  # 유효 매칭으로 저장함

    if len(matches) == 0:  # 유효 매칭이 하나도 없으면
        matches = np.empty((0, 2), dtype=int)  # 빈 배열로 반환 준비를 함
    else:  # 유효 매칭이 있으면
        matches = np.concatenate(matches, axis=0)  # 리스트를 하나의 배열로 합침

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)  # 매칭/미매칭 결과를 모두 반환함


class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # tracker를 유지할 최대 미갱신 프레임 수를 저장함
        self.min_hits = min_hits  # 안정적인 tracker로 인정하기 위한 최소 연속 매칭 횟수를 저장함
        self.iou_threshold = iou_threshold  # detection과 tracker를 연결할 때 사용할 IoU 기준값을 저장함
        self.trackers = []  # 현재 활성화된 tracker 객체들을 저장할 리스트임
        self.frame_count = 0  # 지금까지 처리한 프레임 수를 기록함

    def update(self, dets=np.empty((0, 5)), class_names=None):
        self.frame_count += 1  # 새 프레임이 들어왔으므로 프레임 카운트를 증가시킴

        trks = np.zeros((len(self.trackers), 4))  # 현재 tracker들의 예측 박스를 저장할 배열을 만듦
        to_del = []  # 잘못된 tracker를 삭제하기 위한 인덱스 목록임
        ret = []  # 최종적으로 화면에 표시할 tracker 결과를 저장할 리스트임

        for t, trk in enumerate(self.trackers):  # 모든 tracker에 대해 반복함
            pos = trk.predict()[0]  # 현재 tracker의 다음 위치를 예측함
            trks[t] = [pos[0], pos[1], pos[2], pos[3]]  # 예측 박스를 배열에 저장함
            if np.any(np.isnan(pos)):  # 예측값에 NaN이 하나라도 있으면
                to_del.append(t)  # 나중에 삭제할 tracker 목록에 추가함

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 잘못된 예측값이 들어간 행을 제거함
        for t in reversed(to_del):  # 삭제 인덱스를 뒤에서부터 순회함
            self.trackers.pop(t)  # 잘못된 tracker를 리스트에서 제거함

        det_boxes = dets[:, :4] if len(dets) > 0 else np.empty((0, 4))  # detection이 있으면 박스 좌표만 추출하고, 없으면 빈 배열을 사용함

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(  # detection과 tracker를 IoU 기반으로 매칭함
            det_boxes, trks, self.iou_threshold
        )

        for m in matched:  # 매칭된 detection-tracker 쌍을 하나씩 처리함
            det_idx, trk_idx = m[0], m[1]  # detection 인덱스와 tracker 인덱스를 꺼냄
            cls_name = class_names[det_idx] if class_names is not None else ""  # detection에 대응하는 클래스 이름을 가져옴
            self.trackers[trk_idx].update(dets[det_idx, :4], cls_name)  # 매칭된 tracker를 detection 정보로 업데이트함

        for i in unmatched_dets:  # 매칭되지 않은 detection들을 하나씩 처리함
            trk = KalmanBoxTracker(dets[i, :4])  # 새 detection으로 새로운 tracker를 생성함
            if class_names is not None:  # 클래스 이름 정보가 있으면
                trk.cls_name = class_names[i]  # 새 tracker에 클래스 이름을 저장함
            self.trackers.append(trk)  # 생성한 tracker를 tracker 목록에 추가함

        i = len(self.trackers)  # tracker 개수를 변수에 저장함
        for trk in reversed(self.trackers):  # tracker를 뒤에서부터 순회함
            d = trk.get_state()[0]  # 현재 tracker의 상태를 박스 좌표로 가져옴
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # 최근 업데이트되었고 표시 조건을 만족하면
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # 박스 좌표와 ID를 합쳐 결과 목록에 추가함
            i -= 1  # 현재 인덱스를 1 감소시킴
            if trk.time_since_update > self.max_age:  # 오랫동안 업데이트되지 않은 tracker면
                self.trackers.pop(i)  # tracker 목록에서 제거함

        if len(ret) > 0:  # 화면에 표시할 tracker 결과가 하나라도 있으면
            return np.concatenate(ret), self.trackers  # 결과 배열과 전체 tracker 목록을 반환함
        return np.empty((0, 5)), self.trackers  # 표시할 결과가 없으면 빈 배열과 tracker 목록을 반환함


cfg_path = "yolov3.cfg"  # YOLOv3 설정 파일 경로를 저장함
weights_path = "yolov3.weights"  # YOLOv3 가중치 파일 경로를 저장함
video_path = "slow_traffic_small.mp4"  # 입력 비디오 파일 경로를 저장함

net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)  # Darknet 형식의 YOLOv3 네트워크를 불러옴
layer_names = net.getLayerNames()  # 네트워크 전체 레이어 이름 목록을 가져옴
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]  # 실제 출력에 사용하는 레이어 이름만 추출함

cap = cv.VideoCapture(video_path)  # 비디오 파일을 열어 프레임 단위로 읽기 위한 객체를 생성함
if not cap.isOpened():  # 비디오가 정상적으로 열리지 않으면
    print("비디오를 열 수 없습니다.")  # 에러 메시지를 출력함
    exit()  # 프로그램 실행을 종료함

tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)  # SORT 추적기를 생성하고 파라미터를 설정함

CONF_THRESHOLD = 0.5  # 객체 검출 confidence 임계값을 설정함
NMS_THRESHOLD = 0.4  # NMS 중복 제거 임계값을 설정함

while True:  # 비디오 프레임을 끝까지 처리하기 위해 무한 반복문을 시작함
    ret, frame = cap.read()  # 비디오에서 한 프레임을 읽어옴
    if not ret:  # 더 이상 읽을 프레임이 없으면
        break  # 반복문을 종료함

    h, w = frame.shape[:2]  # 현재 프레임의 높이와 너비를 가져옴

    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)  # 프레임을 YOLO 입력 형식의 blob으로 변환함
    net.setInput(blob)  # 변환한 blob을 네트워크 입력으로 설정함
    outputs = net.forward(output_layers)  # 출력 레이어까지 forward 연산을 수행하여 검출 결과를 얻음

    boxes = []  # 검출된 박스 정보를 저장할 리스트임
    confidences = []  # 검출 신뢰도를 저장할 리스트임
    class_ids = []  # 검출된 클래스 인덱스를 저장할 리스트임
    class_names = []  # 검출된 클래스 이름을 저장할 리스트임

    for output in outputs:  # 출력 레이어마다 반복함
        for detection in output:  # 각 detection 결과를 하나씩 확인함
            scores = detection[5:]  # 클래스별 점수 부분만 잘라냄
            class_id = int(np.argmax(scores))  # 가장 점수가 높은 클래스 인덱스를 구함
            confidence = float(scores[class_id])  # 가장 높은 클래스 점수를 confidence로 사용함

            if confidence > CONF_THRESHOLD:  # confidence가 임계값보다 크면
                cls_name = COCO_CLASSES[class_id]  # 클래스 인덱스를 실제 클래스 이름으로 변환함
                if cls_name not in TRACK_CLASSES:  # 추적 대상 클래스가 아니면
                    continue  # 이 detection은 건너뜀

                center_x = int(detection[0] * w)  # 검출 박스 중심 x 좌표를 원본 프레임 기준으로 변환함
                center_y = int(detection[1] * h)  # 검출 박스 중심 y 좌표를 원본 프레임 기준으로 변환함
                bw = int(detection[2] * w)  # 검출 박스 너비를 원본 프레임 기준으로 변환함
                bh = int(detection[3] * h)  # 검출 박스 높이를 원본 프레임 기준으로 변환함

                x = int(center_x - bw / 2)  # 좌상단 x 좌표를 계산함
                y = int(center_y - bh / 2)  # 좌상단 y 좌표를 계산함

                boxes.append([x, y, bw, bh])  # 박스 정보를 리스트에 저장함
                confidences.append(confidence)  # confidence를 리스트에 저장함
                class_ids.append(class_id)  # 클래스 인덱스를 리스트에 저장함
                class_names.append(cls_name)  # 클래스 이름을 리스트에 저장함

    indices = cv.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)  # 중복 박스를 제거하기 위해 NMS를 적용함

    detections = []  # 최종 detection 정보를 저장할 리스트임
    det_class_names = []  # 최종 detection 클래스 이름을 저장할 리스트임

    if len(indices) > 0:  # NMS 후 살아남은 박스가 하나 이상 있으면
        for i in indices.flatten():  # 살아남은 박스 인덱스를 하나씩 순회함
            x, y, bw, bh = boxes[i]  # 해당 박스의 좌표와 크기를 가져옴
            x1 = max(0, x)  # 좌상단 x 좌표가 0보다 작지 않도록 보정함
            y1 = max(0, y)  # 좌상단 y 좌표가 0보다 작지 않도록 보정함
            x2 = min(w - 1, x + bw)  # 우하단 x 좌표가 프레임 너비를 넘지 않도록 보정함
            y2 = min(h - 1, y + bh)  # 우하단 y 좌표가 프레임 높이를 넘지 않도록 보정함

            detections.append([x1, y1, x2, y2, confidences[i]])  # SORT에 넣을 detection 형식으로 저장함
            det_class_names.append(class_names[i])  # 해당 detection의 클래스 이름도 같이 저장함

    if len(detections) > 0:  # detection이 하나 이상 있으면
        detections = np.array(detections, dtype=np.float32)  # NumPy 배열로 변환함
    else:  # detection이 하나도 없으면
        detections = np.empty((0, 5), dtype=np.float32)  # 빈 detection 배열을 생성함

    tracks, all_trackers = tracker.update(detections, det_class_names)  # 현재 detection 결과로 SORT 추적기를 업데이트함

    for trk in all_trackers:  # 현재 tracker들을 하나씩 확인함
        if trk.time_since_update > 0:  # 이번 프레임에 갱신되지 않은 tracker는
            continue  # 화면에 그리지 않고 건너뜀

        bbox = trk.get_state()[0]  # tracker의 현재 박스 상태를 가져옴
        x1, y1, x2, y2 = map(int, bbox)  # 박스 좌표를 정수형으로 변환함
        obj_id = trk.id + 1  # 객체 ID를 사람이 보기 쉽게 1부터 시작하도록 설정함
        label = trk.cls_name if trk.cls_name else "object"  # 클래스 이름이 있으면 사용하고 없으면 object로 표시함

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 프레임 위에 객체 경계 상자를 그림
        cv.putText(  # 프레임 위에 객체 ID와 클래스 이름을 텍스트로 표시함
            frame,  # 텍스트를 그릴 대상 프레임임
            f"ID {obj_id}: {label}",  # 표시할 텍스트 내용임
            (x1, max(0, y1 - 10)),  # 텍스트를 그릴 시작 위치를 지정함
            cv.FONT_HERSHEY_SIMPLEX,  # 글꼴 종류를 지정함
            0.6,  # 글자 크기를 지정함
            (0, 255, 0),  # 글자 색상을 초록색으로 지정함
            2  # 글자 두께를 지정함
        )

    cv.imshow("YOLOv3 + SORT Tracking", frame)  # 결과 프레임을 화면 창에 표시함

    key = cv.waitKey(30) & 0xFF  # 30ms 대기하면서 키 입력을 확인함
    if key == 27:  # ESC 키가 눌리면
        break  # 반복문을 종료함

cap.release()  # 비디오 객체를 해제함
cv.destroyAllWindows()  # OpenCV로 연 모든 창을 닫음
```


Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

1. 과제 개요

이번 과제는 Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 랜드마크를 추출하고, 이를 이미지 위에 점으로 시각화하는 프로그램을 구현하는 것이다.
FaceMesh는 얼굴의 주요 특징점을 매우 촘촘하게 추출할 수 있는 얼굴 랜드마크 검출 모델이며, 눈, 코, 입, 얼굴 윤곽 등 다양한 위치 정보를 얻을 수 있다.

본 구현에서는 test.jpg 이미지를 입력으로 사용하여 얼굴을 검출하고, 검출된 랜드마크를 OpenCV의 circle() 함수를 이용해 이미지 위에 점으로 표시하도록 구성하였다.
이를 통해 얼굴의 형태와 주요 구조를 시각적으로 확인할 수 있도록 하였다.

2. 배경지식

2-1. 얼굴 랜드마크(Facial Landmark)

얼굴 랜드마크는 얼굴에서 의미 있는 특정 위치를 나타내는 점이다.
예를 들어 눈의 가장자리, 코끝, 입술의 경계, 얼굴 윤곽선 등이 이에 해당한다.
이러한 랜드마크는 얼굴 정렬, 표정 분석, 얼굴 인식, AR 필터, 자세 추정 등 다양한 분야에서 활용된다.

2-2. Mediapipe FaceMesh

Mediapipe는 Google에서 제공하는 멀티미디어 기반 머신러닝 프레임워크로, 얼굴, 손, 자세 등 다양한 객체를 빠르게 검출할 수 있다.
그중 FaceMesh는 얼굴의 세밀한 랜드마크를 추출하는 기능을 제공하며, 얼굴의 구조를 정밀하게 분석할 수 있다.
이번 과제에서는 이 FaceMesh를 사용하여 얼굴의 특징점을 추출하였다.

2-3. 이미지 좌표와 정규화 좌표

Mediapipe가 반환하는 랜드마크 좌표는 0~1 범위의 정규화된 값이다.
따라서 실제 이미지 위에 점을 표시하려면 이미지의 너비와 높이를 곱해 픽셀 좌표로 변환해야 한다.
이번 구현에서는 landmark.x * w, landmark.y * h 방식으로 실제 좌표를 계산하였다.

2-4. OpenCV 시각화

OpenCV는 이미지와 비디오를 처리하는 대표적인 컴퓨터 비전 라이브러리이다.
이번 과제에서는 OpenCV를 사용하여 이미지를 불러오고, 색상 공간을 변환하며, 랜드마크 위치에 점을 그린 뒤 최종 결과를 화면에 출력하였다.

3. 사용한 주요 알고리즘 및 구성 요소
3-1. FaceMesh 모듈 초기화

Mediapipe의 FaceMesh 객체를 생성하여 얼굴 랜드마크 검출기를 초기화하였다.
이때 정적인 이미지 1장을 처리하기 위해 static_image_mode=True로 설정하였고, 최대 1개의 얼굴만 검출하도록 max_num_faces=1로 지정하였다.

3-2. 이미지 입력

OpenCV의 cv.imread()를 사용하여 test.jpg 파일을 불러왔다.
이미지가 정상적으로 열리지 않을 경우에는 오류 메시지를 출력하고 프로그램을 종료하도록 구성하였다.

3-3. BGR → RGB 변환

OpenCV는 기본적으로 BGR 형식으로 이미지를 읽지만, Mediapipe는 RGB 형식의 이미지를 입력으로 사용한다.
따라서 cv.cvtColor()를 이용하여 BGR 이미지를 RGB로 변환한 뒤 FaceMesh에 전달하였다.

3-4. 랜드마크 검출

변환된 RGB 이미지를 face_mesh.process()에 입력하여 얼굴 랜드마크를 검출하였다.
검출 결과는 results.multi_face_landmarks에 저장되며, 얼굴이 검출된 경우 각 랜드마크 점의 좌표를 순회하며 처리하였다.

3-5. 랜드마크 시각화

검출된 랜드마크 좌표는 정규화된 값이므로, 이미지 크기를 기준으로 실제 픽셀 좌표로 변환하였다.
이후 OpenCV의 cv.circle()을 사용하여 각 랜드마크 위치에 초록색 점을 표시하였다.

4. 주요 코드 설명

4-1. 라이브러리 불러오기
import cv2 as cv
import mediapipe as mp

OpenCV와 Mediapipe 라이브러리를 불러오는 부분이다.

4-2. FaceMesh 객체 생성
```
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
```
FaceMesh 모듈을 준비하고, 얼굴 랜드마크를 검출하기 위한 객체를 생성하는 부분이다.
```
static_image_mode=True : 정적인 이미지 처리
max_num_faces=1 : 최대 1개의 얼굴 검출
refine_landmarks=True : 더 세밀한 랜드마크 추출
min_detection_confidence=0.5 : 최소 검출 신뢰도
4-3. 이미지 불러오기
img = cv.imread("test.jpg")

if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()
```
입력 이미지 test.jpg를 불러오고, 이미지가 없거나 경로가 잘못된 경우 프로그램을 종료하도록 하였다.

4-4. 색상 변환
```
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
```
OpenCV의 BGR 이미지를 Mediapipe가 처리할 수 있는 RGB 형식으로 변환하는 부분이다.

4-5. 랜드마크 검출
```
results = face_mesh.process(rgb)
```
RGB 이미지를 FaceMesh에 입력하여 얼굴 랜드마크를 검출하는 부분이다.

4-6. 랜드마크를 점으로 표시
```
if results.multi_face_landmarks:
    h, w, _ = img.shape

    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv.circle(img, (x, y), 1, (0, 255, 0), -1)
```

검출된 랜드마크 좌표를 실제 이미지 좌표로 변환한 뒤, 각 점을 이미지 위에 표시하는 부분이다.

landmark.x * w : x 좌표를 이미지 너비 기준 픽셀 좌표로 변환
landmark.y * h : y 좌표를 이미지 높이 기준 픽셀 좌표로 변환
cv.circle() : 해당 위치에 점 표시

4-7. 결과 출력
```
cv.imshow("FaceMesh Result", img)
cv.waitKey(0)
cv.destroyAllWindows()
face_mesh.close()
```
랜드마크가 표시된 이미지를 화면에 출력하고, 키 입력이 있을 때까지 창을 유지한 뒤 종료하는 부분이다.


실습 이미지
<img width="1001" height="770" alt="image" src="https://github.com/user-attachments/assets/b999525d-d536-4009-9618-0f2d26f1a1a4" />

전체코드
```
import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import mediapipe as mp  # Mediapipe 라이브러리를 mp라는 이름으로 불러옴

# Mediapipe의 FaceMesh 모듈을 가져와서 변수에 저장함
mp_face_mesh = mp.solutions.face_mesh

# FaceMesh 객체를 생성함
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # 입력이 실시간 영상이 아니라 정적인 이미지 1장임을 설정함
    max_num_faces=1,  # 최대 1개의 얼굴만 검출하도록 설정함
    refine_landmarks=True,  # 눈, 입술 등 세부 랜드마크를 더 정교하게 추출하도록 설정함
    min_detection_confidence=0.5  # 얼굴 검출 최소 신뢰도 기준을 0.5로 설정함
)

# test.jpg 이미지를 불러와 img 변수에 저장함
img = cv.imread("test.jpg")

# 이미지가 정상적으로 불러와지지 않았으면
if img is None:
    print("이미지를 불러올 수 없습니다.")  # 오류 메시지를 출력함
    exit()  # 프로그램 실행을 종료함

# OpenCV는 BGR 형식을 사용하므로, Mediapipe에서 사용하는 RGB 형식으로 변환함
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 변환한 RGB 이미지에서 얼굴 랜드마크를 검출함
results = face_mesh.process(rgb)

# 얼굴 랜드마크가 하나라도 검출되었으면
if results.multi_face_landmarks:
    h, w, _ = img.shape  # 원본 이미지의 높이와 너비를 가져옴

    for face_landmarks in results.multi_face_landmarks:  # 검출된 각 얼굴에 대해 반복함
        for landmark in face_landmarks.landmark:  # 얼굴의 각 랜드마크 점에 대해 반복함
            x = int(landmark.x * w)  # 정규화된 x 좌표를 이미지 너비 기준 실제 픽셀 좌표로 변환함
            y = int(landmark.y * h)  # 정규화된 y 좌표를 이미지 높이 기준 실제 픽셀 좌표로 변환함

            cv.circle(img, (x, y), 1, (0, 255, 0), -1)  # 계산한 위치에 반지름 1의 초록색 점을 그림

# 랜드마크가 그려진 결과 이미지를 화면에 출력함
cv.imshow("FaceMesh Result", img)

# 키 입력이 있을 때까지 창을 유지함
cv.waitKey(0)

# OpenCV 창을 모두 닫음
cv.destroyAllWindows()

# FaceMesh 객체를 종료하고 자원을 해제함
face_mesh.close()
```
