import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# -----------------------------
# COCO 클래스 이름
# -----------------------------
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

# 추적할 클래스만 선택
TRACK_CLASSES = {"person", "car", "bus", "truck", "motorbike", "bicycle"}


# -----------------------------
# SORT 구현에 필요한 함수들
# -----------------------------
def iou_batch(bb_test, bb_gt):
    """
    bb_test: Nx4
    bb_gt: Mx4
    box format: [x1, y1, x2, y2]
    """
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)

    bb_test = np.expand_dims(bb_test, 1)   # N x 1 x 4
    bb_gt = np.expand_dims(bb_gt, 0)       # 1 x M x 4

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    union = area_test + area_gt - inter
    return inter / (union + 1e-6)


def convert_bbox_to_z(bbox):
    """
    [x1,y1,x2,y2] -> [x, y, s, r]
    x,y는 중심점, s는 면적, r은 종횡비
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / (h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    """
    [x, y, s, r] -> [x1,y1,x2,y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    return np.array([
        x[0] - w / 2.,
        x[1] - h / 2.,
        x[0] + w / 2.,
        x[1] + h / 2.
    ]).reshape((1, 4))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls_name = ""

    def update(self, bbox, cls_name=""):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        if cls_name:
            self.cls_name = cls_name

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_indices = np.array(list(zip(row_ind, col_ind))) if len(row_ind) > 0 else np.empty((0, 2), dtype=int)

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0] if len(matched_indices) > 0 else True:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1] if len(matched_indices) > 0 else True:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), class_names=None):
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        det_boxes = dets[:, :4] if len(dets) > 0 else np.empty((0, 4))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, trks, self.iou_threshold
        )

        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            cls_name = class_names[det_idx] if class_names is not None else ""
            self.trackers[trk_idx].update(dets[det_idx, :4], cls_name)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            if class_names is not None:
                trk.cls_name = class_names[i]
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret), self.trackers
        return np.empty((0, 5)), self.trackers


# -----------------------------
# YOLOv3 로드
# -----------------------------
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"
video_path = "slow_traffic_small.mp4"

net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("비디오를 열 수 없습니다.")
    exit()

tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    class_names = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > CONF_THRESHOLD:
                cls_name = COCO_CLASSES[class_id]
                if cls_name not in TRACK_CLASSES:
                    continue

                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)
                class_names.append(cls_name)

    indices = cv.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    detections = []
    det_class_names = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)

            detections.append([x1, y1, x2, y2, confidences[i]])
            det_class_names.append(class_names[i])

    if len(detections) > 0:
        detections = np.array(detections, dtype=np.float32)
    else:
        detections = np.empty((0, 5), dtype=np.float32)

    tracks, all_trackers = tracker.update(detections, det_class_names)

    # 현재 프레임에 보이는 tracker들만 그리기
    for trk in all_trackers:
        if trk.time_since_update > 0:
            continue

        bbox = trk.get_state()[0]
        x1, y1, x2, y2 = map(int, bbox)
        obj_id = trk.id + 1
        label = trk.cls_name if trk.cls_name else "object"

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

    cv.imshow("YOLOv3 + SORT Tracking", frame)

    key = cv.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()