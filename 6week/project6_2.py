import cv2 as cv
import mediapipe as mp

# Mediapipe FaceMesh 모듈 준비
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 이미지 불러오기
img = cv.imread("test.jpg")

if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# BGR -> RGB 변환
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 얼굴 랜드마크 검출
results = face_mesh.process(rgb)

# 랜드마크를 점으로 표시
if results.multi_face_landmarks:
    h, w, _ = img.shape

    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv.circle(img, (x, y), 1, (0, 255, 0), -1)

# 결과 출력
cv.imshow("FaceMesh Result", img)
cv.waitKey(0)
cv.destroyAllWindows()

face_mesh.close()