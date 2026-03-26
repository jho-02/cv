과제 1 : SIFT를 이용한 특징점 검출 및 시각화
과제 설명

1번 과제에서는 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT(Scale-Invariant Feature Transform) 알고리즘을 적용하고
이미지 내의 특징점을 검출 및 시각화했습니다.
SIFT는 이미지에서 의미 있는 구조적 정보를 가지는 지점을 특징점으로 검출하고, 
이를 기술자(descriptor)로 표현하는 알고리즘이다. 본 실습에서는 해당 특징점을 추출한 후
원본 이미지와 특징점이 표시된 이미지를 나란히 출력하여 시각적으로 확인했습니다.

배경 지식
1. Local Feature

Local Feature는 이미지 전체가 아닌 특정 위치에서 추출되는 특징을 의미합니다.
이러한 특징은 이미지의 일부가 변형되거나 가려지더라도 비교적 안정적으로 유지되기 때문에 이미지 매칭이나 객체 인식 등의 다양한 컴퓨터 비전 문제에서 활용됩니다.

2. SIFT (Scale-Invariant Feature Transform)

SIFT는 대표적인 지역 특징 추출 알고리즘으로, 다음과 같은 특성을 가집니다.

- 크기 변화(scale)에 강인함
- 회전(rotation)에 강인함
- 조명 변화에도 비교적 안정적

SIFT는 이미지에서 특징점을 검출하고, 각 특징점 주변의 패턴을 기반으로 고유한 기술자(descriptor)를 생성 하며 이후 이미지 간 대응점을 찾는 데 사용됩니다.

3. 특징점 시각화

검출된 특징점은 단순한 점이 아니라 다음 정보를 포함합니다.

- 위치 (x, y)
- 크기 (scale)
- 방향 (orientation)

cv.drawKeypoints()에서 DRAW_RICH_KEYPOINTS 옵션을 사용하면 이러한 정보를 시각적으로 함께 표현할 수 있습니다.

주요 코드 설명
1. 이미지 불러오기
```
img = cv.imread("mot_color70.jpg")
```

OpenCV의 cv.imread()를 사용하여 이미지를 불러옵니다.
이미지를 정상적으로 읽지 못할 경우 None이 반환되므로 이를 확인하여 예외 처리를 수행했습니다.

2. 색상 변환 (BGR → RGB)
```
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
```

OpenCV는 이미지를 BGR 형식으로 읽기 때문에, Matplotlib에서 올바른 색상으로 출력하기 위해 RGB로 변환했습니다.

3. SIFT 객체 생성

```
sift = cv.SIFT_create(nfeatures=300)
```
SIFT 객체를 생성하고, nfeatures=300 옵션을 통해 검출할 특징점의 개수를 제한했습니다.
이는 특징점이 과도하게 많아지는 것을 방지하고 결과를 보다 명확하게 보기 위함입니다.

4. 특징점 검출 및 기술자 계산

```
keypoints, descriptors = sift.detectAndCompute(img, None)
```

- keypoints: 특징점 위치, 크기, 방향 등의 정보
- descriptors: 특징점을 수치적으로 표현한 벡터

이 단계에서 SIFT 알고리즘이 이미지의 중요한 지점을 자동으로 검출합니다.

5. 특징점 시각화

```
img_keypoints = cv.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
```
검출된 특징점을 이미지 위에 표시한다.
DRAW_RICH_KEYPOINTS 옵션을 사용하여 특징점의 크기와 방향까지 함께 시각화했습니다.

실습 이미지

<img width="1918" height="617" alt="image" src="https://github.com/user-attachments/assets/0767978b-e734-4b45-841f-cd7c0394a7d5" />

전체코드

```
import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 사용합니다.
import matplotlib.pyplot as plt  # 결과 이미지를 시각화하기 위해 matplotlib을 사용합니다.

# 이미지 파일을 불러옵니다.
img = cv.imread("mot_color70.jpg")

# 이미지가 정상적으로 불러와졌는지 확인합니다.
# 만약 None이라면 경로 문제 등으로 이미지를 읽지 못한 경우입니다.
if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# OpenCV는 기본적으로 BGR 형식으로 이미지를 읽기 때문에,
# matplotlib에서 올바른 색으로 출력하기 위해 RGB로 변환합니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# SIFT 객체를 생성합니다.
# nfeatures=300으로 설정하여 특징점 개수를 최대 300개로 제한합니다.
sift = cv.SIFT_create(nfeatures=300)

# detectAndCompute 함수를 사용하여 특징점을 검출하고,
# 각 특징점에 대한 기술자(descriptor)를 함께 계산합니다.
keypoints, descriptors = sift.detectAndCompute(img, None)

# 검출된 특징점을 원본 이미지 위에 시각화합니다.
# DRAW_RICH_KEYPOINTS 옵션을 사용하여 특징점의 크기와 방향까지 함께 표시합니다.
img_keypoints = cv.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 시각화된 이미지도 matplotlib 출력에 맞게 RGB로 변환합니다.
img_keypoints_rgb = cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB)

# 출력 이미지 크기를 설정합니다.
plt.figure(figsize=(16, 8))

# 첫 번째 subplot에 원본 이미지를 출력합니다.
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")  # 제목 설정
plt.axis("off")  # 축 정보 제거

# 두 번째 subplot에 특징점이 시각화된 이미지를 출력합니다.
plt.subplot(1, 2, 2)
plt.imshow(img_keypoints_rgb)
plt.title(f"SIFT Keypoints ({len(keypoints)} points)")  # 특징점 개수를 함께 표시
plt.axis("off")  # 축 정보 제거

# subplot 간 간격을 자동으로 조정합니다.
plt.tight_layout()

# 최종 결과를 화면에 출력합니다.
plt.show()
```


과제 2 : SIFT를 이용한 두 영상 간 특징점 매칭

과제 설명

2번과제에서는 두 개의 이미지에서 SIFT(Scale-Invariant Feature Transform) 특징점을 추출한 뒤
이를 바탕으로 두 영상 사이의 대응점을 매칭하고 결과를 시각화했습니다.
특징점 매칭은 서로 다른 두 이미지에서 같은 물체나 같은 위치를 나타내는 지점을 찾는 과정입니다.
이번 실습에서는 두 이미지에서 SIFT 특징점과 기술자를 추출한 후, BFMatcher를 사용하여 각 특징점 사이의 유사도를 비교하고
매칭 결과를 이미지 위에 선으로 연결하여 확인했습니다.

배경 지식

1. 특징점 매칭

특징점 매칭은 두 이미지에서 추출된 특징점들 중 서로 대응되는 점을 찾는 과정입니다.
같은 장면을 다른 시점에서 촬영하거나, 약간의 이동 및 회전이 있는 경우에도 대응되는 특징점을 찾을 수 있다면 두 이미지 간 관계를 파악할 수 있습니다.


2. SIFT 기술자(Descriptor)

SIFT는 특징점의 위치만 검출하는 것이 아니라 각 특징점 주변의 지역적인 패턴을 수치 벡터 형태의 기술자(descriptor)로 표현하며
이를 이용하면 두 이미지에서 서로 비슷한 특징점을 비교할 수 있습니다.

3. BFMatcher

BFMatcher(Brute-Force Matcher)는 한 이미지의 기술자와 다른 이미지의 기술자를 하나씩 직접 비교하여 가장 유사한 쌍을 찾는 방식입니다.
SIFT 기술자는 실수형 벡터이므로 거리 계산 방식으로 cv.NORM_L2를 사용합니다.
또한 crossCheck=True 옵션을 사용하면 한쪽 이미지에서 선택한 최근접 이웃이 반대쪽 이미지에서도 서로를 가장 가까운 특징점으로 선택하는 경우만 매칭으로 인정하므로 비교적 안정적인 결과를 얻을 수 있습니다.

4. 매칭 결과 시각화

검출된 특징점 매칭 결과는 cv.drawMatches()를 통해 두 이미지를 나란히 배치한 후, 대응되는 특징점 사이를 선으로 연결하여 시각화할 수 있습니다.
이 과정을 통해 두 이미지 사이에서 어떤 지점들이 서로 대응되는지 직관적으로 확인할 수 있습니다.

주요 코드 설명
1. 두 이미지 불러오기

```
img1 = cv.imread("4week/mot_color70.jpg")
img2 = cv.imread("4week/mot_color83.jpg")
```

cv.imread()를 사용하여 두 개의 이미지를 불러옵니다.
이미지 중 하나라도 정상적으로 읽히지 않으면 이후 특징점 추출과 매칭을 수행할 수 없으므로, None 여부를 확인하는 과정이 필요합니다.

2. SIFT 객체 생성

```
sift = cv.SIFT_create()
```
SIFT 객체를 생성하여 두 이미지에 대해 동일한 방식으로 특징점과 기술자를 추출할 수 있도록 합니다.

3. 특징점 검출 및 기술자 계산

```
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

각 이미지에서 특징점(keypoints)과 기술자(descriptors)를 추출합니다.
이 기술자는 이후 두 이미지 간의 유사한 특징점을 찾는 기준이 됩니다.

4. BFMatcher 생성 및 매칭 수행

```
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
```

SIFT 기술자에 맞게 cv.NORM_L2를 사용하여 BFMatcher를 생성한 후, 두 이미지의 기술자를 비교하여 매칭을 수행합니다.
crossCheck=True를 사용하여 보다 신뢰도 있는 매칭만 남기도록 했습니다.

5. 거리 기준 정렬

```
matches = sorted(matches, key=lambda x: x.distance)
```

매칭 결과를 거리(distance) 기준으로 오름차순 정렬했습니다.
거리가 작을수록 두 특징점이 더 유사하다고 볼 수 있으므로 더 좋은 매칭을 앞쪽에 배치할 수 있습니다.

6. 상위 매칭만 선택

```
good_matches = matches[:50]
```

모든 매칭 결과를 한 번에 시각화하면 선이 너무 많아져 결과를 해석하기 어렵기 때문에 거리 기준으로 상위 50개의 매칭만 선택했습니다.

7. 매칭 결과 시각화

```
result = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

cv.drawMatches()를 사용하여 두 이미지의 대응되는 특징점을 선으로 연결했습니다.
NOT_DRAW_SINGLE_POINTS 옵션을 사용하여 실제로 매칭된 특징점만 표시하도록 했습니다.

실습 이미지
<img width="1917" height="640" alt="image" src="https://github.com/user-attachments/assets/5e22da25-2c39-40ab-b1c1-94a5d75e7e63" />


전체 코드
```
import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 사용합니다.
import matplotlib.pyplot as plt  # 결과 이미지를 출력하기 위해 matplotlib을 사용합니다.

# 두 장의 이미지를 불러옵니다.
# 첫 번째 이미지는 기준 이미지로 사용하고,
# 두 번째 이미지는 이와 비교할 이미지로 사용합니다.
img1 = cv.imread("4week/mot_color70.jpg")
img2 = cv.imread("4week/mot_color83.jpg")

# 두 이미지 중 하나라도 정상적으로 불러오지 못한 경우
# 이후 작업을 진행할 수 없으므로 프로그램을 종료합니다.
if img1 is None or img2 is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# SIFT 객체를 생성합니다.
# 이 객체를 이용하여 특징점과 기술자를 추출할 수 있습니다.
sift = cv.SIFT_create()

# 첫 번째 이미지에서 특징점과 기술자를 추출합니다.
kp1, des1 = sift.detectAndCompute(img1, None)

# 두 번째 이미지에서 특징점과 기술자를 추출합니다.
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher 객체를 생성합니다.
# SIFT 기술자는 실수형 벡터이므로 거리 계산 방식으로 NORM_L2를 사용합니다.
# crossCheck=True로 설정하여 서로 일치하는 매칭만 남기도록 합니다.
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# 두 이미지의 기술자를 비교하여 특징점 매칭을 수행합니다.
matches = bf.match(des1, des2)

# 매칭 결과를 거리(distance) 기준으로 오름차순 정렬합니다.
# 거리가 작을수록 더 유사한 특징점이라고 볼 수 있습니다.
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과가 너무 많으면 화면이 복잡해질 수 있으므로
# 상위 50개의 매칭만 선택하여 시각화합니다.
good_matches = matches[:50]

# 선택된 매칭 결과를 시각화합니다.
# 두 이미지를 나란히 놓고, 대응되는 특징점들을 선으로 연결합니다.
result = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# OpenCV는 BGR 형식으로 이미지를 다루기 때문에,
# matplotlib에서 올바른 색으로 출력하기 위해 RGB로 변환합니다.
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

# 출력 창의 크기를 설정합니다.
plt.figure(figsize=(18, 8))

# 매칭 결과 이미지를 출력합니다.
plt.imshow(result_rgb)

# 제목에 현재 표시된 매칭 개수를 함께 표시합니다.
plt.title(f"SIFT Matching Result ({len(good_matches)} matches)")

# 축 정보는 필요하지 않으므로 제거합니다.
plt.axis("off")

# 레이아웃이 겹치지 않도록 자동으로 정렬합니다.
plt.tight_layout()

# 최종 결과를 화면에 출력합니다.
plt.show()
```


과제 3 : 호모그래피를 이용한 이미지 정합 (Image Alignment)

과제 설명

3번 과제에서는 두 이미지에서 SIFT 특징점을 검출한 뒤, 이를 기반으로 대응점을 찾고
그 대응점을 이용하여 호모그래피(Homography) 행렬을 계산함으로써 한 이미지를 다른 이미지에 정렬하는 이미지 정합(Image Alignment)을 수행하였습니다.
이미지 정합은 서로 다른 시점에서 촬영된 두 이미지가 있을 때 한 이미지를 다른 이미지의 좌표계에 맞추어 변환하는 과정입니다.
이번 실습에서는 img1.jpg, img2.jpg, img3.jpg 중 두 장을 선택하여 특징점 매칭을 수행하고
그 결과를 바탕으로 호모그래피를 계산한 뒤 warpPerspective()를 이용하여 정합 결과를 확인하였습니다.

배경 지식

1. 이미지 정합(Image Alignment)

이미지 정합은 두 이미지 사이의 기하학적 관계를 찾아 한 이미지를 다른 이미지에 맞추는 과정입니다.
같은 장면을 서로 다른 위치나 각도에서 촬영한 경우에도, 대응되는 점을 잘 찾을 수 있다면 두 이미지를 같은 기준으로 정렬할 수 있습니다.

2. 호모그래피(Homography)

호모그래피는 한 평면에서의 점이 다른 이미지에서 어디로 이동하는지를 나타내는 3×3 변환 행렬입니다.
이를 이용하면 한 이미지의 원근 변화, 회전, 이동 등을 반영하여 다른 이미지 위에 정렬할 수 있습니다.
충분한 대응점만 확보할 수 있다면 한 이미지와 다른 이미지 사이의 투시 변환 관계를 계산할 수 있습니다.

3. knnMatch와 Ratio Test

특징점 매칭 과정에서는 잘못된 대응점(outlier)이 포함될 수 있기 때문에
단순히 가장 가까운 특징점 하나만 사용하는 것보다 두 개의 최근접 이웃을 비교하는 방식이 더 안정적입니다.
knnMatch()는 각 특징점에 대해 가장 가까운 이웃 2개를 찾고
이후 Ratio Test를 적용하여 첫 번째 거리와 두 번째 거리의 비율이 충분히 작은 경우만 좋은 매칭으로 선택합니다.
이 방법은 애매한 매칭을 줄여 보다 신뢰도 높은 대응점만 남기는 데 도움이 됩니다.

4. RANSAC

실제 매칭 결과에는 틀린 대응점이 섞여 있을 수 있습니다.
이러한 이상점(outlier)이 그대로 포함되면 잘못된 호모그래피가 계산될 수 있기 때문에
cv.findHomography()에서 RANSAC을 사용하여 일관된 관계를 가지는 대응점만 선택하도록 하였습니다.
이를 통해 보다 안정적인 정합 결과를 얻을 수 있습니다.

주요 코드 설명
1. 두 이미지 불러오기
img1 = cv.imread("4week/img1.jpg")
img2 = cv.imread("4week/img2.jpg")

cv.imread()를 사용하여 두 개의 이미지를 불러왔습니다.
이미지 중 하나라도 정상적으로 읽히지 않으면 특징점 추출과 정합을 수행할 수 없기 때문에, None 여부를 확인하는 과정이 필요합니다.

2. SIFT 특징점 검출 및 기술자 계산

```
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

SIFT 객체를 생성한 뒤, 각 이미지에서 특징점과 기술자를 추출하였습니다.
이 기술자는 두 이미지 사이에서 서로 대응되는 지점을 찾기 위한 기준이 됩니다.

3. BFMatcher와 knnMatch 사용

```
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des2, des1, k=2)
```

SIFT 기술자는 실수형 벡터이므로 거리 계산 방식으로 cv.NORM_L2를 사용하였습니다.
이후 knnMatch()를 통해 각 특징점마다 가장 가까운 두 개의 이웃을 찾도록 하였습니다.

4. 좋은 매칭점 선별

```
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

최근접 이웃 두 개의 거리 비율을 비교하여, 첫 번째 매칭이 두 번째보다 충분히 더 좋은 경우만 선택하였습니다.
이 과정은 잘못된 매칭을 줄이고 이후 호모그래피 계산의 정확도를 높이기 위한 단계입니다.

5. 호모그래피 계산

```
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
```

좋은 매칭점들로부터 대응 좌표를 추출한 뒤, cv.findHomography()를 사용하여 호모그래피 행렬을 계산하였습니다.
이때 cv.RANSAC을 적용하여 이상점의 영향을 줄이도록 하였습니다.


실습이미지
<img width="1918" height="412" alt="image" src="https://github.com/user-attachments/assets/234f2932-a269-4152-8fcd-b9810053d88a" />


전체코드
```
import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 사용합니다.
import numpy as np  # 좌표 배열을 다루기 위해 NumPy를 사용합니다.
import matplotlib.pyplot as plt  # 결과 이미지를 시각화하기 위해 matplotlib을 사용합니다.

# 정합에 사용할 두 장의 이미지를 불러옵니다.
# 여기서는 img1을 기준 이미지로, img2를 변환할 이미지로 사용합니다.
img1 = cv.imread("4week/img1.jpg")
img2 = cv.imread("4week/img2.jpg")

# 두 이미지 중 하나라도 정상적으로 불러오지 못한 경우
# 이후 연산을 진행할 수 없으므로 프로그램을 종료합니다.
if img1 is None or img2 is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# SIFT 객체를 생성합니다.
# 이 객체를 이용하여 특징점과 기술자를 추출합니다.
sift = cv.SIFT_create()

# 첫 번째 이미지에서 특징점과 기술자를 추출합니다.
kp1, des1 = sift.detectAndCompute(img1, None)

# 두 번째 이미지에서 특징점과 기술자를 추출합니다.
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher 객체를 생성합니다.
# SIFT 기술자는 실수형 벡터이므로 거리 계산 방식으로 NORM_L2를 사용합니다.
bf = cv.BFMatcher(cv.NORM_L2)

# 각 특징점에 대해 가장 가까운 이웃 2개를 찾습니다.
# 이를 통해 ratio test를 적용할 수 있습니다.
matches = bf.knnMatch(des2, des1, k=2)

# 좋은 매칭만 저장할 리스트를 생성합니다.
good_matches = []

# knnMatch로 구한 최근접 이웃 2개의 거리를 비교하여
# 더 신뢰도 높은 매칭만 선별합니다.
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 호모그래피를 계산하려면 최소 4개의 대응점이 필요하므로,
# 좋은 매칭 개수가 4개 미만이면 더 이상 진행하지 않습니다.
if len(good_matches) < 4:
    print("호모그래피 계산에 필요한 매칭점이 부족합니다.")
    exit()

# 두 번째 이미지에서의 대응 좌표들을 추출합니다.
# queryIdx는 knnMatch 기준으로 현재 질의 이미지의 특징점을 의미합니다.
src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 첫 번째 이미지에서의 대응 좌표들을 추출합니다.
# trainIdx는 비교 대상 이미지에서 매칭된 특징점을 의미합니다.
dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 좋은 매칭점들을 이용하여 호모그래피 행렬을 계산합니다.
# RANSAC을 사용하여 잘못된 매칭점(outlier)의 영향을 줄입니다.
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# 호모그래피 계산에 실패한 경우 프로그램을 종료합니다.
if H is None:
    print("호모그래피 계산 실패")
    exit()

# 두 이미지의 높이와 너비를 각각 구합니다.
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 계산된 호모그래피를 이용하여 img2를 img1의 좌표계로 변환합니다.
# 출력 크기는 두 이미지를 합친 파노라마 크기로 설정합니다.
warped = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))

# 기준 이미지인 img1을 결과 이미지의 왼쪽 영역에 그대로 배치합니다.
warped[0:h1, 0:w1] = img1

# RANSAC 결과로 얻은 inlier / outlier 정보를 리스트 형태로 변환합니다.
matches_mask = mask.ravel().tolist()

# 특징점 매칭 결과를 시각화합니다.
# matchesMask를 사용하면 RANSAC에서 inlier로 판단된 매칭만 표시할 수 있습니다.
matching_result = cv.drawMatches(
    img2, kp2,
    img1, kp1,
    good_matches, None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# OpenCV는 BGR 형식으로 이미지를 처리하므로,
# matplotlib에서 올바른 색상을 출력하기 위해 RGB로 변환합니다.
warped_rgb = cv.cvtColor(warped, cv.COLOR_BGR2RGB)
matching_rgb = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)

# 출력 창의 크기를 설정합니다.
plt.figure(figsize=(20, 8))

# 왼쪽 subplot에 특징점 매칭 결과를 출력합니다.
plt.subplot(1, 2, 1)
plt.imshow(matching_rgb)
plt.title(f"Matching Result (Inliers: {sum(matches_mask)}/{len(good_matches)})")
plt.axis("off")

# 오른쪽 subplot에 정합된 이미지를 출력합니다.
plt.subplot(1, 2, 2)
plt.imshow(warped_rgb)
plt.title("Warped Image")
plt.axis("off")

# subplot 간 간격을 자동으로 정리합니다.
plt.tight_layout()

# 최종 결과를 화면에 출력합니다.
plt.show()
```
