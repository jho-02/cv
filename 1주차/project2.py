import cv2 as cv

# 이미지 불러오기
img = cv.imread("welcome.jpg")

if img is None: # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지 없음") 
    exit() #  프로그램 종료

brush = 5 # 브러시 크기 초기값
drawing = False # 마우스 드로잉 상태
color = (255,0,0) # 초기 색상 (파랑) - OpenCV는 BGR 순서로 색상을 표현

def draw(event, x, y, flags, param): # 마우스 이벤트 콜백 함수
    global drawing, color, brush, img # 전역 변수 사용 선언

    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽 버튼 클릭 시 드로잉 시작
        drawing = True # 드로잉 상태로 변경
        color = (255,0,0)  # 파랑
        cv.circle(img,(x,y),brush,color,-1) # 원 그리기 (이미지, 중심 좌표, 반지름, 색상, 채우기)

    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 버튼 클릭 시 드로잉 시작
        drawing = True # 드로잉 상태로 변경
        color = (0,0,255)  # 빨강
        cv.circle(img,(x,y),brush,color,-1) # 원 그리기 (이미지, 중심 좌표, 반지름, 색상, 채우기)

    elif event == cv.EVENT_MOUSEMOVE and drawing: # 마우스 이동 중 드로잉 상태일 때
        cv.circle(img,(x,y),brush,color,-1) # 원 그리기 (이미지, 중심 좌표, 반지름, 색상, 채우기)

    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP: # 왼쪽 또는 오른쪽 버튼에서 손을 뗄 때
        drawing = False # 드로잉 상태 종료


cv.namedWindow("Paint") # 윈도우 생성
cv.setMouseCallback("Paint", draw) # "Paint" 윈도우에 마우스 이벤트 콜백 함수 등록

while True: # 메인 루프

    temp = img.copy() # 원본 이미지를 복사하여 임시 이미지 생성

    cv.putText(temp,f"brush:{brush}",(10,30), # 텍스트 추가 (이미지, 텍스트, 위치, 폰트, 크기, 색상, 두께)
               cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)# 텍스트 추가 (이미지, 텍스트, 위치, 폰트, 크기, 색상, 두께)

    cv.imshow("Paint",temp) # "Paint" 윈도우에 임시 이미지 표시

    key = cv.waitKey(1) & 0xFF # 키 입력 대기 및 하위 8비트만 추출

    if key == ord('q'): # 'q' 키를 누르면 루프 종료
        break # 프로그램 종료

    elif key == ord('+') or key == ord('='): # '+' 또는 '=' 키를 누르면 브러시 크기 증가
        brush = min(15, brush+1) # 브러시 크기 최대값 15로 제한

    elif key == ord('-') or key == ord('_'): # '-' 또는 '_' 키를 누르면 브러시 크기 감소
        brush = max(1, brush-1) # 브러시 크기 최소값 1로 제한

cv.destroyAllWindows() # 모든 OpenCV 윈도우 닫기