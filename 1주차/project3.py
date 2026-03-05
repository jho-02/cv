import cv2 as cv 

img = cv.imread("welcome.jpg") # 이미지 불러오기

if img is None: # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지 없음") # 이미지가 없으면 메시지 출력
    exit() # 프로그램 종료


img = cv.resize(img, None, fx=0.5, fy=0.5) # 이미지 축소 (fx, fy: 가로, 세로 축소 비율, interpolation=cv.INTER_AREA: 축소할 때 좋은 품질을 제공하는 보간 방법)

orig = img.copy() # 원본 이미지 복사 (ROI 선택 후 원본 이미지를 유지하기 위해)

drawing = False #   마우스 드래그 상태를 나타내는 변수 (False: 드래그 중 아님, True: 드래그 중)
x0, y0 = -1, -1 # 드래그 시작점의 좌표를 저장하는 변수 (초기값은 -1로 설정하여 유효하지 않은 좌표로 표시)
roi = None # 선택된 ROI(Region of Interest)를 저장하는 변수 (초기값은 None으로 설정하여 ROI가 선택되지 않았음을 나타냄)


def mouse(event, x, y, flags, param): # 마우스 이벤트를 처리하는 콜백 함수 (event: 마우스 이벤트 종류, x, y: 마우스 좌표, flags: 이벤트 플래그, param: 추가 매개변수)
    global drawing, x0, y0, img, roi # 전역 변수 사용 (drawing, x0, y0, img, roi 변수를 함수 내에서 수정하기 위해 global 키워드 사용)

   
    if event == cv.EVENT_LBUTTONDOWN and roi is None: # 왼쪽 버튼이 눌렸고 ROI가 선택되지 않은 경우 (드래그 시작)
        drawing = True # 드래그 상태로 변경 (True로 설정하여 드래그 중임을 나타냄)
        x0, y0 = x, y # 드래그 시작점의 좌표 저장 (x, y는 마우스 이벤트로 전달된 현재 좌표)

    elif event == cv.EVENT_MOUSEMOVE and drawing: # 마우스가 움직이고 드래그 중인 경우 (드래그 진행)
        temp = orig.copy() # 원본 이미지 복사 (드래그 진행 중에 원본 이미지를 유지하기 위해)
        cv.rectangle(temp, (x0, y0), (x, y), (0,255,0), 2) # 드래그 시작점 (x0, y0)과 현재 마우스 좌표 (x, y)를 이용하여 사각형 그리기 (초록색, 두께 2)
        cv.imshow("Image", temp) # 드래그 진행 중인 이미지를 화면에 표시 (temp 이미지에는 현재 드래그 상태가 반영된 사각형이 그려져 있음)

    elif event == cv.EVENT_LBUTTONUP and drawing: # 왼쪽 버튼이 떼어지고 드래그 중인 경우 (드래그 종료)
        drawing = False # 드래그 상태 종료 (False로 설정하여 드래그 중이 아님을 나타냄)

        x1, y1 = x, y # 드래그 종료점의 좌표 저장 (x, y는 마우스 이벤트로 전달된 현재 좌표)

        x_min, x_max = min(x0,x1), max(x0,x1) # 드래그 시작점과 종료점의 x 좌표를 이용하여 최소값과 최대값 계산 (x_min: 드래그 영역의 왼쪽 경계, x_max: 드래그 영역의 오른쪽 경계)
        y_min, y_max = min(y0,y1), max(y0,y1) # 드래그 시작점과 종료점의 y 좌표를 이용하여 최소값과 최대값 계산 (y_min: 드래그 영역의 상단 경계, y_max: 드래그 영역의 하단 경계)

        roi = orig[y_min:y_max, x_min:x_max] # 선택된 ROI(Region of Interest)를 원본 이미지에서 추출 (y_min:y_max, x_min:x_max 영역을 슬라이싱하여 roi 변수에 저장)

        img = orig.copy() #  원본 이미지 복사 (ROI 선택 후 원본 이미지를 유지하기 위해)
        cv.rectangle(img, (x_min,y_min), (x_max,y_max), (0,255,0), 2) # 선택된 ROI 영역을 원본 이미지에 사각형으로 표시 (초록색, 두께 2)

        cv.imshow("ROI", roi) # 선택된 ROI를 화면에 표시 (roi 이미지에는 선택된 영역이 포함되어 있음)


cv.namedWindow("Image") # "Image"라는 이름의 윈도우 생성 (마우스 이벤트를 처리하기 위해 윈도우가 필요함)
cv.setMouseCallback("Image", mouse) # "Image" 윈도우에 마우스 콜백 함수 등록 (마우스 이벤트가 발생할 때 mouse 함수가 호출되도록 설정)

while True: # 무한 루프를 사용하여 프로그램이 종료될 때까지 계속 실행 (사용자가 'q' 키를 눌러 종료할 때까지)

    cv.imshow("Image", img) # 현재 이미지를 화면에 표시 (img 이미지에는 ROI 선택 상태가 반영되어 있음)

    key = cv.waitKey(1) & 0xFF # 키 입력 대기 (1ms 동안 대기, & 0xFF: 키 입력을 8비트로 제한하여 처리)

    if key == ord('q'):# 'q' 키가 눌렸을 때 루프 종료 (프로그램 종료)
        break # 루프 종료 (프로그램 종료)
 
    elif key == ord('r'):# 'r' 키가 눌렸을 때 이미지와 ROI 초기화 (리셋)
        img = orig.copy() # 원본 이미지 복사 (이미지와 ROI를 초기 상태로 되돌리기 위해)
        roi = None # ROI 초기화 (None으로 설정하여 ROI가 선택되지 않은 상태로 되돌리기)
        print("리셋") # 리셋 메시지 출력 (이미지와 ROI가 초기화되었음을 알림)

    elif key == ord('s'): # 's' 키가 눌렸을 때 선택된 ROI 저장 (save)
        if roi is not None: # ROI가 선택된 경우 (roi 변수가 None이 아닌 경우)
            cv.imwrite("roi.png", roi) # 선택된 ROI를 "roi.png" 파일로 저장 (roi 이미지가 roi.png 파일로 저장됨)
            print("roi.png 저장됨") # 저장 완료 메시지 출력 (roi.png 파일이 저장되었음을 알림)
        else:  # ROI가 선택되지 않은 경우 (roi 변수가 None인 경우)
            print("ROI 없음") # ROI가 선택되지 않았음을 알리는 메시지 출력 (저장할 ROI가 없음을 알림)

cv.destroyAllWindows() # 모든 OpenCV 윈도우 닫기 (프로그램 종료)