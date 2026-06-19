import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


print("MNIST 모델 학습 중...")

# 0~9까지의 손글씨 데이터셋(MNIST) 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 0~255 사이의 픽셀 값을 0~1 사이로 정규화 (학습 속도와 성능 향상)
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN 모델 입력에 맞게 형태 변환 (데이터 개수, 가로, 세로, 채널수(흑백=1))
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 순차적 레이어 구성을 위한 Sequential 모델 정의
model = models.Sequential([
    # 특징 추출 레이어 1: 3x3 크기의 필터 32개 사용
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)), # 이미지 크기를 절반으로 줄여 주요 특징 강화

    # 특징 추출 레이어 2: 필터 64개로 더 복잡한 모양을 추출
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"), # 완전히 연결된 Dense 레이어 (특징 결합)
    layers.Dense(10, activation="softmax") # 최종 출력층: 0~9까지 10개 클래스의 확률 출력
])

# 모델 컴파일: 최적화 알고리즘 = Adam, 손실 함수 설정
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 모델 학습 진행 (전체 데이터를 총 3번 반복 학습)
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
print("학습 완료!")



def predict_digit(canvas):
    # 흑백으로 변환
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # 사용자가 글씨를 쓴 영역 좌표 찾기
    coords = cv2.findNonZero(gray)

    if coords is None:
        return None

    # 글씨가 존재하는 최소한의 사각형 영역(Bounding Box) 추출
    x, y, w, h = cv2.boundingRect(coords)
    digit = gray[y:y+h, x:x+w]

    # MNIST와 유사하게 만들기 위해 정사각형 형태로 여백(Padding) 추가
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)

    # 정중앙에 글씨가 배치되도록 오프셋 계산
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    # 외곽 여백을 확보하며 20x20 사이즈로 재배열함
    digit = cv2.resize(square, (20, 20))

    # 최종 28x28 크기의 빈 도화지 정중앙에 20x20 글씨를 삽입
    mnist_img = np.zeros((28, 28), dtype=np.uint8)
    mnist_img[4:24, 4:24] = digit

    # 딥러닝 모델 입력용 전처리
    mnist_img = mnist_img / 255.0
    mnist_img = mnist_img.reshape(1, 28, 28, 1)

    # 인공지능 모델 예측 실행
    pred = model.predict(mnist_img, verbose=0)
    result = np.argmax(pred) # 가장 확률이 높은 숫자 (0~9)
    confidence = np.max(pred) # 해당 예측의 신뢰도 (0.0 ~ 1.0)

    return result, confidence



cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if not ret:
    print("카메라를 열 수 없습니다.")
    cap.release()
    exit()

frame = cv2.flip(frame, 1)

# 사용자가 마우스로 추적할 물체(손가락 끝, 파란색 뚜껑 등)를 사각형 드래그로 선택 --> 해당 객체를 ROI로 설정
roi = cv2.selectROI("Select Tracking Object", frame, False)
cv2.destroyWindow("Select Tracking Object")

x, y, w, h = roi
track_window = (x, y, w, h)

# ROI 설정 및 색상 가중치 계산을 위한 HSV 변환
roi_frame = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

# 특정 색상 범위 마스크 생성
mask = cv2.inRange(hsv_roi, np.array((0., 30., 30.)), np.array((180., 255., 255.)))

# 채도(H) 채널에 대한 히스토그램 계산 및 정규화
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# CamShift 알고리즘 반복 중단 조건 설정 (10번 반복하거나 정확도가 1 이상 개선 안 되면 중단)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 글씨를 기록할 빈 검은색 캔버스 생성
canvas = np.zeros_like(frame)

xp, yp = 0, 0
drawing = True
predicted_digit = "None"
confidence_text = ""

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 전체 이미지에서 내가 선택한 물체의 색상 분포와 일치하는 확률 맵 계산
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # CamShift 알고리즘 실행
    ret_cam, track_window = cv2.CamShift(dst, track_window, term_crit)

    # 화면에 초록색 사각형 그리기
    pts = cv2.boxPoints(ret_cam)
    pts = np.intp(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 추적된 객체의 중심점(cx, cy) 추출
    cx = int(ret_cam[0][0])
    cy = int(ret_cam[0][1])
    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1) # 중심점에 빨간 점 표시

    # 숫자 그리기 로직
    if drawing:
        if xp == 0 and yp == 0:
            xp, yp = cx, cy
        cv2.line(canvas, (xp, yp), (cx, cy), (255, 255, 255), 12)
        xp, yp = cx, cy
    else:
        xp, yp = 0, 0

    # 원본 웹캠 영상 위에 글씨가 그려진 캔버스 레이어를 합성
    display = cv2.add(frame, canvas)

    # 화면에 AI 예측 정보 및 안내 가이드 텍스트 출력
    cv2.putText(display, f"Prediction: {predicted_digit}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, confidence_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, "SPACE: Predict | C: Clear | D: Draw On/Off | ESC: Exit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 통합 화면을 윈도우 창에 띄우기
    cv2.imshow("CamShift MNIST Digit Recognition", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

    elif key == ord('c'):
        canvas[:] = 0
        predicted_digit = "None"
        confidence_text = ""
        xp, yp = 0, 0

    elif key == ord('d'):
        drawing = not drawing
        xp, yp = 0, 0

    elif key == 32:
        result = predict_digit(canvas)
        if result is not None:
            digit, confidence = result
            predicted_digit = str(digit)
            confidence_text = f"Confidence: {confidence:.2f}"
        else:
            predicted_digit = "None"
            confidence_text = "No drawing detected"

cap.release()
cv2.destroyAllWindows()