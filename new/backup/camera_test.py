import cv2
import numpy as np
from picamera2 import Picamera2

# Picamera2 객체 생성
picam2 = Picamera2()

# 카메라 설정
picam2.configure(picam2.create_still_configuration())
picam2.start()

# OpenCV 윈도우 생성
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

try:
    while True:
        # 카메라에서 이미지 캡처
        frame = picam2.capture_array()
        
        # OpenCV 윈도우에 이미지 표시
        cv2.imshow("Camera", frame)
        
        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 카메라 정리 및 윈도우 종료
    picam2.stop()
    cv2.destroyAllWindows()
