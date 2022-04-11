import dlib
import cv2
import numpy as np

# 랜드마크 만들기
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))


# create face detector, predictor
detector = dlib.get_frontal_face_detector()     #얼굴 인식용 클래스
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   # 얼굴 랜드마크용 클래스


# create VideoCapture object (input the video)
# 0 입력시 노트북 카메라 실행
vid_in = cv2.VideoCapture(0)          
# "---" for the video file
#vid_in = cv2.VideoCapture("baby_vid.mp4")

# capture the image in an infinite loop
# -> make it looks like a video
while True:
    # Get frame from video
    # get success : ret = True / fail : ret= False
    ret, image_o = vid_in.read()

   # 카메라 크기조절
    image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)   #이미지크기 조절 함수
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식하기 ( 1 은 인식범위)
    face_detector = detector(img_gray, 1)
    # 얼굴 인식되면 얼굴갯수 출력 (혼자면 1출력)   인식x -> O출력 
    print("The number of faces detected : {}".format(len(face_detector)))   

    # 얼굴개수만큼 반복하여 윤곽표시
    # 하나의 얼굴은 하나의 윤곽
    for face in face_detector:
        # 얼굴인식하면 사각형 그리기
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
                      (0, 0, 255), 3)

        # make prediction and transform to numpy array
        landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기

        #create list to contain landmarks
        landmark_list = []

        # append (x, y) in landmark_list
        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)


    cv2.imshow('result', image)

    # wait for keyboard input
    key = cv2.waitKey(1)

    # esc 눌러서 종료
    if key == 27:
        break

vid_in.release()