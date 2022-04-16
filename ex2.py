import dlib
import cv2
import pyrebase
import numpy as np
import os
import ex3
def MyFace(a):
    
  config = {  "apiKey": "AIzaSyBTGrklWDyt2cWXL3XVgV7AvJLeQXX60iQ",
    "authDomain": "smart-doorlock-c9e00.firebaseapp.com",
    "databaseURL": "https://smart-doorlock-c9e00-default-rtdb.firebaseio.com",
    "projectId": "smart-doorlock-c9e00",
    "storageBucket": "smart-doorlock-c9e00.appspot.com",
    "messagingSenderId": "404335392148",
    "appId": "1:404335392148:web:75c6d6f8572fea4dbf4b81",
    "measurementId": "G-7TB1L2YDF4",
    "serviceAccount": "smart-doorlock-c9e00-firebase-adminsdk-qge67-165a3f65b6.json"}

  # firebase 사진 불러오기
  firebase = pyrebase.initialize_app(config)
  auth = firebase.auth()
  storage = firebase.storage()
  all_files = storage.child().list_files()




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









  #upload
  #storage.child(my_image).put(my_image)
  #download
  #storage.child(my_image).download(filename="facefolder/1.jpg", path=os.path.basename(my_image))

  for file in all_files:      # 사진 하나씩 firebase 에서 다운로드
    try:
      file.download_to_filename("facefolder/1.jpg")

      image = cv2.imread("facefolder/1.jpg")   # 이미지 출력
      img = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)   #이미지크기 조절 함수
      img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #색

      detector = dlib.get_frontal_face_detector()     #얼굴 인식용 클래스
      predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   # 얼굴 랜드마크용 클래스  

      face_detector = detector(img_gray, 1)           #얼굴인식개수

      for face in face_detector :
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()),
                  (0, 0, 255), 3)
        landmarks = predictor(img, face)  # 얼굴에서 68개 점 찾기                  
        landmark_list = []
        for p in landmarks.parts():
          landmark_list.append([p.x, p.y])
          cv2.circle(img, (p.x, p.y), 2, (0, 255, 0), -1)

        ex3.ucle(a,landmark_list)                      #얼굴 유사도 계산을위해 좌표값 보내기
        print(landmark_list)
        print()    

    #cv2.imshow('result', img)
    #cv2.waitKey()   

     
    except:
      print("Download Failed")





