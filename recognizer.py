import cv2
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect frames of different sizes, list of faces rectangles
    faces = face_cas.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if faces is ():
        return frame, []
    for (x, y, w, h) in faces:
        croped_face = gray[y:y+h, x:x+w]
        croped_face = cv2.resize(croped_face, (200,200))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame, croped_face

cap = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

id = 0

# font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame= cap.read()
    img, cropped_face = get_face(frame)
    img = cv2.flip(img,1)
    try:
        id, result = recognizer.predict(cropped_face)
        if result < 500:
            confidence = int(100*(1- result/300))
            display_string = str(confidence) + '% ' +str(id)+' is confident'
            cv2.putText(img, display_string, (100,120), font, 1,(255,0,0),2)
            if confidence > 80:
                cv2.putText(img, 'Unclocked', (250,450), font, 1, (0,255,0),2)
                cv2.imshow('image', img)
            else:
                cv2.putText(img, 'locked', (250,450), font, 1, (0,0,255),2)
                cv2.imshow('image', img)
    except:
        cv2.putText(img, 'face not found', (220,120), font, 1,(0,0,255), 1)
        cv2.putText(img, 'locked', (250,450), font, 1, (0,0,255),2)
        cv2.imshow('image', img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
