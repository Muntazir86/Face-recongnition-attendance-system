# Import OpenCV2 for image processing
import cv2
import os

def assure_path_exists():
    path = "dataset/"
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        croped_face = gray[y:y+h, x:x+w]
    return croped_face

face_id = input('enter your id ')
# Start capturing video
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize sample face image
count = 0

assure_path_exists()

# Start looping
while True:

    # Capture video frame
    rat, image_frame = vid_cam.read()
    croped_face = get_face(image_frame)
    if croped_face is not None:
        croped_face = cv2.resize(croped_face,(200,200))

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", croped_face)
        cv2.putText(croped_face, str(count), (10, 10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('croped_face', croped_face)
        # Increment sample face image
        count += 1
    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count >= 100:

        print("Successfully Captured")
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
