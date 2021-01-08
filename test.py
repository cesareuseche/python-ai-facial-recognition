import cv2

# Loading the pre-trained data on face frontals from opencv (the haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

print('running another test')
