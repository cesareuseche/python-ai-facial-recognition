import cv2

# Loading the pre-trained data on face frontals from opencv (the haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

#Choose an image to detect faces in 
img = cv2.imread('LBJ.jpg')

# Must converted to grayscale
#grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# video capture from webcam
webcam = cv2.VideoCapture(0)

## Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    #Must Convert to graycale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecting faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0), 5 )

    # Display the images with the faces
    cv2.imshow('My face Detector', frame)
    key = cv2.waitKey(1)

    # Stops if Q key is pressed
    if key==81 or key==113:
        break   
# Release VideoCapture object
webcam.release()
#print(face_coordinates)



print('running another test')
