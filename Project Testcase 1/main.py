import cv2
import numpy as np
from keras.models import load_model

# Load the emotion model
model = load_model('emotion_model.h5')

# Set up the cascades for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set up the emotions labels
emotions = ["happy", "sad", "angry", "neutral", "surprised", "disgusted"]

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face
    for (x,y,w,h) in faces:
        # Crop the face and resize it to 48x48 pixels
        face_cropped = gray[y:y+h, x:x+w]
        face_cropped = cv2.resize(face_cropped, (48, 48))

        # Predict the emotion using the emotion model
        prediction = model.predict(face_cropped.reshape(1, 48, 48, 1))
        emotion_index = np.argmax(prediction)
        emotion = emotions[emotion_index]

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Wait for the user to press a key
    key = cv2.waitKey(1)
    if key == 27: # Esc key
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()