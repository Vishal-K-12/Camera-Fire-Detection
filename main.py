import cv2
import winsound as ws
from tensorflow import keras

def preprocessing_image(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
    img = cv2.resize(img,(196,196))  # resize
    img = img / 255 #scale
    return img 

def is_fire_detected(probability) :
    if probability > 0.95 : return True
    else : return False
    

model = keras.models.load_model("fire_detector.h5")

cap = cv2.VideoCapture(0)

while True :
    _, frame = cap.read()

    img = preprocessing_image(frame)
    img = img[None, ...]
    prediction = model.predict(img)
    prediction = prediction.reshape(-1)
    print(prediction)
    
    if is_fire_detected(prediction) :
        label = f"Fire! | Probability : {prediction * 100}"
        ws.Beep(1000, 1000)
    else : label = f"No fire | Probability : {prediction * 100}"

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()