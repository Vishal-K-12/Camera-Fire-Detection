import cv2

def load_models() : 
    '''
    Fungsi ini digunakan untuk melakukan load terhadap semua model wajah, umur, dan gender.
    '''
    face_proto = "opencv_face_detector.pbtxt" 
    face_model = "opencv_face_detector_uint8.pb"

    age_proto = "age_deploy.prototxt" 
    age_model = "age_net.caffemodel"

    gender_proto = "gender_deploy.prototxt"
    gender_model = "gender_net.caffemodel"

    
    return face_proto, face_model, age_proto, age_model, gender_proto, gender_model

def face_box(net, frame, threshold = 0.7) :
    '''
    Fungsi ini digunakan untuk mendeteksi wajah pada frame yang diberikan.
    
    Hasil return dari fungsi ini adalah frame yang sudah di crop wajahnya dan list dari bounding box dari wajah yang terdeteksi.
    '''
    
    frame_dnn = frame.copy() # frame_dnn adalah frame yang akan digunakan untuk dnn
    frame_height, frame_width = frame_dnn.shape[0], frame_dnn.shape[1]
    
    # Blob adalah array multidimensi yang menyimpan data pada image. Dalam kasus ini blob-nya adalah wajah yang sudah dicrop.
    blob = cv2.dnn.blobFromImage(frame_dnn, 1.0, (300,300), [104, 117, 123], True, False) 
    
    net.setInput(blob) # Memberikan input blob ke dalam neural network
    detections = net.forward() # detections adalah output dari network
    
    blob_boxes = [] # blob_boxes adalah list dari bounding box dari wajah yang terdeteksi
    
    for i in range(detections.shape[2]) : # detections.shape[2] adalah jumlah wajah yang terdeteksi pada frame
        confidence = detections[0, 0, i, 2] # confidence adalah tingkat kepercayaan dari deteksi wajah
        if confidence > threshold : # Jika confidence lebih besar dari batas threshold maka wajah tersebut akan di crop
            x1 = int(detections[0, 0, i, 3] * frame_width) # x1 adalah posisi x dari bounding box
            y1 = int(detections[0, 0, i, 4] * frame_height) # y1 adalah posisi y dari bounding box
            x2 = int(detections[0, 0, i, 5] * frame_width)  
            y2 = int(detections[0, 0, i, 6] * frame_height)
            blob_boxes.append([x1, y1, x2, y2]) # Menambahkan bounding box dari wajah yang terdeteksi ke dalam list blob_boxes 
            cv2.rectangle(frame_dnn, (x1, y1), (x2, y2), (0, 255, 0), 1) # Membuat bounding box pada frame_dnn
            
    return frame_dnn, blob_boxes

face_proto = "opencv_face_detector.pbtxt" # path to the face detection model
face_model = "opencv_face_detector_uint8.pb" # path to the face detection model

age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) 

age_list = ['(0-3)', '(4-7)', '(8-14)', '(15-24)', '(25-32)', '(33-47)', '(48-53)', '(54-100)']
gender_list = ['Male', 'Female']

# Load network
age_net = cv2.dnn.readNet(age_model, age_proto) # network untuk age detection
gender_net = cv2.dnn.readNet(gender_model, gender_proto) # network untuk gender detection
face_net = cv2.dnn.readNet(face_model, face_proto) # network untuk face detection

cam = cv2.VideoCapture(0) # open camera untuk capture video

while True : 
    ret, frame = cam.read()
    frame_face, blob_boxes = face_box(face_net, frame) # Frame Face adalah frame yang sudah di crop wajahnya dan blob_boxes adalah list dari bounding box dari wajah yang terdeteksi
    
    for bbox in blob_boxes :
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] # Crop wajah dari frame 
        
        try :
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        except : 
            print("Face is not detected")
            
        gender_net.setInput(blob)
        gender_predict = gender_net.forward() # Melakukan forward pass terhadap neural network gender_net. Forward Pass artinya memberi input data ke dalam neural network dan mengeluarkan output dari neural network. 
        
        gender = gender_list[gender_predict[0].argmax()] # mencari nilai maksimum dari list peluang gender_predict. Nilai maksimum tersebut selanjutnya menjadi predicted gender
        
        age_net.setInput(blob)
        age_predict = age_net.forward() 
        age = age_list[age_predict[0].argmax()] # Mencari nilai maksimum dari list peluang pada age_predict

        label = f"Gender : {gender} , Age : {age}"
        
        cv2.rectangle(frame_face, (bbox[0], bbox[1]), (bbox[2], bbox[1]), (0, 255, 0), 2)
        cv2.putText(frame_face, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
    cv2.imshow("Age Gender Recognition", frame_face)
    k = cv2.waitKey(1)
        
    if k == ord('q') :
        break
        
cam.release()
cv2.destroyAllWindows()