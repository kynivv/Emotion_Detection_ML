import cv2
import numpy as np
from keras.models import model_from_json
from pathlib import Path

class Model_test():
    def main():
        emotion_dict = {
            0: 'Angry',
            1: 'Disgusted',
            2: 'Fearful',
            3: 'Happy',
            4: 'Neutral',
            5: 'Sad',
            6: 'Surprised'
        }
    
        BASE_DIR = Path(__file__).resolve().parent
    
        # Load model architecture
        model_json = (BASE_DIR / "emotion_model.json").read_text(encoding="utf-8")
        emotion_model = model_from_json(model_json)
    
        # Load model weights
        emotion_model.load_weights(BASE_DIR / "emotion_model.h5")
    
        # Webcam 
        def open_first_available_camera():
            """
            Tries to open the first available camera by iterating through indices.
            """
            # Iterate through potential camera indices
            for index in range(10): 
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    print(f"Successfully opened camera at index {index}")
                    return cap
                else:
                    cap.release()
                    print(f"Camera at index {index} is not available")
    
            print("No active webcam found.")
            return None
    
        cap = open_first_available_camera()
    
    
        while True:
            # Face Haar cascade
            ret, frame = cap.read()
            frame = cv2.resize(frame, dsize=(1024, 576))
            if not ret:
                break
            face_detector = cv2.CascadeClassifier(BASE_DIR / 'haarcascades' / 'haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame_small = cv2.resize(gray_frame, None, fx=0.5, fy=0.5)
    
    
            # Detecting faces
            num_faces = face_detector.detectMultiScale(gray_frame_small, scaleFactor=1.5, minNeighbors = 5)
            num_faces *= 2 # scale boxes back
    

            # Preprocess each face
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-5), (x+w, y+h+10), (0, 255, 0), 2)
                roi_gray_frame = gray_frame[y:y+h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
    
    
                # Predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 250, 0), 1, cv2.LINE_AA)
    
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # quit app on pressing "Q"
                break
        cap.release()
        cv2.destroyAllWindows()
