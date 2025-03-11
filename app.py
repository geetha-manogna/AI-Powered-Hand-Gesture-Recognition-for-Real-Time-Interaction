import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# Load trained model
model = tf.keras.models.load_model("sign_language_mobilenet.h5")

# Define class names (Update according to your dataset)
class_names = ["A", "B", "C", "D", "E","F","G","H","I","J"]

# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start webcam feed
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Allow webcam to warm up
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame for natural view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get bounding box around hand
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Expand box slightly
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                # Resize and preprocess hand image
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = np.expand_dims(hand_img, axis=0) / 255.0

                # Predict sign language gesture
                prediction = model.predict(hand_img)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Add label with background
                label = f"{predicted_class} ({confidence:.2f}%)"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                label_x = x_min
                label_y = y_min - 10 if y_min - 10 > 10 else y_min + 20

                cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5),
                              (label_x + label_size[0], label_y + 5), (0, 255, 0), -1)
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display results
    cv2.imshow("Sign Language Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera & close windows
cap.release()
cv2.destroyAllWindows()
