# **Hand Gesture Recognition**

## **Project Overview**

This project is an AI-based **Hand Gesture Recognition** system that utilizes **Mediapipe** for hand tracking, **MobileNet** for gesture classification, and **SSD (Single Shot MultiBox Detector)** for hand detection. The application processes real-time video input to detect and classify hand gestures effectively.

## **Installation Instructions**

To set up and run the application, follow these steps:

### **1\. Clone the Repository**

| git clone https://github.com/your-repo-link/hand-gesture-recognition.git |
| :---- |

| cd hand-gesture-recognition |
| :---- |

### **2\. Install Dependencies**

Ensure you have Python 3.7+ installed, then install the required libraries:

| pip install \-r requirements.txt |
| :---- |

Alternatively, manually install the dependencies:

| pip install opencv-python numpy tensorflow tensorflow-hub mediapipe |
| :---- |

### **3\. Ensure Model availability**

Ensure you have the trained models **`sign_language_mobilenet.h5`** (for classification) and **SSD model** (for hand detection) in the project directory. If not, download or train them following the steps below.

## **Model Training**

### **Training MobileNet for Gesture Classification**

* The MobileNet model is trained on a dataset of hand gesture images.  
* The dataset consists of labeled images for each gesture class.  
* Training involves:  
  1. **Preprocessing:** Resize images to 224x224 and normalize pixel values.  
  2. **Model Training:** Fine-tuning MobileNet on gesture images using transfer learning.  
  3. **Saving the Model:** The trained model is saved as `sign_language_mobilenet.h5`.

### **Training SSD for Hand Detection**

* The **Single Shot MultiBox Detector (SSD)** is used for detecting hands in real-time.  
* It is trained on datasets like COCO or custom-labeled hand images.  
* The SSD model outputs bounding boxes around detected hands.

## **How to run the application:**

To start the application, run:

| python app.py |
| :---- |

### **How It Works:**

1. **Hand Detection:**  
   * The SSD model detects hands in the webcam feed.  
   * Bounding boxes are drawn around detected hands.  
2. **Hand Tracking:**  
   * **Mediapipe** extracts key hand landmarks.  
   * These landmarks refine the detected hand region.  
3. **Gesture Classification:**  
   * The extracted hand region is resized to 224x224.  
   * **MobileNet** predicts the gesture class.  
   * The prediction is displayed on the screen with confidence scores.  
4. **Press 'q' to exit** the application.

## **Expected Output**

Once the application is running, you will see:

* **Detected hands with bounding boxes**.  
* **Predicted gesture displayed with confidence scores**.  
* **Real-time processing for smooth interaction**.

## **Video Output**
https://drive.google.com/file/d/1jwHdtEZCwqZry5ARCsWs-RKnYKWAmtPi/view?usp=sharing

## **Citations & Acknowledgments**

* **Mediapipe Hands API** for hand tracking: [Mediapipe Documentation](https://developers.google.com/mediapipe)  
* **MobileNet** for hand gesture classification: [TensorFlow Official](https://www.tensorflow.org/)  
* **Hand Gesture Recognition Using Mediapipe**: [GitHub Repository](https://github.com/kinivi/hand-gesture-recognition-mediapipe)  
* **MediaPipe Gesture Recognizer Guide**: [Google AI](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python)  
* **OpenCV** for image processing and webcam handling: [OpenCV](https://opencv.org/)


