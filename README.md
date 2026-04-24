# SignLanguageTranslator

A real-time sign language translator that uses computer vision and machine learning to translate hand gestures into speech.

## Features

- Real-time hand tracking using webcam  
- Sign classification and recognition using a machine learning model  
- Text-to-speech output for detected words  
- Simple graphical start screen  

## Code Explanation

1. Hand landmarks are detected using MediaPipe (21 points per hand)  
2. Landmarks are normalized to ensure consistency
3. A random forest algorithm is trained using a data set
4. We use the model to recognize letter in real time
6. Words are spoken
7. Uses edge-tts for speech generation
9. Automatically speaks detected words  

## Installation

Clone the repository

Install dependencies:

pip install opencv-python mediapipe numpy scikit-learn joblib pygame edge-tts customtkinter  

## Using the program

### Controls

- T → Start translating  
- S → Stop translating  
- Q → Quit  

## 📁 Project Structure

.
├── main.py  
├── training.py  
├── dataset_landmarks/  
└── model/  
    └── modelo_senas.pkl  

## 🛠️ Libraries used

- OpenCV  
- MediaPipe  
- Scikit-learn  
- NumPy  
- Pygame  
- edge-tts  
- CustomTkinter  
