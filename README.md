# Facial Emotion Detector

This project uses your webcam to perform real-time facial emotion analysis and displays the results in a local web interface.

It's built with Python, **OpenCV**, and the **DeepFace** library for analysis, with a **Flask** server to stream the video to a simple HTML/Tailwind CSS front-end.


### Tech Stack

* **Python 3.10 / 3.11** (Python 3.12+ may cause errors)
* **Flask:** To create the web server and stream video.
* **OpenCV:** To capture and process the webcam feed.
* **DeepFace:** To perform the emotion analysis.
* **HTML / Tailwind CSS:** For the simple web interface.

---

### How to Run

**1. Clone the Repository**
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

**2. Create and Activate a Virtual Environment**
```bash
# Create venv
python3 -m venv venv

# Activate venv (macOS/Linux)
source venv/bin/activate
# (Windows)
# .\venv\Scripts\activate
```
**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. CRITICAL: Manually Download the Model The deepface library sometimes fails to download the model file automatically. You must download it manually.**
 * Download this file: https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5

 * Place it in this exact folder: /Users/<your_username>/.deepface/weights/

(On Mac, press Cmd + Shift + . in your Home folder to see the hidden .deepface directory. You may need to create the weights folder inside it.)


**5. Run the Server**
```bash
python emotion_detector.py
```
You will see output indicating the server is running on port 5001.

**6. Open the Interface Open your web browser and go to: https://www.google.com/search?q=http://127.0.0.1:5001**

You should now see your webcam feed with live emotion analysis.