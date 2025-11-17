import cv2
from deepface import DeepFace
import time
from typing import List, Dict, Any, cast
from flask import Flask, render_template, Response
import logging

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
# Suppress informational logs from Flask
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# --- CONFIGURATION (from original script) ---
FRAME_SKIP = 10 
DETECTOR_BACKEND = 'opencv' 
CAMERA_INDEX = 1 # Use '1' for Mac's built-in, '0' for others

# --- GLOBAL VARS ---
# We use a global to initialize the camera only once
cap = None

def initialize_camera():
    """Initializes the camera object."""
    global cap
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX) 
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam at index {CAMERA_INDEX}.")
        print("Camera initialized successfully.")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        print("Tip: Try changing CAMERA_INDEX to 0, 1, or 2.")
        cap = None # Ensure cap is None if init fails

def generate_frames():
    """
    This is the core generator function that captures video,
    runs emotion analysis, draws on the frame, and yields it
    as a JPEG for the web stream.
    """
    global cap
    if cap is None:
        print("Camera not initialized. Attempting to initialize...")
        initialize_camera()
        if cap is None: # If still None after attempt, yield nothing
            return

    frame_count = 0
    last_detected_faces: List[Dict[str, Any]] = [] 

    while True:
        # Read a single frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame. Re-initializing camera...")
            # Attempt to re-initialize camera if it disconnects
            initialize_camera()
            time.sleep(1)
            continue

        # Flip the frame horizontally (for a mirror-like view)
        frame = cv2.flip(frame, 1)
        
        # We only run the analysis every 'FRAME_SKIP' frames
        if frame_count % FRAME_SKIP == 0:
            try:
                detected_faces_raw = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'], 
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False, 
                    silent=True
                )
                detected_faces = cast(List[Dict[str, Any]], detected_faces_raw)
                last_detected_faces = detected_faces
            except Exception as e:
                # print(f"Error during analysis: {e}") # Uncomment for debugging
                last_detected_faces = [] # Clear faces if analysis fails

        # Draw rectangles and text for all faces found in the last analysis
        if last_detected_faces:
            for face_data in last_detected_faces:
                region = face_data.get('region')
                emotion_raw = face_data.get('dominant_emotion')

                if not region or not emotion_raw:
                    continue 

                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                emotion = emotion_raw.capitalize()
                
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # --- Draw the text (with a nice background) ---
                text = f"Emotion: {emotion}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y - 5), (0, 255, 0), -1)
                cv2.putText(frame, text, (x, y - 10), font, font_scale, (0, 0, 0), thickness)

        frame_count += 1

        # --- MJPEG STREAMING PART ---
        # Encode the frame as a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        # Convert the buffer to bytes
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the required multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """
    This route serves the main HTML page.
    """
    # Renders the 'index.html' file from the 'templates' folder
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    This route streams the video frames from the generate_frames() function.
    """
    # 'multipart/x-mixed-replace' is the magic mimetype for MJPEG streams
    return Response(
        generate_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def main():
    """
    Main function to initialize the camera and start the Flask server.
    """
    print("Initializing camera...")
    initialize_camera()
    
    if cap is None:
        print("Failed to initialize camera. Exiting.")
        return

    print("Starting Flask server...")
    print("Open http://127.0.0.1:5001 in your browser.")
    # 'threaded=True' allows Flask to handle multiple requests
    # (e.g., serving the HTML and the video stream simultaneously)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()