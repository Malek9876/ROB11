import cv2
import numpy as np
import os
import time
import gradio as gr
from skimage.feature import local_binary_pattern, hog
import joblib 
from tensorflow.keras.models import load_model

# --- Configuration ---
PROCESS_EVERY_N_FRAMES = 5

# --- Model and Asset Loading ---
# Load CNN Model
model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
if not os.path.exists(model_path):
    print(f"Error: Pre-trained CNN model not found at '{model_path}'")
    cnn_model = None
else:
    try:
        cnn_model = load_model(model_path, compile=False)
        print("CNN model loaded successfully.")
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        cnn_model = None

# Load Haar Cascade for face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Haar Cascade loaded successfully.")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    face_cascade = None

# Load our trained models
try:
    print("Loading trained LBP+KNN model from 'knn_model.joblib'...")
    knn_model = joblib.load('knn_model.joblib')
    print("LBP+KNN model loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'knn_model.joblib' not found. Please run the train_models.py script first.")
    knn_model = None

try:
    print("Loading trained HOG+SVM model from 'svm_model.joblib'...")
    svm_model = joblib.load('svm_model.joblib')
    print("HOG+SVM model loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'svm_model.joblib' not found. Please run the train_models.py script first.")
    svm_model = None


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --- Utility Functions ---
def get_face(frame):
    if frame is None:
        return None, None, "Input is empty"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None, None, "No face detected"
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], (x, y, w, h), None

def draw_on_face(frame, coords, label):
    output_frame = frame.copy()
    if coords:
        (x, y, w, h) = coords
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

# --- Master Prediction Functions ---
def process_frame_for_stream(frame, frame_counter, last_knn_text, last_svm_text, last_cnn_text, last_coords):
    frame_counter = int(frame_counter + 1)
    
    if frame is None:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return frame_counter, last_knn_text, last_svm_text, last_cnn_text, last_coords, blank_frame, blank_frame, blank_frame, last_knn_text, last_svm_text, last_cnn_text

    if frame_counter % PROCESS_EVERY_N_FRAMES != 0 and last_coords is not None:
        try:
            knn_pred_label = last_knn_text.splitlines()[1].split(': ')[1]
            svm_pred_label = last_svm_text.splitlines()[1].split(': ')[1]
            cnn_pred_label = last_cnn_text.splitlines()[1].split(': ')[1]
        except IndexError:
            knn_pred_label, svm_pred_label, cnn_pred_label = "--", "--", "--"
            
        knn_frame = draw_on_face(frame, last_coords, f"LBP+KNN: {knn_pred_label}")
        svm_frame = draw_on_face(frame, last_coords, f"HOG+SVM: {svm_pred_label}")
        cnn_frame = draw_on_face(frame, last_coords, f"CNN: {cnn_pred_label}")
        return frame_counter, last_knn_text, last_svm_text, last_cnn_text, last_coords, knn_frame, svm_frame, cnn_frame, last_knn_text, last_svm_text, last_cnn_text

    face, coords, error_msg = get_face(frame)
    
    if error_msg:
        blank_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        error_text = f"Latency: 0.00 ms\nPrediction: --\n{error_msg}"
        return frame_counter, error_text, error_text, error_text, None, blank_frame, blank_frame, blank_frame, error_text, error_text, error_text

    start_time_knn = time.time()
    lbp = local_binary_pattern(cv2.resize(face, (128, 128)), P=24, R=8, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    emotion_knn = EMOTIONS[knn_model.predict(hist.reshape(1, -1))[0]]
    latency_knn = (time.time() - start_time_knn) * 1000
    knn_frame = draw_on_face(frame, coords, f"LBP+KNN: {emotion_knn}")
    knn_text = f"Latency: {latency_knn:.2f} ms\nPrediction: {emotion_knn}\nConfidence: N/A"

    start_time_svm = time.time()
    hog_features = hog(cv2.resize(face, (64, 128)), orientations=9, pixels_per_cell=(8, 😎, cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    emotion_svm = EMOTIONS[svm_model.predict(hog_features.reshape(1, -1))[0]]
    latency_svm = (time.time() - start_time_svm) * 1000
    svm_frame = draw_on_face(frame, coords, f"HOG+SVM: {emotion_svm}")
    svm_text = f"Latency: {latency_svm:.2f} ms\nPrediction: {emotion_svm}\nConfidence: N/A"

    start_time_cnn = time.time()
    face_roi_cnn = cv2.resize(face, (64, 64))
    face_roi_cnn = face_roi_cnn.astype("float") / 255.0
    face_roi_cnn = np.expand_dims(face_roi_cnn, axis=-1)
    face_roi_cnn = np.expand_dims(face_roi_cnn, axis=0)
    preds_cnn = cnn_model.predict(face_roi_cnn, verbose=0)[0]
    emotion_cnn = EMOTIONS[preds_cnn.argmax()]
    confidence_cnn = np.max(preds_cnn)
    latency_cnn = (time.time() - start_time_cnn) * 1000
    cnn_frame = draw_on_face(frame, coords, f"CNN: {emotion_cnn} ({confidence_cnn:.2f})")
    cnn_text = f"Latency: {latency_cnn:.2f} ms\nPrediction: {emotion_cnn}\nConfidence: {confidence_cnn:.2f}"
    
    return frame_counter, knn_text, svm_text, cnn_text, coords, knn_frame, svm_frame, cnn_frame, knn_text, svm_text, cnn_text

def process_upload_image(frame):
    outputs = process_frame_for_stream(frame, -1, "", "", "", None)
    return outputs[5:]

# --- Gradio Blocks Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Real-time Facial Expression Detection: 3-Model Comparison")
    
    frame_counter_state = gr.State(0)
    last_knn_text_state = gr.State("Initializing...\nPrediction: --\nConfidence: --")
    last_svm_text_state = gr.State("Initializing...\nPrediction: --\nConfidence: --")
    last_cnn_text_state = gr.State("Initializing...\nPrediction: --\nConfidence: --")
    last_coords_state = gr.State(None)

    with gr.Tabs():
        with gr.TabItem("Live Webcam"):
            webcam_input = gr.Image(sources="webcam", type="numpy", streaming=True, label="Webcam Feed")
        with gr.TabItem("Upload Image"):
            upload_input = gr.Image(sources="upload", type="numpy", label="Upload an Image")
            submit_btn = gr.Button("Process Image")

    gr.Markdown("## Model Outputs")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### LBP + KNN Output")
            knn_image_output = gr.Image(label="LBP+KNN Detection")
            knn_text_output = gr.Textbox(label="LBP+KNN Stats", lines=3, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### HOG + SVM Output")
            svm_image_output = gr.Image(label="HOG+SVM Detection")
            svm_text_output = gr.Textbox(label="HOG+SVM Stats", lines=3, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### Mini-Xception (CNN) Output")
            cnn_image_output = gr.Image(label="CNN Detection")
            cnn_text_output = gr.Textbox(label="CNN Stats", lines=3, interactive=False)
            
    stream_inputs = [webcam_input, frame_counter_state, last_knn_text_state, last_svm_text_state, last_cnn_text_state, last_coords_state]
    stream_outputs = [frame_counter_state, last_knn_text_state, last_svm_text_state, last_cnn_text_state, last_coords_state,
                      knn_image_output, svm_image_output, cnn_image_output,
                      knn_text_output, svm_text_output, cnn_text_output]
    
    webcam_input.stream(fn=process_frame_for_stream, inputs=stream_inputs, outputs=stream_outputs)
    
    upload_outputs = [knn_image_output, knn_text_output, svm_image_output, svm_text_output, cnn_image_output, cnn_text_output]
    submit_btn.click(process_upload_image, inputs=upload_input, outputs=upload_outputs)

# --- Main Execution Block ---
<<<<<<< HEAD
if _name_ == "_main_":
=======
if __name__ == "__main__":
>>>>>>> 3bcc4c619ce520c185adce959214b356aee1dad3
    if None in [cnn_model, face_cascade, knn_model, svm_model]:
        print("\nCould not start the application due to one or more models failing to load.")
        print("Please ensure all model files exist and the training script has been run.")
    else:
        print("\nLaunching the Gradio Interface...")
        demo.launch(share=True)