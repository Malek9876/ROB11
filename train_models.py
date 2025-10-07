import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from tqdm import tqdm

print("Script started...")

# --- Configuration ---
# IMPORTANT: Update this path to the root directory of your dataset
DATASET_PATH = 'fer2013' 
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')

# The emotion labels must be in the same order as the CNN model expects (0=angry, 1=disgust, etc.)
# os.listdir might not guarantee order, so we explicitly define it.
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# --- Check if the dataset path exists ---
if not os.path.isdir(TRAIN_DIR):
    print(f"ERROR: Training directory not found at '{TRAIN_DIR}'.")
    print("Please make sure your dataset is in the correct folder structure.")
    exit()

# --- Prepare for Feature Extraction ---
lbp_features_list = []
hog_features_list = []
labels = []

print("Starting feature extraction from image folders... This will take a while.")

# Iterate through each emotion folder
for emotion_label, emotion_name in enumerate(EMOTIONS):
    emotion_dir = os.path.join(TRAIN_DIR, emotion_name)
    
    if not os.path.isdir(emotion_dir):
        print(f"Warning: Directory for '{emotion_name}' not found. Skipping.")
        continue

    # Get a list of all images in the emotion directory
    image_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images for '{emotion_name}'...")
    
    for image_path in tqdm(image_files, desc=f'Emotion: {emotion_name}'):
        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue

        # --- LBP Feature Extraction ---
        lbp_image = cv2.resize(image, (128, 128))
        lbp = local_binary_pattern(lbp_image, P=24, R=8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features_list.append(hist)
        
        # --- HOG Feature Extraction ---
        hog_image = cv2.resize(image, (64, 128))
        hog_features = hog(hog_image, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
        hog_features_list.append(hog_features)

        # Store the label (0 for angry, 1 for disgust, etc.)
        labels.append(emotion_label)

print("\nFeature extraction complete.")

# Convert lists to numpy arrays for scikit-learn
X_train_lbp = np.array(lbp_features_list)
X_train_hog = np.array(hog_features_list)
y_train = np.array(labels)

print(f"Total features extracted: {len(y_train)}")
print(f"Shape of LBP features: {X_train_lbp.shape}")
print(f"Shape of HOG features: {X_train_hog.shape}")


# --- Train and Save LBP+KNN Model ---
print("\nTraining LBP+KNN model...")
# Using n_jobs=-1 will use all available CPU cores to speed up training
knn_model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1) 
knn_model.fit(X_train_lbp, y_train)
print("KNN training complete. Saving model to 'knn_model.joblib'...")
joblib.dump(knn_model, 'knn_model.joblib')

# --- Train and Save HOG+SVM Model ---
print("\nTraining HOG+SVM model...")
# A pipeline automatically scales the data before feeding it to the SVM
svm_model = make_pipeline(StandardScaler(), LinearSVC(random_state=42, dual='auto', max_iter=2000))
svm_model.fit(X_train_hog, y_train)
print("SVM training complete. Saving model to 'svm_model.joblib'...")
joblib.dump(svm_model, 'svm_model.joblib')

print("\nAll models have been trained and saved successfully!")