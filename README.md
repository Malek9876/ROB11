# Real-time Facial Expression Detection

A Python application that uses a web interface to detect facial expressions in real-time from a webcam or an uploaded image. It compares the performance of three different machine learning models: LBP+KNN, HOG+SVM, and a Mini-Xception CNN.

## Features

-   Live analysis from a webcam.
-   Analysis of uploaded static images.
-   Side-by-side comparison of three different ML models.
-   Displays latency and prediction for each model.

## Model Performance

The models were evaluated on the FER-2013 test set. The results are as follows:

| Model                   | Accuracy (%) | Mean Latency (ms) |
| ----------------------- | :----------: | :---------------: |
| LBP + KNN               |    31.83%    |       8.67        |
| HOG + SVM               |    51.53%    |       3.80        |
| **Mini-Xception (CNN)** |  **66.17%**  |      62.80        |

The Mini-Xception (CNN) model provides the highest accuracy, while the classic HOG+SVM model is the fastest.

## Setup and Usage

Follow these steps to get the project running.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### Step 2: Install Dependencies

It is recommended to use a Python virtual environment. Install all required libraries by running:

```bash
pip install gradio opencv-python scikit-learn scikit-image numpy tensorflow joblib tqdm
```

### Step 3: Download Required Files

You need to download two things and place them in the project's root folder:

1.  **The FER-2013 Dataset:**
    -   Download the dataset from a source like [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
    -   Extract it so that you have a folder named `fer2013` in your project directory. This folder should contain `train` and `test` subdirectories.

2.  **The Pre-trained CNN Model:**
    -   Download the model weights file: `fer2013_mini_XCEPTION.102-0.66.hdf5`.
    -   You can find it [here](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5).
    -   Place this `.hdf5` file in the root of the project directory.

### Step 4: Train the Models (One-time only)

Before running the main application, you must train the KNN and SVM models on the dataset.

**Note:** This process is computationally intensive and may take 20-60 minutes.

```bash
python train_models.py
```

This will create two new files: `knn_model.joblib` and `svm_model.joblib`.

### Step 5: Run the Application

Once the models are trained, you can launch the web interface:

```bash
python emotion_detector.py```

Open the local URL provided in the terminal (e.g., `http://127.0.0.1:7860`) in your web browser to use the application.