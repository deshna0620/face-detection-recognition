# Face Detection and Recognition App

A project that implements Face Detection and Recognition using YOLOv8 (CNN model) and FaceNet for embedding-based face comparison, as per internship assignment requirements.

## Features

- Face Detection using YOLOv8 Nano (yolov8n.pt)
- Face Recognition using FaceNet Embeddings
- Mask and Cap augmentation applied to original face images
- Cosine Similarity calculated to verify face matching accuracy
- Streamlit Web App to upload and test faces
- Terminal-based testing scripts provided
- Accuracy requirement: Above 98% similarity threshold

## Project Structure

face-detection-recognition/
│
├── data/
│   ├── face1.png
│   ├── face2.png
│   ├── face1_masked.png
│   ├── face2_capped.png
│
├── app.py                  # Streamlit Web App
├── detect.py               # YOLOv8 Detection Script
├── face_recognition.py     # FaceNet Recognition Script
├── utils.py                # Mask and Cap Augmentation Script
├── requirements.txt        # Required Python packages
├── yolov8n.pt              # YOLOv8 Nano Model
└── README.md               # Project documentation

## Setup Instructions

### 1. Clone the Repository

```

git clone [https://github.com/deshna0620/face-detection-recognition.git](https://github.com/deshna0620/face-detection-recognition.git)
cd face-detection-recognition

```

### 2. Create Virtual Environment (Optional but Recommended)

```

python -m venv venv
venv\Scripts\activate   # On Windows

```

### 3. Install Required Packages

```

pip install -r requirements.txt

```

### 4. Ensure YOLOv8 Nano Model is Present

```

yolov8n.pt   # Already included or download from Ultralytics official releases

```

## Usage Instructions

### Step 1: Generate Masked and Capped Face Images

```

python utils.py

```

### Step 2: Detect Faces Using YOLOv8

```

python detect.py

```

### Step 3: Compare Faces Using FaceNet

```

python face\_recognition.py

```

### Step 4: Run Streamlit Web Application

```

streamlit run app.py

```

Visit the URL displayed in the terminal, usually http://localhost:8501

## Notes

- Upload two face images via the Streamlit app to test detection and recognition.
- YOLOv8 results will be saved in the runs/detect directory.
- FaceNet computes embeddings and similarity.
- A similarity score above 98% indicates a face match as required.

## Example Output

```

Similarity between data/face1\_masked.png and data/face2\_capped.png: 99.12%
Faces Match (above 98% confidence)

```
