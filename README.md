# Deepfake Detector

EfficientNet-B4 based deepfake detection model trained on 140k Real vs Fake Faces dataset.

## Setup

### 1. Clone the repo
git clone <your-repo-url>
cd deepfake_detector

### 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download dataset
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/

### 5. Train
python train.py

### 6. Test
python test.py

### 7. Predict single image
python predict.py path/to/image.jpg

## Results
- Val Accuracy : 90%
- AUC-ROC      : 0.89
 
 
 
 
 
 
 
 
 
 
 
 
 
