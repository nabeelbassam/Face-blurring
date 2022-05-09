# Face Bluring project

The main objective of this project is to blur faces in a given picture if found, In order to blur faces shown in images, you need to first detect these faces and their position in the image then blur the detected faces.

## 🔩 requirement
Opencv
## 🔧 Usage
1. Clone or download this repo
2. Open project folder in CMD
3. Install required packages
```bash
   pip install -r requirements.txt
```
To run the face bluring code Run:
```bash
   python code/blur_face.py --face_cascade [face cascade path] --image [image path]
```
## 📍 Note
* I am using face detection model in [models](./models) folder that can detects faces but codes are valid for any cascade model.
* If you run the code without any parameters, it will load the cascade model from models and load an image from the data samples
