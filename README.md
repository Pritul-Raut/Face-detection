# 👁️‍🗨️ Face Recognition System – Full Report

---

## 🧠 Main Algorithm – Step by Step

This system consists of two Python scripts working together:

1. `encode.py` – prepares the dataset (known faces).
2. `face_recognis.py` – performs real-time face recognition using webcam.

### 🔄 High-Level Workflow

1. **Face Data Preparation (encode.py)**:
   - Read face images from a folder.
   - Detect faces and encode them into 128-d vectors.
   - Save the encodings and associated names using `pickle`.

2. **Real-Time Face Recognition (face_recognis.py)**:
   - Load encoded face data using `pickle`.
   - Open webcam and read frames.
   - Detect and encode faces from each frame.
   - Compare them with known encodings.
   - If match found, display name; else display “Unidentified”.

---

## 📦 `face_recognition` Module – Internal Working

### 🔍 1. Face Detection

- Uses HOG (Histogram of Oriented Gradients) or CNN for detecting face location.
- Function: `face_recognition.face_locations(image)`

### 🧬 2. Face Encoding

- Converts face to a unique 128-d vector using a deep neural network.
- Function: `face_recognition.face_encodings(image, face_locations)`

### 🧪 3. Face Comparison

- Uses Euclidean distance to compare encodings.
- Functions:
  - `face_recognition.compare_faces(known_encodings, test_encoding)`
  - `face_recognition.face_distance(known_encodings, test_encoding)`

### ✅ Matching Logic

- If distance < 0.6 → Match
- `compare_faces` returns `True` or `False` for each known encoding.

---

## 📁 Role of `encode.py` – Encoding Phase

### 📌 Purpose

To prepare known faces and store their encodings for later recognition.

### 🔄 How It Works

1. Read all images from a folder (`Images/`).
2. Detect faces in each image.
3. Encode each face into a 128-d vector.
4. Save the encodings and their associated names using `pickle`.

### 📝 Internal Code Structure

```python
import cv2, os, pickle
import face_recognition

path = 'Images'
images, names = [], []

for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))
    name = os.path.splitext(file)[0]
    images.append(img)
    names.append(name)

encodelist = [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]

with open('encodefile.p', 'wb') as f:
    pickle.dump((encodelist, names), f)
```

### 🧾 Output

- File: `encodefile.p`
- Contains:
  - List of 128-d encodings
  - Corresponding names

---

## 🎥 Role of `face_recognis.py` – Recognition Phase

### 📌 Purpose

To recognize faces from a webcam feed using the saved encodings.

### 🔄 How It Works

1. Load `encodefile.p` using `pickle`.
2. Start webcam stream.
3. Flip and process each frame.
4. Detect faces in the frame.
5. Encode the detected faces.
6. Compare with known encodings.
7. Display name or “Unidentified”.

### 📝 Key Code Logic

```python
file = open('encodefile.p', 'rb')
encodelistknown, names = pickle.load(file)
file.close()

faceloc = face_recognition.face_locations(img)
facencode = face_recognition.face_encodings(img, faceloc)

for encodeface, Faceloction in zip(facencode, faceloc):
    match = face_recognition.compare_faces(encodelistknown, encodeface)
    facedis = face_recognition.face_distance(encodelistknown, encodeface)
    matchindex = np.argmin(facedis)
    if match[matchindex]:
        name = names[matchindex]
```

---

## 🔗 Relationship Between `encode.py` and `face_recognis.py`

| `encode.py`               | `face_recognis.py`           |
|---------------------------|-------------------------------|
| Detects & encodes known faces | Loads those encodings            |
| Saves to `encodefile.p`        | Uses `encodefile.p` to match     |
| One-time or occasional use     | Runs in a loop for live detection |

---

## 🧪 Important Library Functions

| Function | Purpose |
|----------|---------|
| `face_recognition.face_locations()` | Detects faces in an image |
| `face_recognition.face_encodings()` | Encodes faces into vectors |
| `face_recognition.compare_faces()` | Compares known vs unknown encodings |
| `face_recognition.face_distance()` | Measures similarity (lower = better match) |
| `cv2.VideoCapture()` | Opens webcam |
| `cvzone.cornerRect()` | Draws styled rectangle |
| `cv2.putText()` | Displays name on screen |

---

## ✅ Summary

This face recognition system consists of:

- **`encode.py`** – Encodes known faces and saves them.
- **`face_recognis.py`** – Detects and recognizes faces in real-time.

It uses deep learning via the `face_recognition` library built on dlib to compare 128-dimensional facial features.

Great for: security systems, attendance apps, automation, etc.

---
