
# NeuroScan AI — Brain Tumor MRI Classification

![UI Example](<sample/Screenshot 2026-10-28 223332.png>)

A deep-learning project for classifying brain MRI images into four categories. This project explores transfer learning, fine-tuning, and model comparison using **PyTorch and TensorFlow**, then integrates a trained PyTorch ResNet-18 model into a FastAPI application with a custom interactive web interface.

> **Disclaimer:** This is an educational machine-learning project, not a medical device. Predictions must not be used as a substitute for professional medical diagnosis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Web Interface](#web-interface)
- [Sample MRI Image](#sample-mri-image)
- [Dataset](#dataset)
- [Model Experiments](#model-experiments)
- [PyTorch vs. TensorFlow](#pytorch-vs-tensorflow)
- [Fine-Tuning and Overfitting](#fine-tuning-and-overfitting)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Testing Predictions](#testing-predictions)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

## Overview

The goal of this project is to investigate how transfer learning and fine-tuning affect brain MRI image classification, compare model implementations across two deep-learning frameworks, and deploy the selected model through a web application.

The project combines model experimentation with practical application development, including image preprocessing, inference, API development, and an interactive frontend.

### Main Objectives

- Classify brain MRI images into four categories.
- Experiment with pretrained ResNet architectures.
- Compare frozen-backbone training with fine-tuning.
- Compare PyTorch and TensorFlow/Keras implementations.
- Monitor validation performance to select promising checkpoints.
- Build a FastAPI backend for model inference.
- Develop an interactive web UI for uploading MRI scans and viewing predictions.
- Return class probabilities, confidence scores, and inference latency through the API.

## Features

- Four-class brain MRI image classification.
- Transfer learning using ResNet-18 and ResNet-50.
- Frozen-backbone and fine-tuning experiments.
- Comparison of PyTorch and TensorFlow validation results.
- Custom NeuroScan AI web interface.
- Drag-and-drop MRI image upload.
- Quick testing using sample images.
- Predicted class and confidence score.
- Probability distribution for all four classes.
- Inference latency measurement.
- FastAPI REST API.
- Interactive Swagger API documentation.
- Command-line prediction testing.
- Model loading and preprocessing at application startup.

## Web Interface

The project includes a custom web dashboard named **NeuroScan AI — Brain Tumor MRI Classification Platform**.

The interface provides an accessible way to test the trained model without manually sending HTTP requests.

### UI Features

**1. MRI Image Upload**

- Drag and drop an MRI image into the upload area.
- Browse local files to select an image.
- Supports JPEG and PNG images.
- Displays a preview of the selected scan.
- Shows the filename, image dimensions, and file size.

**2. Automatic Analysis**

The interface includes an Auto Analyze option that can automatically run inference when an image is loaded.

Users can also manually trigger inference using the Run Inference button.

**3. Quick-Test Samples**

The interface provides sample MRI images from the project's `data/` directory. Users can select a sample to test the model without uploading another image.

**4. Prediction Results**

After inference, the dashboard displays:

- Predicted class.
- Model confidence score.
- Probability distribution for all four classes.
- Inference latency in milliseconds.
- Model framework and input resolution.

**5. Probability Visualization**

The interface displays the probability assigned to each class using percentage labels and visual progress bars.

This makes it easier to compare the model's relative scores across the four possible classes.

**6. Performance Information**

The dashboard displays inference latency and the input resolution used by the model.

The current application uses PyTorch ResNet-18 with an input resolution of `224 × 224`.

**7. Raw API Response**

An expandable section displays the raw FastAPI JSON response, with an option to copy it for inspection or debugging.

**8. API Status and Error Handling**

The interface includes API status feedback, loading animations, and error messages for failed inference requests.

### UI Workflow

1. Start the FastAPI server.
2. Open the application in a browser.
3. Upload an MRI scan or select a sample image.
4. Run inference.
5. Review the predicted class, confidence, class probabilities, and latency.

The deployed application currently uses the PyTorch ResNet-18 model. TensorFlow/Keras was used for separate training experiments and framework comparison.

## Sample MRI Image

The following example uses an image from the repository's own `data/` directory.

![Brain MRI sample — Glioma](data/Te-gl_1.jpg)

*Example sample: `Te-gl_1.jpg`. The UI associates this sample with the glioma class. This label does not guarantee that the model will predict it correctly.*

If the image does not appear on GitHub, confirm that `data/Te-gl_1.jpg` exists in the repository and that the filename capitalization matches exactly.

## Dataset

The project experiments used the **Brain Tumor MRI Dataset by Masoud Nickparvar**.

### Dataset Summary

| Property | Value |
|---|---|
| Total images | 7,200 |
| Training images | 4,480 |
| Validation images | 1,120 |
| Test images | 1,600 |
| Number of classes | 4 |
| Image type | Brain MRI |
| Task | Multiclass image classification |

The `data/` directory in the application contains sample images used for quick testing. It is not necessarily the complete dataset used during training.

### Classification Classes

| Class | Description |
|---|---|
| `glioma` | Images categorized as glioma in the dataset |
| `meningioma` | Images categorized as meningioma in the dataset |
| `notumor` | Images categorized as showing no tumor in the dataset |
| `pituitary` | Images categorized as pituitary tumor in the dataset |

These categories represent dataset labels, not definitive clinical diagnoses.

### Image Preprocessing

The deployed PyTorch inference pipeline applies the following transformations:

1. Open the uploaded image.
2. Convert it to RGB.
3. Resize it to `224 × 224` pixels.
4. Convert it into a PyTorch tensor.
5. Normalize the image using ImageNet channel statistics.

Normalization parameters:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Consistent preprocessing is important because the input representation must match the requirements of the trained model.

## Model Experiments

A major part of this project was experimenting with different transfer-learning strategies rather than relying on a single training configuration.

I compared two approaches:

- **Frozen backbone:** Keep the pretrained feature extractor frozen and train the classification head.
- **Fine-tuning:** Unfreeze selected deeper layers of the pretrained network and train them alongside the classification head.

The experiments were conducted using both PyTorch and TensorFlow/Keras.

### 1. PyTorch — ResNet-18

ResNet-18 was used as the lighter architecture and is the model currently integrated into the FastAPI application.

#### Frozen Backbone

The pretrained convolutional backbone was frozen while the classification head was trained for the four target classes.

Best recorded validation accuracy:

**84.46%**

#### Fine-Tuned ResNet-18

For the fine-tuning experiment, deeper feature-extraction layers, including `layer4`, were unfrozen so that the pretrained representations could adapt to the MRI classification task.

Best recorded validation accuracy:

**96.52%**

Fine-tuning improved validation accuracy by **12.06 percentage points** compared with the frozen-backbone experiment.

The best checkpoint from this experiment was previously saved as:

```text
resnet18_strategy_b_best.pt
```

### 2. TensorFlow/Keras — ResNet-50

ResNet-50 was evaluated separately using TensorFlow/Keras to compare another architecture and framework.

#### Frozen Backbone

The pretrained backbone was frozen while the classification head was trained.

Best recorded validation accuracy:

**91.61%**

#### Fine-Tuned ResNet-50

For fine-tuning, deeper layers, including `conv5_block3`, were unfrozen and trained to adapt the network to the target dataset.

Best recorded validation accuracy:

**93.66%**

Fine-tuning improved validation accuracy by **2.05 percentage points** compared with the frozen-backbone experiment.

The best checkpoint from this experiment was previously saved as:

```text
tensorflow_resnet50_strategy_b_best.keras
```

## PyTorch vs. TensorFlow

One of the main goals of this project was to compare the results of transfer learning and fine-tuning across PyTorch and TensorFlow/Keras.

### Validation Results

| Framework | Architecture | Training strategy | Best validation accuracy |
|---|---|---|---:|
| PyTorch | ResNet-18 | Frozen backbone | 84.46% |
| PyTorch | ResNet-18 | Fine-tuning | **96.52%** |
| TensorFlow/Keras | ResNet-50 | Frozen backbone | 91.61% |
| TensorFlow/Keras | ResNet-50 | Fine-tuning | **93.66%** |

### Observations

- Fine-tuning improved validation accuracy in both experiments.
- PyTorch ResNet-18 improved from 84.46% to 96.52%.
- TensorFlow ResNet-50 improved from 91.61% to 93.66%.
- The highest recorded validation accuracy was achieved by the fine-tuned PyTorch ResNet-18.
- The selected PyTorch model was integrated into the FastAPI inference application.

These results compare the recorded experiments. They do not establish that one framework is universally better than the other because the architectures, training configurations, and implementations differ.

**Important:** These are validation results, not final held-out test results. A fair final evaluation should use the same untouched test set and report additional metrics.

## Fine-Tuning and Overfitting

Fine-tuning allows deeper pretrained layers to adapt to the target dataset. It can improve performance when the original pretrained features do not fully capture the characteristics of the new domain.

However, training more parameters also increases the risk of overfitting.

For this project, comparing frozen-backbone and fine-tuned models helped demonstrate how training strategy affects validation performance.

Recommended evaluation practices include:

- Monitor training and validation loss across epochs.
- Compare training and validation metrics to identify overfitting.
- Select checkpoints using validation metrics.
- Keep the test set separate from model selection.
- Evaluate each class independently.
- Report precision, recall, macro-F1, and a confusion matrix.
- Compare training histories across the different experiments.

The highest validation accuracy should not be treated as a substitute for a final independent test evaluation.

## System Architecture

The application consists of a browser-based frontend, a FastAPI backend, and a PyTorch inference pipeline.

### Frontend

The frontend is built using:

- HTML
- CSS
- JavaScript

It handles image selection, image previews, API requests, loading states, and result visualization.

### Backend

The backend uses FastAPI to:

- Serve the web interface.
- Load the trained model at application startup.
- Validate uploaded image content.
- Preprocess the image.
- Run inference.
- Calculate class probabilities and confidence.
- Measure inference latency.
- Return a JSON response.

### Inference Pipeline

The deployed inference pipeline follows these steps:

1. Receive an image through the `/predict` endpoint.
2. Validate that the uploaded file is an image.
3. Open the image and convert it to RGB.
4. Resize and normalize the image.
5. Add the batch dimension expected by ResNet-18.
6. Run inference without gradient calculation.
7. Apply softmax to the model outputs.
8. Select the class with the highest probability.
9. Return the predicted class, confidence, class probabilities, and latency.

## Installation

### Prerequisites

- Python 3.10 or newer recommended.
- A compatible PyTorch and torchvision installation.
- The trained PyTorch checkpoint expected by the application.

### 1. Clone the Repository

```bash
git clone https://github.com/anasahmedx5/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The application dependencies include:

- FastAPI
- Uvicorn
- PyTorch
- torchvision
- Pillow
- python-multipart
- Requests

TensorFlow is relevant to the separate training experiments and may require a separate environment if it is not included in the runtime requirements.

### 4. Verify the Model Checkpoint

The deployed application is configured to load:

```text
model/model_T24.pt
```

This path is defined in `app/config.py`.

Make sure the checkpoint exists at the configured location and matches the expected ResNet-18 architecture and class order.

If the checkpoint is stored elsewhere, update `MODEL_PATH` in `app/config.py` accordingly.

## Running the Application

From the project root, run:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Alternatively, the application supports:

```bash
python app/main.py
```

Once the server starts, open the following URLs.

| Resource | URL |
|---|---|
| NeuroScan AI UI | http://127.0.0.1:8000/ |
| Swagger API documentation | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

## Testing Predictions

The repository includes a command-line script for testing images in the `data/` directory.

Keep the FastAPI server running, open another terminal at the project root, and run:

```bash
python app/test_predict.py
```

The script sends each supported image to the `/predict` endpoint and prints:

- Image filename.
- Predicted class.
- Confidence score.
- Class probability distribution.
- Inference latency.

This provides a convenient way to test the deployed inference API independently of the browser interface.

## API Documentation

The application exposes the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the NeuroScan AI web interface |
| `GET` | `/health` | Returns API status, framework, and classes |
| `GET` | `/api/samples` | Lists available sample images |
| `POST` | `/predict` | Accepts an image and returns prediction results |
| `GET` | `/docs` | Opens interactive Swagger documentation |

### Example Prediction Response

A successful request to `POST /predict` returns a JSON object similar to the following.

The values are illustrative examples, not measured predictions.

```json
{
  "framework": "pytorch",
  "label": "glioma",
  "confidence": 0.96,
  "probabilities": {
    "glioma": 0.96,
    "meningioma": 0.02,
    "notumor": 0.01,
    "pituitary": 0.01
  },
  "latency_ms": 42
}
```

The actual values depend on the uploaded image, model checkpoint, and execution environment.

## Project Structure

The following is a simplified overview of the main application files.

```text
brain-tumor-mri-classification/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── test_predict.py
│   │
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── data/
│   └── Te-gl_1.jpg
│
├── model/
│   └── model_T24.pt
│
├── requirements.txt
└── README.md
```

### Main Files

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI server, model loading, preprocessing, and inference |
| `app/config.py` | Framework configuration, model path, class names, and input size |
| `app/test_predict.py` | Command-line testing of local sample images |
| `app/static/index.html` | NeuroScan AI frontend structure |
| `app/static/styles.css` | Web interface styling |
| `app/static/app.js` | Frontend interactions and API requests |
| `data/` | Sample images used for testing |
| `model/` | Configured model checkpoint |
| `requirements.txt` | Python runtime dependencies |

Training notebooks and experiment checkpoints may be stored separately from the runtime application files.

## Technologies

### Machine Learning and Deep Learning

- Python
- PyTorch
- torchvision
- TensorFlow/Keras
- ResNet-18
- ResNet-50
- Transfer learning
- Fine-tuning
- Multiclass image classification

### Backend and Inference

- FastAPI
- Uvicorn
- Pillow
- REST API development
- Softmax probability calculation

### Frontend

- HTML
- CSS
- JavaScript
- Interactive image upload
- Dynamic prediction visualization

## Future Improvements

Potential improvements include:

- Evaluate the final selected model on the held-out test set.
- Add a confusion matrix and per-class evaluation metrics.
- Report macro-F1, precision, and recall.
- Visualize training and validation loss across epochs.
- Compare the models under a consistent experimental protocol.
- Add automated tests for API responses and invalid uploads.
- Improve file-size validation and server-side upload restrictions.
- Document the exact training configuration for each checkpoint.
- Improve reproducibility with pinned dependency versions.
- Evaluate generalization using an independent dataset.
- Investigate model calibration and explainability techniques.

## Disclaimer

This project is intended for educational and experimental purposes in computer vision and deep learning.

The model has not been established as clinically safe or validated for medical diagnosis. Its predictions may be incorrect and should not be used to make medical decisions. Always consult qualified medical professionals for diagnosis and treatment.
