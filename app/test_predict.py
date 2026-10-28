import os
import requests

# API Endpoint URL
URL = "http://127.0.0.1:8000/predict"

# Path to your data folder
DATA_DIR = "data"

# Supported image extensions
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

def test_local_images():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    # Find all images in the data directory
    image_files = [
        f for f in os.listdir(DATA_DIR) 
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not image_files:
        print(f"No image files found in '{DATA_DIR}' directory.")
        return

    print(f"Found {len(image_files)} images. Testing predictions...\n")
    print("-" * 65)

    for img_name in image_files:
        img_path = os.path.join(DATA_DIR, img_name)
        
        with open(img_path, "rb") as f:
            # Send file via multipart/form-data matching the API spec
            files = {"file": (img_name, f, "image/jpeg")}
            response = requests.post(URL, files=files)

        if response.status_code == 200:
            res_data = response.json()
            label = res_data["label"]
            confidence = res_data["confidence"]
            probs = res_data["probabilities"]
            latency = res_data["latency_ms"]

            print(f"File: {img_name}")
            print(f"Predicted Class : {label}")
            print(f"Confidence      : {confidence * 100:.1f}% ({confidence})")
            print(f"Probabilities   : {probs}")
            print(f"Latency         : {latency} ms")
            print("-" * 65)
        else:
            print(f"Failed to process {img_name}. Status code: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_local_images()