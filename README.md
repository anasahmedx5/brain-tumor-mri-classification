# Brain Tumor MRI Classification (FastAPI + UI)

Brain tumor classification system using PyTorch (ResNet-18) and FastAPI, with a web UI displaying prediction, confidence, class probabilities, and latency.

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run the Server

From the project root:

```bash
python -m uvicorn app.main:app --port 8000
```
*(or simply: `python app/main.py`)*

---

## 3. Open the UI

Open your browser at:

👉 **http://127.0.0.1:8000/**

- Drag & drop or browse an MRI scan (`.jpg`, `.png`).
- Or click any of the 4 pre-loaded test samples.
- View real-time **Predicted Class**, **Confidence**, **Probability Distribution**, and **Latency (ms)**.

---

## 4. Test Script (CLI)

To test sample images from the terminal:

```bash
python app/test_predict.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check (`status`, `framework`, `classes`) |
| `POST` | `/predict` | Multipart image upload → returns prediction & latency |
| `GET` | `/docs` | Interactive Swagger API docs |
