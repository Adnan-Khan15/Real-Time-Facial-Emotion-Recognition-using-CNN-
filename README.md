# Real-Time Facial Emotion Recognition (CNN)

**Live webcam + image upload app** that uses a custom-trained CNN to predict 7 emotions from face crops. Built with TensorFlow (macOS/M‑series ready) and Streamlit for an interactive frontend. 

Built by Adnan Khan, Kshant Carvalho, Sherlynn Vaz, Aaryan Singh for DEEP LEARNING AND REINFORCEMENT LEARNING Mini Project.

Model performance (final trained model)

Test accuracy (top-1): 55.06% (test loss: 1.1882) — measured on held-out test images. This is the number reported from eval_best.py (model file: models/emotion_best.keras).

Dataset: ~28,709 labeled images across 7 emotion classes (training set) with ~3,589 files in validation/test split used during evaluation.

---

## 🔥 TL;DR

* **Model:** custom CNN trained on a facial-emotions dataset
* **Test accuracy:** **~55.06%** (your model; replace if you retrain)
* **What you get:** `app.py` (Streamlit UI), `train.py`, `prepare_dataset.py`, `models/emotion_best.keras`

---

## ✨ Features

* Real-time webcam emotion detection with bounding box
* Smooth confidence chart (live) for all 7 classes
* Upload image to get single-shot predictions
* Confidence thresholding & smoothing controls in sidebar
* macOS (Apple Silicon) GPU support via `tensorflow-macos` + `tensorflow-metal`

---

## 📁 Project layout (important files)

```
emotion_project/
├─ app.py                # Streamlit app (UI + real-time video)
├─ train.py              # Training script for the CNN
├─ prepare_dataset.py    # Dataset preprocessing helper
├─ models/               # Saved Keras model(s)
│  └─ emotion_best.keras
├─ data/                 # (recommended) put the dataset files here
├─ dataset.zip           # alternative: dataset zip at project root
├─ requirements.txt
└─ README.md
```

---

## 📥 How to download the dataset from Kaggle (step-by-step)

> Use either the Kaggle website GUI or the Kaggle CLI. Below shows both methods.

### Option A — Download via Kaggle website (easy)

1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. Search for the dataset you used (e.g. `FER2013` or the dataset linked in your project instructions).
3. Click **Download** to save a `.zip` to your machine.
4. Move the downloaded `dataset.zip` into the project root (the same folder as `app.py`) or into `data/`.

**Recommended:** place the zip as `emotion_project/dataset.zip` or unzip into `emotion_project/data/` so your scripts can find it.

### Option B — Download using Kaggle CLI (recommended for reproducibility)

1. Install Kaggle CLI (if not already):

   ```bash
   pip install kaggle
   ```
2. Create an API token on Kaggle:

   * On kaggle.com -> Account -> Create API token. This downloads `kaggle.json`.
   * Move `kaggle.json` to `~/.kaggle/kaggle.json` and set permissions:

     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
3. Download dataset (replace `<owner/dataset-name>` with the dataset slug):

   ```bash
   kaggle datasets download -d <owner/dataset-name> -p ./ --unzip
   ```

   Example (if using FER2013 mirror):

   ```bash
   kaggle datasets download -d alx91/fer2013 -p ./data --unzip
   ```
4. After this step you should have raw image files / csv inside `data/`.

---

## 🗂️ Where to put the dataset in the project

* If you downloaded `dataset.zip` via browser: place it at project root or `data/` then unzip:

  ```bash
  # from project root
  unzip dataset.zip -d data/
  ```
* If you used Kaggle CLI with `--unzip -p ./data` then files already land in `data/`.
* `prepare_dataset.py` expects raw files in `data/` (or follow README comments in that script). If it expects `dataset/train` folder, create that structure:

  ```bash
  mkdir -p data/train data/val
  # move class subfolders into data/train
  ```

---

## 🛠 Setup (macOS / M‑series friendly)

1. Create and activate Python venv (use the included `requirements.txt`):

   ```bash
   cd /path/to/emotion_project
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

   ⚠️ **Important:** use the provided `requirements.txt` — it pins `numpy`, `tensorflow-macos`, `tensorflow-metal`, `h5py` etc. to versions that work well on macOS M1/M2.

2. (Optional) If you run into binary mismatch errors (e.g. `numpy.dtype size changed`), run:

   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

---

## ▶️ How to run the app (Streamlit)

1. Activate venv:

   ```bash
   source venv/bin/activate
   ```
2. Run Streamlit:

   ```bash
   streamlit run app.py
   ```
3. Open the Local URL shown in terminal (usually `http://localhost:8501`).
4. Sidebar controls allow webcam toggle, smoothing, and confidence threshold. You can also **Upload Image** to get a single-shot prediction.

---

## ▶️ How to train your own model

1. Ensure dataset is preprocessed and available under `data/` or `dataset/train` (see `prepare_dataset.py`).
2. Run training script:

   ```bash
   python prepare_dataset.py   # if your project needs preprocessing
   python train.py
   ```
3. Trained model will be saved in `models/` (e.g. `models/emotion_best.keras`). Replace the model file in repo if you want the app to use the new weights.

---

## 🧰 Troubleshooting & tips

* **Streamlit errors about `tensorflow` / `h5py` / `numpy`** — use the pinned `requirements.txt` and reinstall.
* **Video freezing or graph not updating** — ensure `streamlit-webrtc` is installed and you run `streamlit run app.py` from within the *same* virtual environment where packages are installed.
* **Model load errors on custom layers** — `load_model(..., compile=False)` is used in the app; if your saved model contains custom layers, include `custom_objects` (see `app.py` comments).
* **If GitHub upload is too large** — remove large dataset before pushing or use Git LFS.

---

## ✅ Quick commands summary

```bash
# setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# download dataset (example with kaggle CLI)
kaggle datasets download -d <owner/dataset-name> -p ./data --unzip

# prepare & train
python prepare_dataset.py
python train.py

# run app
streamlit run app.py
```

---

## 📦 What to commit to GitHub

* `app.py`, `train.py`, `prepare_dataset.py`, `requirements.txt`, `README.md`, and `models/emotion_best.keras` (if you want to share weights).
* **Avoid** committing `venv/` — keep it in `.gitignore`.
* Consider putting `dataset.zip` in a separate release or using Git LFS for large dataset files.

---

If you want, I can now:

* generate a clean `README.md` file in your repo and push it for you, or
* produce a short project description / GitHub Topics list, or
* create a `setup.md` with screenshots showing each step.

Which one would you like next?
