
# 🧠 Vision_Transformer_Captioning – Image Captioning with Vision Transformers

**Vision_Transformer_Captioning** is a deep learning pipeline for generating descriptive captions from images using a Vision Transformer-based architecture.  
It combines object detection with transformer-based language modeling to generate human-like textual descriptions from visual data.

---

## 🏗️ Key Features

- 📸 Object detector training via `train_detector.py`
- 🧾 Image caption generation using `train_caption.py`
- 🧪 Evaluation support for both COCO and NoCaps datasets
- 🔁 Flexible scripts for offline/online inference
- 📊 Includes benchmarking and reproducibility support
- 🤖 GPT-based caption fluency scoring utility
- 🖼️ Streamlit app for interactive caption viewing

---

## 📂 Project Structure

```
Vision_Transformer_Captioning/
├── train_detector.py         # Train object detection model
├── train_caption.py          # Train transformer-based captioning model
├── eval_caption.py           # Evaluate captioning performance
├── inference_caption.py      # Run caption generation
├── eval_nocaps.py            # Eval on NoCaps benchmark
├── Makefile                  # Task automation
├── requirements.txt          # Python dependencies
```

---

## 🔧 Setup

### Requirements

- Python 3.8+
- PyTorch ≥ 1.7
- CUDA-compatible GPU
- COCO or NoCaps datasets

### Installation

```bash
git clone https://github.com/Adityarajsingh2904/Vision_Transformer_Captioning.git
cd Vision_Transformer_Captioning
pip install -r requirements.txt
```

### Retrieve configs and dataset utilities

The training scripts rely on configuration files under `configs/` and dataloader
code inside `datasets/`. These directories are omitted here for brevity. Clone
the original repository and copy them into this folder so that paths like
`configs/caption/coco_config.yaml` resolve correctly:

```bash
git clone https://github.com/Adityarajsingh2904/Vision_Transformer_Captioning.git full_repo
cp -r full_repo/configs ./configs
cp -r full_repo/datasets ./datasets
```

## Training

```bash
python train_caption.py
```
This command trains the Vision Transformer-based captioning model using the COCO dataset. Hydra will automatically load `configs/caption/coco_config.yaml` once the `configs/` folder has been copied.

## Inference

```bash
python inference_caption.py img_path=path/to/image.jpg
```
This runs inference on a single image and outputs the generated caption. You can modify the output directory and checkpoint path inside the config or via Hydra command-line overrides.

## Evaluation

```bash
python eval_caption.py split=val
```
Evaluates the model using standard metrics (BLEU, CIDEr, etc.) on the validation split.

Ensure that the config paths and dataset setup in configs/ match your local environment before executing these commands.

---

## 🚀 Usage

### Train Detector

```bash
python train_detector.py
```

### Train Captioning Model

```bash
python train_caption.py
```

### Inference

```bash
python inference_caption.py img_path=path/to/image.jpg
```

### Evaluate

```bash
python eval_caption.py
```

### GPT Caption Fluency Evaluation

```bash
python gpt_caption_evaluator.py captions.txt --api_key YOUR_OPENAI_KEY
```

### Streamlit Caption Viewer

```bash
streamlit run caption_viewer.py
```

---

## 👤 Maintainer

**Aditya Raj Singh**  
📧 thisis.adityarajsingh@gmail.com  
🔗 [GitHub](https://github.com/Adityarajsingh2904)

---

## 📜 License

This project is released under the **MIT License**.
