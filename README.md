
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

## Training

```bash
python train_caption.py --cfg-path configs/coco_config.yaml
```
This command trains the Vision Transformer-based captioning model using the COCO dataset. Ensure dataset and config files are correctly placed.

## Inference

```bash
python inference_caption.py --image path/to/image.jpg
```
This runs inference on a single image and outputs the generated caption. You can modify the output directory and checkpoint path inside the script or via config if supported.

## Evaluation

```bash
python eval_caption.py --split val
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
python inference_caption.py --image_path path/to/image.jpg
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
