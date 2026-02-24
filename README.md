# Currency-Classifier

A deep learning model that classifies Indian currency denominations (₹10, ₹20, ₹50, ₹100, ₹200, ₹500) using MobileNetV2 transfer learning with TensorFlow/Keras.

## Features

- **Two Interfaces**: Desktop GUI (Tkinter) and Web App (Streamlit)
- **Pre-trained Models**: Ready-to-use Keras and TensorFlow Lite formats
- **Lightweight**: TFLite model optimized for cloud deployment and mobile
- **Easy to Use**: Run either application without training
- **Accurate Classification**: Classifies Indian currency (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)

## Project Structure

```
├── currency_classifier_gui.py     
├── streamlit_app.py               
├── best_currency_classifier.h5    
├── best_currency_classifier.tflite
├── requirements.txt               
├── .streamlit/                      
└── .gitignore
```

## Quick Start

### Installation
```bash
git clone https://github.com/Daksh1685/Currency-Classifier.git
cd Currency-Classifier
pip install -r requirements.txt
```

### Usage

**Desktop GUI (Tkinter)**
```bash
python currency_classifier_gui.py
```

**Web Interface (Streamlit)**
```bash
streamlit run streamlit_app.py
```

## Model Specs

- **Architecture**: MobileNetV2 with custom dense layers
- **Input**: 224×224×3 RGB images
- **Output**: 6 currency denominations
- **Framework**: TensorFlow/Keras
- **Formats**: `.h5` (Keras) and `.tflite` (optimized)

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit, NumPy, Pillow

See `requirements.txt` for all dependencies.

## License

MIT License

---

**Daksh1685** | [GitHub](https://github.com/Daksh1685/Currency-Classifier)
