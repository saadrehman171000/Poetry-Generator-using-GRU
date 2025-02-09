# 🎭 Urdu Poetry Generator (Generative Nazam)

An AI-powered system that generates Urdu Nazams using GRU (Gated Recurrent Unit) neural network, creating authentic Roman Urdu poetry from user prompts.

## ✨ Features

- 🤖 **GRU Model**: Advanced neural network for natural language generation
- 📝 **Roman Urdu**: Trained on authentic Urdu poetry dataset
- 🎨 **Interactive UI**: Streamlit interface with real-time generation
- 📊 **Analysis**: Poetry metrics and generation history
- 🎯 **Customizable**: Adjustable line length and poem size

## 🛠️ Project Structure

- `main.ipynb`: Training notebook
- `app.py`: Streamlit web application
- `roman urdu poetry.csv`: Training dataset
- `poetry_gru_model.h5`: Trained model
- `word_encoder.pkl`: Word tokenizer

## 🚀 Quick Start

1. **Clone & Install**:
```bash
git clone https://github.com/saadrehman171000/Poetry-Generator-using-GRU.git
cd Poetry-Generator-using-GRU
pip install -r requirements.txt
```

2. **Run Application**:
```bash
streamlit run app.py
```

3. **Generate Poetry**:
- Enter starting words (e.g., "dil ke armaan")
- Adjust words per line (3-15) and total lines (2-10)
- Click "Generate Poetry"

## ⚠️ Troubleshooting

- **Model Loading**: Verify `poetry_gru_model.h5` and `word_encoder.pkl` exist
- **Performance**: First generation may be slower due to model loading
- **Generation**: Use common Roman Urdu words for better results

---
<p align="center">Made with ❤️ for Urdu Poetry | 
<a href="https://github.com/saadrehman171000/Poetry-Generator-using-GRU" target="_blank">GitHub</a></p>
