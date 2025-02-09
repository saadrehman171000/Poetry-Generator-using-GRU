
# Generative_Nazam

This project focuses on generating **Nazam** (a form of Urdu poetry) using a Deep Learning model. The model, built using **GRU** (Gated Recurrent Unit), is trained on a dataset of Roman Urdu poetry to generate the next part of a **Nazam** based on the user's input.

## Features
- **Generative Model**: Uses a GRU model to generate the next part of a **Nazam** based on a user's input.
- **Roman Urdu Text**: The model is trained specifically on Roman Urdu poetry, enabling it to generate authentic poetic text.

## Project Structure
1. **main.ipynb**: Jupyter notebook for training the GRU model on the dataset and saving the trained model and tokenizer.
2. **roman urdu poetry.csv**: Dataset of Roman Urdu poetry used for training the model.
3. **poetry_lstm_model.h5**: The trained model, ready for use.
4. **training_history.pkl**: Stores the training history (loss, accuracy) of the model.
5. **word_encoder.pkl**: The tokenizer used to convert text into tokenized sequences for model input.

## Requirements

Install the necessary Python libraries using pip:

```bash
pip install -r requirements.txt
```

This will install:
- `tensorflow` for the deep learning model.
- `numpy`, `pandas`, and `scikit-learn` for data handling and processing.

## Setup

1. **Prepare Dataset**: Ensure you have the dataset file `roman urdu poetry.csv`, which contains two columns: `Poet` and `Poetry` (Roman Urdu text). The `Poetry` column should contain the text of the Nazams.
   
2. **Training the Model**: 
   - Open and run the `main.ipynb` notebook to preprocess the data, train the GRU model, and save the model (`poetry_lstm_model.h5`) and tokenizer (`word_encoder.pkl`).
   - The model and tokenizer are saved for future use.

3. **Using the Model**: The trained model can be loaded for further inference or deployment as needed.
