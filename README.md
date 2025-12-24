
***

# SMS Spam Classification with TensorFlow

This project builds and evaluates a neural-network-based classifier to detect spam SMS messages using TensorFlow and Keras.[1]

## Project overview

- Binary classification of SMS messages into **ham** (not spam) and **spam**.[1]
- Uses a balanced subset of the “SPAM text message 20170820 – Data.csv” dataset.[1]
- Implements a simple but effective neural network with word embeddings and a GlobalAveragePooling layer.[1]

## Dataset

- Input file: `SPAM text message 20170820 - Data.csv`.[1]
- Important columns:
  - `Category`: label (`ham` or `spam`).[1]
  - `Message`: raw SMS text.[1]
- The script:
  - Splits the data into ham and spam subsets.[1]
  - Downsamples ham to match the number of spam messages (class balancing).[1]
  - Maps labels to integers: ham → 0, spam → 1.[1]

## Preprocessing pipeline

- Train/test split (80% train, 20% test, `random_state=434`).[1]
- Text tokenization with Keras `Tokenizer`:
  - `num_words = 500` (top 500 most frequent words).[1]
  - `oov_token = "<OOV>"` for out-of-vocabulary words.[1]
- Sequence preparation:
  - Convert texts to integer sequences.[1]
  - Pad/truncate sequences to `maxlen = 50`, padding/truncating **post**.[1]

## Model architecture

Implemented using `tf.keras.models.Sequential`:[1]

- `Embedding(vocab_size=500, output_dim=16, input_length=50)`  
- `GlobalAveragePooling1D()`  
- `Dense(32, activation="relu")`  
- `Dropout(0.3)`  
- `Dense(1, activation="sigmoid")`  

The model is compiled with:[1]

- Loss: `BinaryCrossentropy(from_logits=True)`  
- Optimizer: `adam`  
- Metrics: `accuracy`  

> Note: The final layer uses a sigmoid activation, so `from_logits=False` would be the more conventional setting for binary cross-entropy.[1]

## Training and evaluation

- Training:
  - Epochs: up to 30.[1]
  - Callback: `EarlyStopping(monitor="val_loss", patience=3)`.[1]
  - Validation data: held-out test set (`Testing_pad`, `test_labels`).[1]
- Typical results (from the included run):  
  - Training accuracy increases to ~0.97–0.98.[1]
  - Validation accuracy peaks around ~0.97–0.99.[1]
- After training, the script:
  - Evaluates on the test set with `model.evaluate(Testing_pad, test_labels)`.[1]
  - Plots training vs. validation accuracy over epochs.[1]

## Inference example

The notebook defines a helper function to classify new messages:[1]

```python
predict_msg = [
    "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    "Ok lar... Joking wif u oni...",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
]

def predict_spam(predict_msg):
    new_seq = token.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen=50,
                           padding=padding_type,
                           truncating='post')
    return model.predict(padded)
```

For the three example messages, the model outputs spam probabilities such as:[1]

```python
array([[3.8677291e-04],
       [4.2803233e-04],
       [9.9962217e-01]], dtype=float32)
```

Low values (~0.0004) correspond to ham; high values (~0.9996) correspond to spam.[1]

## Requirements

Main Python dependencies:[1]

- Python 3.12 (or compatible 3.x)  
- TensorFlow  
- NumPy  
- pandas  
- scikit-learn (`train_test_split`)  
- matplotlib  

You can install them with:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## How to run

1. Place the dataset CSV at the path used in the notebook or update the `pd.read_csv(...)` path accordingly.[1]
2. Run the notebook cells in order:
   - Data loading and preprocessing.[1]
   - Model definition and compilation.[1]
   - Training (`model.fit(...)`) and evaluation.[1]
3. Use `predict_spam(...)` to test custom SMS messages.[1]

---

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153508662/fb468988-e9ac-4a4f-9341-b4f9cb778d20/paste.txt)
