# CNN Sentiment Analysis with Streamlit

An educational NLP project that classifies movie-review text as **Positive** or **Negative** using a Keras 1D convolutional neural network (CNN), with a Streamlit web interface.

## Features

- Text cleanup and label validation
- Stratified train/validation/test split
- Training-only vocabulary adaptation with Keras `TextVectorization` (unigrams and bigrams)
- Embedding → 1D convolution → global max pooling → dense layers → sigmoid output
- Early stopping and evaluation with precision, recall, F1, and a confusion matrix
- A Streamlit interface for entering a review and viewing the model's predicted class and raw positive score
- A small hand-written challenge set to probe negation, contrast, typos, and mixed sentiment

## Project files

- `sentiment_analysis.ipynb` — training, evaluation, and model-export workflow
- `app.py` — Streamlit interface
- `requirements.txt` — Python dependencies
- `data/README.md` — dataset requirements and placement instructions
- `sentiment_model.keras` — generated after training; not included because the required CSV was not supplied and the model artifact could not be regenerated from the uploaded files alone

## Setup

Use Python 3.11 in a virtual environment. From this directory:

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Add the dataset and train

Place your original `cnn_text_classification.csv` in `data/`. It must contain:

- `text`: review text
- `label`: `0` = Negative, `1` = Positive
- optional `id`: ignored by training

Open `sentiment_analysis.ipynb`, run all cells from top to bottom, and confirm the data source and license are documented before publishing. The notebook saves `sentiment_model.keras` and writes the app code. The app in this repository is already provided as `app.py`.

## Run the app

After training creates `sentiment_model.keras` beside `app.py`:

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). The app displays a clear setup message if the model file is missing.

## Deploy on Streamlit Community Cloud

1. Push this project folder to a GitHub repository.
2. Add the real CSV if redistribution is allowed, or provide documented instructions to obtain it.
3. Run the notebook and include `sentiment_model.keras` in the repository if its size and licensing permit. Otherwise use an appropriate model-artifact hosting solution and adapt `app.py` to fetch the artifact securely.
4. In Streamlit Community Cloud, create an app from the repository, choose `app.py` as the entry point, and deploy.
5. Test positive, negative, empty, long, negated, and mixed-opinion examples after deployment.

**Important:** The app will not be usable as a live demo until a valid `sentiment_model.keras` is supplied.

## Evaluation notes and limitations

The supplied notebook's saved output reported **96.67% accuracy (87/90)** on its random test split and **75% accuracy (24/32)** on a separate hand-written challenge set. These values are copied from existing notebook outputs; this packaged project has not been freshly trained because the CSV was not included in the upload.

The gap between these results is a warning that the model may struggle with real-world language, especially negation, sarcasm, mixed opinions, and unfamiliar phrasing. The notebook also adds rule-generated examples to the training set; this is useful for experimentation but can encourage template learning. Do not present the random-split score as evidence of broad real-world reliability. A stronger evaluation should use a larger, independently sourced dataset with a documented license and no near-duplicate overlap.

The app calls the model output a **positive sentiment score**, not confidence, because sigmoid outputs are not automatically calibrated probabilities of correctness.

## Suggested LinkedIn description

Built an end-to-end sentiment analysis project using a Keras CNN and Streamlit. The workflow covers text preprocessing, training-only text vectorization, model training with early stopping, classification metrics, and an interactive demo. I also tested the model on a small hand-written challenge set and documented its weaknesses around negation and mixed sentiment. Next step: validate on a larger independent review dataset.
