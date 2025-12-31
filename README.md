# Movie Review Sentiment Classifier: CNN, LSTM, and BERT (IMDB)

End-to-end sentiment analysis on the 50K IMDB movie reviews dataset (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Includes two PyTorch baselines built from scratch (CNN, bi-LSTM) plus a scaffold to fine-tune `bert-base-uncased` with Hugging Face. The notebook runs the full preprocessing pipeline (clean → tokenize → vocab → index → pad) and trains/evaluates on a 70/15/15 split.

## Results (from saved run)
- CNN (3 epochs, bs=64, lr=1e-3): Val 81.69% | Test 82.32% on CPU
- Bi-LSTM (3 epochs, bs=64, lr=1e-3): Val 83.77% | Test 84.01% on CPU
- BERT: scaffolded for future fine-tune (expected higher accuracy with GPU)

## Repo contents
- `imdb_sentiment.ipynb` — notebook with preprocessing, CNN, LSTM, and BERT section placeholder.
- `requirements.txt` — Python deps.
- `IMDB Dataset.csv` — place the Kaggle CSV here (not tracked by git).

## Setup
1) Python env
   ```bash
   python -m venv .venv
   .\\.venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2) Data: download the IMDB CSV from Kaggle and save as `IMDB Dataset.csv` in this folder.
3) Run the notebook top-to-bottom. Adjustable knobs: `MAX_LENGTH`, `min_freq` for vocab, `batch_size`, `num_epochs`, and whether the LSTM is bidirectional.
4) (Optional) BERT: in the BERT section, set `batch_size=8`, `max_length=128`, `lr=2e-5`, and train 2–3 epochs. Use a GPU for reasonable speed.

## Notes
- CPU runs for CNN/LSTM are fine; BERT is slow without GPU.
- `.gitignore` keeps large artifacts (`IMDB Dataset.csv`, `.ipynb_checkpoints`, `__pycache__`, `.venv`) out of git.

