# Goodreads Genre Classification with DistilBERT

A fine-tuned DistilBERT model for multi-class book review genre classification using the UCSD Goodreads dataset. This project demonstrates end-to-end MLOps practices including data processing, model training, evaluation, experiment tracking with Weights & Biases, and model deployment to Hugging Face Hub.

## Project Links

- **GitHub Repository**: https://github.com/YuvarajK-g25ait2054/Mlops_assignment2
- **Hugging Face Model**: https://huggingface.co/YuvarajK-g25ait2054/distilbert-goodreads-genres-V2
- **Weights & Biases Dashboard**: https://wandb.ai/g25ait2054-iit-jodhpur/mlops-assignment2
- **Kaggle Notebook**: https://www.kaggle.com/code/kyuvarajg25ait2054/g25ait2054-yuvaraj

## Project Overview

This project classifies book reviews into 8 different genres using transfer learning with DistilBERT. The model achieves 66.75% accuracy across 800 test samples with comprehensive experiment tracking and model versioning.

## Dataset

The project uses the **UCSD Book Graph dataset** with reviews from 8 genres:

| Genre | Training Samples | Test Samples |
|-------|-----------------|-------------|
| Poetry | 800 | 200 |
| Children | 800 | 200 |
| Comics & Graphic | 800 | 200 |
| Fantasy & Paranormal | 800 | 200 |
| History & Biography | 800 | 200 |
| Mystery/Thriller/Crime | 800 | 200 |
| Romance | 800 | 200 |
| Young Adult | 800 | 200 |
| **Total** | **6,400** | **1,600** |

**Data Source**: [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html)

## Model Architecture

**Model**: `distilbert-base-cased`

**Rationale**:
- DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of language understanding
- Ideal for text classification with limited computational resources
- The cased version preserves capitalization, important for proper nouns in book reviews
- Pre-trained on large text corpora, excellent for transfer learning

## Project Structure

```
.
├── train.py                      # Model training script with W&B tracking
├── evaluate.py                   # Model evaluation with metrics logging
├── inference.py                  # Interactive CLI for predictions
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── g25ait2054-yuvaraj.ipynb     # Jupyter notebook implementation
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Weights & Biases account (for experiment tracking)
- Hugging Face account (for model hosting)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/YuvarajK-g25ait2054/Mlops_assignment2.git
cd Mlops_assignment2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure W&B** (for experiment tracking):
```bash
wandb login
```

4. **Configure Hugging Face** (for model push):
```bash
huggingface-cli login
```

## Usage

### 1. Training

Train the model with automatic W&B experiment tracking:

```bash
python train.py
```

**What happens**:
- Downloads 2,000 reviews per genre from Goodreads dataset
- Splits data: 800 training + 200 test samples per genre
- Initializes W&B tracking for metrics and hyperparameters
- Fine-tunes DistilBERT for 3 epochs with early stopping
- Saves model locally to `./distilbert-reviews-genres`
- Pushes model to Hugging Face Hub
- Logs training metrics to W&B dashboard

**Expected duration**: ~10-15 minutes on GPU, ~45-60 minutes on CPU

### 2. Evaluation

Evaluate the fine-tuned model:

```bash
python evaluate.py
```

**Outputs**:
- Test set accuracy, precision, recall, F1 score
- Per-genre classification report
- Confusion matrix analysis
- Saves `evaluation_results.json` with detailed metrics
- Uploads evaluation artifact to W&B

### 3. Inference

Interactive prediction on custom reviews:

```bash
python inference.py
```

**Example**:
```
Enter a book review: This fantasy novel has magic, dragons, and epic battles!

Predicted Genre: fantasy_paranormal (confidence: 0.87)

Top 3 predictions:
1. fantasy_paranormal: 87.2%
2. young_adult: 8.1%
3. comics_graphic: 2.3%
```

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base Model | `distilbert-base-cased` |
| Epochs | 3 |
| Training Batch Size | 16 |
| Evaluation Batch Size | 32 |
| Learning Rate | 5e-5 (default) |
| Max Sequence Length | 512 tokens |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Warmup Steps | 100 |
| Evaluation Strategy | per epoch |
| Save Strategy | per epoch |
| Early Stopping | load_best_model_at_end |
| Device | CUDA (if available) / CPU |
| Experiment Tracking | Weights & Biases |
| Model Registry | Hugging Face Hub |

## Results
## Model Performance

| Metric    | Percentage | Score  |
|-----------|-----------:|-------:|
| Accuracy  | 66.75%     | 0.6675 |
| F1 Score  | 65.46%     | 0.6546 |
| Loss      | —          | 0.9423 |

### Per-Genre Performance

| Genre | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Comics & Graphic | 0.69 | 0.84 | 0.76 |
| History & Biography | 0.78 | 0.82 | 0.80 |
| Poetry | 0.54 | 0.55 | 0.54 |
| Children | 0.56 | 0.75 | 0.64 |
| Romance | 0.62 | 0.68 | 0.65 |
| Mystery/Thriller/Crime | 0.84 | 0.85 | 0.85 |
| Young Adult | 0.75 | 0.62 | 0.68 |
| Fantasy & Paranormal | 0.51 | 0.23 | 0.32 |

See `eval_report.json` for complete metrics.

### Project Links

| Platform | Link |
|----------|------|
| **GitHub Repository** | [https://github.com/YuvarajK-g25ait2054/Mlops_assignment2](https://github.com/YuvarajK-g25ait2054/Mlops_assignment2) |
| **Hugging Face Model** | [https://huggingface.co/YuvarajK-g25ait2054/distilbert-goodreads-genres-V2](https://huggingface.co/YuvarajK-g25ait2054/distilbert-goodreads-genres-V2) |
| **W&B Dashboard** | [https://wandb.ai/g25ait2054-iit-jodhpur/mlops-assignment2](https://wandb.ai/g25ait2054-iit-jodhpur/mlops-assignment2) |
| **Kaggle Notebook** | [https://www.kaggle.com/code/kyuvarajg25ait2054/g25ait2054-yuvaraj](https://www.kaggle.com/code/kyuvarajg25ait2054/g25ait2054-yuvaraj) |

### Using the Pre-trained Model

Instead of training from scratch, you can use the hosted model:

```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# load from hugging face
tokenizer = DistilBertTokenizerFast.from_pretrained('g25ait2054/distilbert-goodreads-genre-classifier')
model = DistilBertForSequenceClassification.from_pretrained('g25ait2054/distilbert-goodreads-genre-classifier')

# predict
review = "An epic fantasy adventure with dragons and magic!"
inputs = tokenizer(review, return_tensors='pt', truncation=True, max_length=512)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)
predicted_class = torch.argmax(probs)
```

## License

This project is for academic purposes. Dataset credit: UCSD Book Graph.

## Author

**Yuvaraj K** (g25ait2054)  
IIT Jodhpur - MLOps Assignment

---

*For questions or issues, please open an issue on GitHub.*
