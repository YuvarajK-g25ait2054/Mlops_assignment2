import requests
import gzip
import json
import random
import pickle
import os
import torch
import wandb
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# model parameters
MODEL_NAME = 'distilbert-base-cased'
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# genre URLs for goodreads data
genre_urls = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0
    
    print(f"Fetching from {url[:50]}...")
    response = requests.get(url, stream=True)
    
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head and count >= head:
                break
    
    return random.sample(reviews, min(sample_size, len(reviews)))

# custom dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

def main():
    # load data for each genre
    print("Loading reviews from Goodreads...")
    genre_reviews = {}
    for genre, url in genre_urls.items():
        print(f"Genre: {genre}")
        genre_reviews[genre] = load_reviews(url, head=10000, sample_size=2000)
    
    # save the data
    with open('genre_reviews_dict.pickle', 'wb') as f:
        pickle.dump(genre_reviews, f)
    
    # split into train and test
    print("\nSplitting data into train/test sets...")
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    
    for genre, reviews in genre_reviews.items():
        reviews = random.sample(reviews, 1000)
        for review in reviews[:800]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in reviews[800:]:
            test_texts.append(review)
            test_labels.append(genre)
    
    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
    
    # create label mappings
    unique_labels = set(train_labels)
    label2id = {label: id for id, label in enumerate(unique_labels)}
    id2label = {id: label for label, id in label2id.items()}
    
    # save label mappings
    with open('label_mappings.json', 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    # tokenize the data
    print("\nTokenizing reviews...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    
    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]
    
    # create datasets
    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)
    
    # load model
    print("\nLoading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(id2label)
    ).to(DEVICE)
    
    # initialize wandb
    wandb.init(
        project='mlops-assignment2',
        name='distilbert-finetuning'
    )
    
    # set training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to='wandb'
    )
    
    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # train the model
    print("\nStarting training...")
    trainer.train()
    
    # save model locally
    print("\nSaving model...")
    trainer.save_model('./distilbert-reviews-genres')
    tokenizer.save_pretrained('./distilbert-reviews-genres')
    
    # push to hugging face hub
    print("\nPushing model to Hugging Face Hub...")
    try:
        model.push_to_hub('YuvarajK-g25ait2054/distilbert-goodreads-genres-V2')
        tokenizer.push_to_hub('YuvarajK-g25ait2054/distilbert-goodreads-genres-V2')
        print("Model pushed successfully!")
    except Exception as e:
        print(f"Could not push to HF Hub: {e}")
        print("Make sure you're logged in: huggingface-cli login")
    
    # finish wandb run
    wandb.finish()
    
    print("\nTraining complete! Model saved to ./distilbert-reviews-genres")

if __name__ == '__main__':
    main()
