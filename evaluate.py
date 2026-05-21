import json
import torch
import wandb
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

# model params
MODEL_NAME = 'distilbert-base-cased'
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # initialize wandb
    wandb.init(
        project='goodreads-genre-classification',
        name='model-evaluation'
    )
    
    # load the saved pickle data
    print("Loading test data...")
    with open('genre_reviews_dict.pickle', 'rb') as f:
        genre_reviews = pickle.load(f)
    
    # load label mappings
    with open('label_mappings.json', 'r') as f:
        mappings = json.load(f)
        label2id = mappings['label2id']
        id2label = {int(k): v for k, v in mappings['id2label'].items()}
    
    # prepare test data
    test_texts = []
    test_labels = []
    
    for genre, reviews in genre_reviews.items():
        reviews = reviews[:1000]
        for review in reviews[800:]:
            test_texts.append(review)
            test_labels.append(genre)
    
    print(f"Test samples: {len(test_texts)}")
    
    # tokenize test data
    print("\nTokenizing test data...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('./distilbert-reviews-genres')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_labels_encoded = [label2id[y] for y in test_labels]
    
    # create test dataset
    test_dataset = MyDataset(test_encodings, test_labels_encoded)
    
    # load the fine-tuned model
    print("\nLoading fine-tuned model...")
    model = DistilBertForSequenceClassification.from_pretrained('./distilbert-reviews-genres').to(DEVICE)
    
    # create trainer for evaluation
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=32,
        report_to=[]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # evaluate
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    
    print("\nEvaluation Results:")
    print(f"Loss: {eval_results['eval_loss']:.4f}")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    
    # get predictions for detailed report
    print("\nGenerating predictions...")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    
    # convert to label names
    pred_labels = [id2label[p] for p in preds]
    
    # classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_labels))
    
    # save evaluation results
    results = {
        'loss': eval_results['eval_loss'],
        'accuracy': eval_results['eval_accuracy'],
        'f1': eval_results['eval_f1'],
        'classification_report': classification_report(
            test_labels,
            pred_labels,
            output_dict=True
        )
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")
    
    # log metrics to wandb
    print("\nLogging to Weights & Biases...")
    wandb.log({
        'eval/loss': eval_results['eval_loss'],
        'eval/accuracy': eval_results['eval_accuracy'],
        'eval/f1': eval_results['eval_f1']
    })
    
    # create and upload artifact
    artifact = wandb.Artifact('evaluation-results', type='evaluation')
    artifact.add_file('evaluation_results.json')
    wandb.log_artifact(artifact)
    
    print("Metrics logged to W&B!")
    
    # finish wandb
    wandb.finish()
    
    print("\nEvaluation complete!")

if __name__ == '__main__':
    main()
