import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# model settings
MODEL_PATH = './distilbert-reviews-genres'
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    print("Loading model and tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    
    # load label mappings
    with open('label_mappings.json', 'r') as f:
        mappings = json.load(f)
        id2label = {int(k): v for k, v in mappings['id2label'].items()}
    
    return model, tokenizer, id2label

def predict_genre(review_text, model, tokenizer, id2label):
    # tokenize the input
    inputs = tokenizer(
        review_text, 
        truncation=True, 
        padding=True, 
        max_length=MAX_LENGTH, 
        return_tensors='pt'
    ).to(DEVICE)
    
    # get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
    
    # get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # convert to label
    predicted_genre = id2label[predicted_class]
    confidence = probs[predicted_class].item()
    
    return predicted_genre, confidence, probs

def main():
    # load model
    model, tokenizer, id2label = load_model()
    
    print("\nDistilBERT Genre Classifier Ready!")
    print("Enter a book review to predict its genre (or 'quit' to exit)\n")
    
    while True:
        review = input("Enter review: ")
        
        if review.lower() == 'quit':
            break
        
        if not review.strip():
            continue
        
        # predict
        genre, confidence, probs = predict_genre(review, model, tokenizer, id2label)
        
        print(f"\nPredicted Genre: {genre}")
        print(f"Confidence: {confidence:.2%}\n")
        
        # show all probabilities
        print("All genre probabilities:")
        for idx, prob in enumerate(probs):
            print(f"  {id2label[idx]}: {prob.item():.2%}")
        print()

if __name__ == '__main__':
    main()
