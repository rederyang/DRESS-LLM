import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class StyleDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(file_path: str) -> Tuple[List[str], List[int]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    
    for item in data:
        texts.extend(item['correct_answers'])
        labels.extend([1] * len(item['correct_answers']))
        
        texts.extend(item['incorrect_answers'])
        labels.extend([0] * len(item['incorrect_answers']))
    
    print(f"Loaded {len(texts)} samples from {file_path}")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    return texts, labels

def train_model(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    output_dir: str = "style_classifier"
):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels, train_preds, average='binary'
        )
        
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary'
        )
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Training Loss: {total_loss / len(train_loader):.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Training Precision: {train_precision:.4f}")
        print(f"Training Recall: {train_recall:.4f}")
        print(f"Training F1: {train_f1:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir)
            print(f"\nSaved best model to {output_dir}")
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_texts, train_labels = load_data("Train_DRC.json")
    val_texts, val_labels = load_data("Valid_DRC.json")
    
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    
    train_dataset = StyleDataset(train_texts, train_labels, tokenizer)
    val_dataset = StyleDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model = BertForSequenceClassification.from_pretrained(
        "hfl/chinese-bert-wwm-ext",
        num_labels=2
    ).to(device)
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="style_classifier"
    )

if __name__ == "__main__":
    main() 