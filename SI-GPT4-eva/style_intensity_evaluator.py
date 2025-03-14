import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import os

class StyleIntensityEvaluator:
    def __init__(self, model_path: str = "style_classifier"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_texts(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        processed_texts = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            processed_texts.extend([
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                for input_ids, attention_mask in zip(
                    encoded["input_ids"],
                    encoded["attention_mask"]
                )
            ])
        return processed_texts
    
    def evaluate_style(self, texts: List[str], batch_size: int = 32) -> Tuple[float, int, int]:
        processed_texts = self.preprocess_texts(texts, batch_size)
        style_count = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(processed_texts), batch_size), desc="Evaluating style"):
                batch = processed_texts[i:i + batch_size]
                
                input_ids = torch.stack([item["input_ids"] for item in batch]).to(self.device)
                attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                style_count += (probabilities[:, 1] > 0.5).sum().item()
        
        total_count = len(texts)
        style_intensity = style_count / total_count
        
        return style_intensity, style_count, total_count

def main():
    with open("result.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [item["daiyu_answer"] for item in data]
    
    evaluator = StyleIntensityEvaluator()
    
    style_intensity, style_count, total_count = evaluator.evaluate_style(texts)
    
    results = {
        "total_samples": total_count,
        "target_style_count": style_count,
        "style_intensity": style_intensity
    }
    
    with open("si_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n评估结果:")
    print(f"总样本数: {total_count}")
    print(f"符合红楼梦风格的文本数量: {style_count}")
    print(f"风格强度 (SI): {style_intensity:.4f}")
    print(f"\n结果已保存到 si_results.json")

if __name__ == "__main__":
    main() 