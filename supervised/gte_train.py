import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.nn import MarginRankingLoss

class GTETripletDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query
        query_encoding = self.tokenizer(
            item["query"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize positive document
        pos_doc_encoding = self.tokenizer(
            item["pos_doc"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize negative document
        neg_doc_encoding = self.tokenizer(
            item["neg_doc"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'pos_doc_input_ids': pos_doc_encoding['input_ids'].squeeze(),
            'pos_doc_attention_mask': pos_doc_encoding['attention_mask'].squeeze(),
            'neg_doc_input_ids': neg_doc_encoding['input_ids'].squeeze(),
            'neg_doc_attention_mask': neg_doc_encoding['attention_mask'].squeeze()
        }

def train_model(
    model_name: str = "Alibaba-NLP/gte-multilingual-base",
    train_data_path: str = "data/train/triplet_train_23.jsonl",
    output_dir: str = "models/gte_reranker",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    margin: float = 0.5
):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    # Create dataset and dataloader
    train_dataset = GTETripletDataset(train_data_path, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader))
    
    # Initialize loss function
    ranking_loss = MarginRankingLoss(margin=margin)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            pos_doc_input_ids = batch['pos_doc_input_ids'].to(device)
            pos_doc_attention_mask = batch['pos_doc_attention_mask'].to(device)
            neg_doc_input_ids = batch['neg_doc_input_ids'].to(device)
            neg_doc_attention_mask = batch['neg_doc_attention_mask'].to(device)
            
            # Get query embedding
            query_outputs = model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask
            )
            query_embedding = query_outputs.last_hidden_state[:, 0]
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            # Get positive document embedding
            pos_doc_outputs = model(
                input_ids=pos_doc_input_ids,
                attention_mask=pos_doc_attention_mask
            )
            pos_doc_embedding = pos_doc_outputs.last_hidden_state[:, 0]
            pos_doc_embedding = F.normalize(pos_doc_embedding, p=2, dim=1)
            
            # Get negative document embedding
            neg_doc_outputs = model(
                input_ids=neg_doc_input_ids,
                attention_mask=neg_doc_attention_mask
            )
            neg_doc_embedding = neg_doc_outputs.last_hidden_state[:, 0]
            neg_doc_embedding = F.normalize(neg_doc_embedding, p=2, dim=1)
            
            # Calculate similarity scores
            pos_scores = torch.sum(query_embedding * pos_doc_embedding, dim=1)
            neg_scores = torch.sum(query_embedding * neg_doc_embedding, dim=1)
            
            # Calculate ranking loss
            # We want pos_scores to be higher than neg_scores by at least margin
            target = torch.ones_like(pos_scores)
            loss = ranking_loss(pos_scores, neg_scores, target)
            
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Average loss: {avg_loss:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model() 