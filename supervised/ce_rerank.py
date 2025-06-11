import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import os

class CrossEncoderReranker:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, doc_items: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Rerank documents for a given query
        Args:
            query: The query text
            doc_items: List of dictionaries containing docid and text
            batch_size: Batch size for processing
        Returns:
            List of dictionaries containing docid, text and score
        """
        # Prepare inputs
        doc_texts = [item['text'] for item in doc_items]
        scores = []
        
        # Process in batches
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i + batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_scores = outputs.logits.squeeze().cpu().numpy()
                
            scores.extend(batch_scores)
        
        # Combine documents with scores
        results = [
            {
                'docid': item['docid'],
                'text': item['text'],
                'score': float(score)
            }
            for item, score in zip(doc_items, scores)
        ]
        
        # Sort by score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

def rerank_run_file(run_file: str, output_file: str, model_path: str, doc_file: str):
    """
    Rerank a run file using the cross encoder model
    Args:
        run_file: Path to the input run file
        output_file: Path to save the reranked results
        model_path: Path to the trained cross encoder model
        doc_file: Path to the document file
    """
    # Load documents
    docs = {}
    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            docs[doc['docid']] = doc['text']
    
    # Load run file
    runs = {}
    with open(run_file, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in runs:
                runs[qid] = []
            runs[qid].append(docid)
    
    # Initialize reranker
    reranker = CrossEncoderReranker(model_path)
    
    # Process each query
    results = []
    for qid, docids in tqdm(runs.items(), desc="Reranking queries"):
        # Prepare document items
        doc_items = [
            {'docid': docid, 'text': docs[docid]}
            for docid in docids
        ]
        
        # Rerank
        reranked = reranker.rerank(qid, doc_items)
        
        # Add to results
        for rank, item in enumerate(reranked, 1):
            results.append(f"{qid} Q0 {item['docid']} {rank} {item['score']:.6f} cross_encoder\n")
    
    # Save results
    with open(output_file, 'w') as f:
        f.writelines(results)

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "models/cross_encoder_mbert"
    RUN_FILE = "runs/neuclir23/run.txt"  # Your initial run file
    OUTPUT_FILE = "runs/neuclir23/run.cross_encoder.txt"  # Output reranked run file
    DOC_FILE = "data/neuclir23/documents.jsonl"  # Your document file
    
    rerank_run_file(RUN_FILE, OUTPUT_FILE, MODEL_PATH, DOC_FILE) 