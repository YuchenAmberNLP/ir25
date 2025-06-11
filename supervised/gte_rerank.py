import torch
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import os
import torch.nn.functional as F

class GTEReranker:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Get embeddings for a list of texts"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)

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
        # Get query embedding
        query_embedding = self.get_embeddings([query], batch_size=1)
        
        # Get document embeddings
        doc_texts = [item['text'] for item in doc_items]
        doc_embeddings = self.get_embeddings(doc_texts, batch_size)
        
        # Calculate similarity scores
        scores = (query_embedding @ doc_embeddings.T).squeeze().cpu().numpy() * 100
        
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
    Rerank a run file using the GTE model
    Args:
        run_file: Path to the input run file
        output_file: Path to save the reranked results
        model_path: Path to the trained GTE model
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
    reranker = GTEReranker(model_path)
    
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
            results.append(f"{qid} Q0 {item['docid']} {rank} {item['score']:.6f} gte_reranker\n")
    
    # Save results
    with open(output_file, 'w') as f:
        f.writelines(results)

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "models/gte_reranker"
    RUN_FILE = "runs/neuclir23/run.txt"  # Your initial run file
    OUTPUT_FILE = "runs/neuclir23/run.gte_reranker.txt"  # Output reranked run file
    DOC_FILE = "data/neuclir23/documents.jsonl"  # Your document file
    
    rerank_run_file(RUN_FILE, OUTPUT_FILE, MODEL_PATH, DOC_FILE) 