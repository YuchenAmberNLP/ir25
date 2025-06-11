import json
from typing import List, Dict, Set
import random
from tqdm import tqdm
from collections import defaultdict
import os
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)

def load_topics(topics_file: str) -> Dict[str, str]:
    """Load topics from jsonl file"""
    topics = {}
    with open(topics_file, 'r', encoding='utf-8') as f:
        for line in f:
            topic = json.loads(line)
            topics[topic['qid']] = topic['text']
    return topics

def load_qrels(qrels_file: str) -> Dict[str, Set[str]]:
    """Load qrels from tsv file"""
    qrels = defaultdict(set)
    with open(qrels_file, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if int(rel) > 0:  # Only consider positive relevance
                qrels[qid].add(docid)
    return qrels

def load_documents() -> Dict[str, Dict]:
    """Load documents from NEUCLIR dataset"""
    print("Loading NEUCLIR dataset...")
    ds = load_dataset("neuclir/neuclir1")
    
    # Create a mapping of docid to document info
    docs = {}
    for split in ['zho', 'rus', 'fas']:  # Chinese, Russian, and Persian splits
        print(f"Processing {split} documents...")
        for doc in ds[split]:
            docs[doc['id']] = {
                'text': doc['title'] + " " + doc['text'],  # Add space between title and text
                'lang': split
            }
    
    return docs

def prepare_triplet_data(
    topics_file: str = "data/train/topics/neuclir23_topics.jsonl",
    qrels_file: str = "data/train/qrels/neuclir23_qrels",
    output_file: str = "data/train/triplet_train_23.jsonl",
    num_negatives: int = 3  # Number of negative examples per positive example
):
    """
    Prepare triplet data for margin ranking loss training
    Args:
        topics_file: Path to the topics file
        qrels_file: Path to the qrels file
        output_file: Path to save the output jsonl file
        num_negatives: Number of negative examples to generate per positive example
    """
    # Load data
    print("Loading data...")
    topics = load_topics(topics_file)
    qrels = load_qrels(qrels_file)
    docs = load_documents()
    
    # Group documents by language
    docs_by_lang = defaultdict(list)
    for docid, doc_info in docs.items():
        docs_by_lang[doc_info['lang']].append(docid)
    
    # Generate triplets
    triplets = []
    skipped_positives = 0
    used_positives = 0
    
    for qid, query in tqdm(topics.items(), desc="Generating triplets"):
        # Get positive documents for this query
        pos_docids = qrels[qid]
        
        for pos_docid in pos_docids:
            if pos_docid not in docs:
                skipped_positives += 1
                continue
                
            # Get the language of the positive document
            pos_doc_lang = docs[pos_docid]['lang']
            
            # Get negative documents in the same language
            neg_candidates = [
                docid for docid in docs_by_lang[pos_doc_lang]
                if docid not in pos_docids  # Exclude positive documents
            ]
            
            # Sample negative documents, using all available if less than num_negatives
            sampled_negs = random.sample(neg_candidates, min(num_negatives, len(neg_candidates)))
            
            # Create triplets
            for neg_docid in sampled_negs:
                triplet = {
                    'qid': qid,
                    'query': query,
                    'pos_docid': pos_docid,
                    'pos_doc': docs[pos_docid]['text'],
                    'neg_docid': neg_docid,
                    'neg_doc': docs[neg_docid]['text'],
                    'lang': pos_doc_lang  # Store the language for reference
                }
                triplets.append(triplet)
            used_positives += 1
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save triplets
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\nGenerated {len(triplets)} triplets")
    print(f"Used {used_positives} positive documents")
    print(f"Skipped {skipped_positives} positive documents (not found in dataset)")
    print(f"Saved to {output_file}")
    
    # Print language distribution
    lang_dist = defaultdict(int)
    for triplet in triplets:
        lang_dist[triplet['lang']] += 1
    
    print("\nLanguage distribution:")
    for lang, count in lang_dist.items():
        print(f"{lang}: {count} triplets")

if __name__ == "__main__":
    prepare_triplet_data() 