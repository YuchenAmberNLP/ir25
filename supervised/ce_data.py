import json
import random
from datasets import load_dataset
from tqdm import tqdm

def load_topics(year):
    topics = {}
    with open(f'data/train/topics/neuclir{year}_topics.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            topic = json.loads(line)
            # Find English version
            eng_topic = next((t for t in topic['topics'] if t['lang'] == 'eng'), None)
            if eng_topic:
                topics[topic['topic_id']] = {
                    'query': f"{eng_topic['topic_title']} {eng_topic['topic_description']}",
                    'lang': eng_topic['lang']
                }
    return topics

def load_qrels(year):
    qrels = {}
    with open(f'data/train/qrels/neuclir{year}_qrels.txt', 'r', encoding='utf-8') as f:
        for line in f:
            qid, lang, did, score = line.strip().split()
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append((lang, did))
    return qrels

def get_random_negative(ds, lang, exclude_dids):
    """Get a random negative document from the dataset"""
    lang_map = {'en': 'eng', 'zh': 'zho', 'ru': 'rus', 'fa': 'fas'}
    dataset_lang = lang_map.get(lang, lang)
    
    # Get all document IDs for the language
    all_dids = set(ds[dataset_lang]['id'])
    # Remove positive document IDs
    available_dids = list(all_dids - set(exclude_dids))
    
    if not available_dids:
        return None
    
    # Get random document
    random_did = random.choice(available_dids)
    doc_idx = ds[dataset_lang]['id'].index(random_did)
    return {
        'did': random_did,
        'document': ds[dataset_lang]['text'][doc_idx],
        'lang': lang
    }

def generate_training_data():
    # Load dataset
    print("Loading NeuCLIR1 dataset...")
    ds = load_dataset("neuclir/neuclir1")
    
    # Load topics and qrels for neuclir23 only
    print("Loading topics and qrels...")
    topics = load_topics('23')
    qrels = load_qrels('23')
    
    # Generate training data
    print("Generating training data...")
    training_data = []
    
    for qid, topic_info in tqdm(topics.items()):
        if qid not in qrels:
            continue
            
        # Get positive examples
        positive_pairs = qrels[qid]
        
        for lang, did in positive_pairs:
            # Get document from dataset
            lang_map = {'en': 'eng', 'zh': 'zho', 'ru': 'rus', 'fa': 'fas'}
            dataset_lang = lang_map.get(lang, lang)
            
            try:
                doc_idx = ds[dataset_lang]['id'].index(did)
                document = ds[dataset_lang]['text'][doc_idx]
                
                # Add positive example
                training_data.append({
                    'qid': qid,
                    'query': topic_info['query'],
                    'did': did,
                    'document': document,
                    'lang': lang,
                    'label': 1
                })
                
                # Generate 3 negative examples
                exclude_dids = [did]  # Exclude current positive document
                for _ in range(3):
                    neg_example = get_random_negative(ds, lang, exclude_dids)
                    if neg_example:
                        training_data.append({
                            'qid': qid,
                            'query': topic_info['query'],
                            'did': neg_example['did'],
                            'document': neg_example['document'],
                            'lang': neg_example['lang'],
                            'label': 0
                        })
                        exclude_dids.append(neg_example['did'])
            
            except ValueError:
                print(f"Document {did} not found in {dataset_lang} dataset")
                continue
    
    # Save training data
    print("Saving training data...")
    with open('data/train/cross_encoder_train_23.jsonl', 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(training_data)} training examples")
    print(f"Saved to data/train/cross_encoder_train_23.jsonl")

if __name__ == "__main__":
    generate_training_data() 