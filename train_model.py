"""
ASSIGNMENT 3: Hugging Face Model Deployment
Fine-tune DistilBERT for Fake News Detection
Deploy using Flask with prediction endpoints
"""

import os
import sys
import numpy as np
import pandas as pd
import kagglehub
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

warnings.filterwarnings('ignore')

# Create models directory
os.makedirs('models', exist_ok=True)

print("=" * 80)
print("ASSIGNMENT 3: HUGGING FACE MODEL DEPLOYMENT - TRAINING SCRIPT")
print("=" * 80)

# ============ 1. DOWNLOAD DATASET ============
print("\n[PHASE 1] Downloading Fake News Dataset from Kaggle...")
print("-" * 80)

try:
    print("Downloading fake news dataset...")
    fake_news_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    print(f"✓ Fake news dataset downloaded: {fake_news_path}")
except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    sys.exit(1)

# ============ 2. LOAD AND PREPARE DATA ============
print("\n[PHASE 2] Loading and Preparing Data...")
print("-" * 80)

try:
    import glob
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(fake_news_path, "*.csv"))
    
    # Load fake and real news
    fake_df = None
    real_df = None
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).lower()
        if 'fake' in filename:
            fake_df = pd.read_csv(csv_file)
            print(f"  Loaded: {os.path.basename(csv_file)} ({len(fake_df)} samples)")
        elif 'true' in filename:
            real_df = pd.read_csv(csv_file)
            print(f"  Loaded: {os.path.basename(csv_file)} ({len(real_df)} samples)")
    
    if fake_df is None or real_df is None:
        print("✗ Could not find both fake and real news files")
        sys.exit(1)
    
    # Prepare labels
    fake_df['label'] = 0  # Fake = 0
    real_df['label'] = 1  # Real = 1
    
    # Combine
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Total samples: {len(df)}")
    print(f"  Fake: {(df['label'] == 0).sum()}")
    print(f"  Real: {(df['label'] == 1).sum()}")
    
    # Use title + text for better content
    df['text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
    
    # Select subset for faster training
    print(f"\nUsing subset of data for faster training...")
    sample_size = min(1000, len(df))  # Keep the run practical in the current environment
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"  Training on: {len(df)} samples")
    
    # Split data
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).to_numpy()

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels,
        test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain set: {len(train_texts)} samples")
    print(f"Eval set: {len(eval_texts)} samples")
    
except Exception as e:
    print(f"✗ Error preparing data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ 3. SETUP TOKENIZER AND MODEL ============
print("\n[PHASE 3] Setting Up Tokenizer and Model...")
print("-" * 80)

try:
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    print(f"✓ Tokenizer and model loaded")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# ============ 4. CREATE DATASET CLASSES ============
print("\n[PHASE 4] Creating Custom Dataset Classes...")
print("-" * 80)

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels.tolist())
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)

try:
    print("Tokenizing training data...")
    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
    
    print("Tokenizing evaluation data...")
    eval_dataset = FakeNewsDataset(eval_texts, eval_labels, tokenizer)
    
    print(f"✓ Datasets created")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Eval: {len(eval_dataset)} samples")
    
except Exception as e:
    print(f"✗ Error creating datasets: {e}")
    sys.exit(1)

# ============ 5. FINE-TUNE MODEL ============
print("\n[PHASE 5] Fine-tuning DistilBERT Model...")
print("-" * 80)

try:
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/training_output',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    print("  (This may take 5-10 minutes depending on your hardware)")
    
    trainer.train()
    
    print("✓ Training completed")
    
except Exception as e:
    print(f"✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ 6. EVALUATE MODEL ============
print("\n[PHASE 6] Evaluating Model...")
print("-" * 80)

try:
    print("Evaluating on test set...")
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in DataLoader(eval_dataset, batch_size=16):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of real news
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    hf_accuracy = accuracy_score(all_labels, all_preds)
    hf_precision = precision_score(all_labels, all_preds, average='weighted')
    hf_recall = recall_score(all_labels, all_preds, average='weighted')
    hf_f1 = f1_score(all_labels, all_preds, average='weighted')
    hf_roc_auc = roc_auc_score(all_labels, all_probs)
    hf_cm = confusion_matrix(all_labels, all_preds)
    
    print(f"✓ Model Evaluation Complete")
    print(f"\n  Accuracy:  {hf_accuracy:.4f} ({hf_accuracy*100:.2f}%)")
    print(f"  Precision: {hf_precision:.4f}")
    print(f"  Recall:    {hf_recall:.4f}")
    print(f"  F1-Score:  {hf_f1:.4f}")
    print(f"  ROC-AUC:   {hf_roc_auc:.4f}")
    
except Exception as e:
    print(f"✗ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ 7. SAVE MODEL AND TOKENIZER ============
print("\n[PHASE 7] Saving Model and Tokenizer...")
print("-" * 80)

try:
    # Save model and tokenizer
    model_path = 'models/distilbert_fake_news'
    os.makedirs(model_path, exist_ok=True)
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"✓ Model saved: {model_path}")
    
    # Save statistics
    stats = {
        "model_type": "DistilBERT",
        "task": "Fake News Detection (Binary Classification)",
        "accuracy": round(hf_accuracy, 4),
        "precision": round(hf_precision, 4),
        "recall": round(hf_recall, 4),
        "f1_score": round(hf_f1, 4),
        "roc_auc": round(hf_roc_auc, 4),
        "confusion_matrix": hf_cm.tolist(),
        "test_samples": len(eval_labels),
        "model_path": model_path,
        "tokenizer_path": model_path,
        "classes": ["Fake News", "Real News"],
        "max_sequence_length": 256
    }
    
    joblib.dump(stats, 'models/model_stats.pkl')
    print(f"✓ Statistics saved: models/model_stats.pkl")
    
except Exception as e:
    print(f"✗ Error saving model: {e}")
    sys.exit(1)

# ============ 8. PRINT SUMMARY ============
print("\n" + "=" * 80)
print("ASSIGNMENT 3 - TRAINING COMPLETE")
print("=" * 80)

print("\n📊 MODEL METRICS:")
print("-" * 80)
print(f"{'Metric':<15} {'Value':<15}")
print("-" * 80)
print(f"{'Accuracy':<15} {hf_accuracy:.4f} ({hf_accuracy*100:.2f}%)")
print(f"{'Precision':<15} {hf_precision:.4f}")
print(f"{'Recall':<15} {hf_recall:.4f}")
print(f"{'F1-Score':<15} {hf_f1:.4f}")
print(f"{'ROC-AUC':<15} {hf_roc_auc:.4f}")

print("\n📁 SAVED FILES:")
print("-" * 80)
print(f"✓ Model: models/distilbert_fake_news/")
print(f"  - pytorch_model.bin")
print(f"  - config.json")
print(f"  - tokenizer.json")
print(f"  - tokenizer_config.json")
print(f"  - vocab.txt")
print(f"✓ Statistics: models/model_stats.pkl")

print("\n📊 CONFUSION MATRIX:")
print("-" * 80)
print(f"{'':10} {'Predicted Fake':>15} {'Predicted Real':>15}")
print(f"{'Actual Fake':10} {hf_cm[0, 0]:>15} {hf_cm[0, 1]:>15}")
print(f"{'Actual Real':10} {hf_cm[1, 0]:>15} {hf_cm[1, 1]:>15}")

print("\n🚀 NEXT STEPS:")
print("-" * 80)
print("1. Run: python app.py")
print("2. Open: http://localhost:5003")
print("3. Use the dashboard to make predictions")

print("\n" + "=" * 80)
print("✓ Training complete! Model ready for Flask deployment.")
print("=" * 80)
