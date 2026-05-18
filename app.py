"""
ASSIGNMENT 3: Hugging Face Model Deployment - Flask Application
Serve DistilBERT model for Fake News Detection
"""

import os
import sys
from pathlib import Path
import re
import numpy as np
import torch
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import traceback

app = Flask(__name__)
CORS(app)
BASE_DIR = Path(__file__).resolve().parent


def load_optional_model_dir(model_dir):
    path = BASE_DIR / model_dir
    return path if path.exists() else None


class FallbackNewsClassifier:
    # Extensive keywords for better classification
    fake_keywords = {
        "shocking", "breaking", "exclusive", "urgent", "exposed", "hoax", "scam",
        "busted", "fake", "lie", "lies", "lied", "corrupt", "conspiracy", "banned",
        "secret", "outrageous", "disgusting", "terrifying", "explosive", "alert", "rigged",
        "unbelievable", "disturbing", "horrific", "insane", "shocking", "fraud", "deceive",
        "misleading", "false", "conspiracy", "cover-up", "scandal", "shocking truth",
        "they don't want you", "mainstream media", "coverup", "hidden truth", "wake up",
        "proof", "evidence suppressed", "you won't believe", "doctors hate", "secret revealed",
        "shocking discovery", "never before seen", "leaked", "whistleblower", "forbidden",
        "suppressed", "censored", "censorship", "silenced", "disappeared", "vanished"
    }
    real_keywords = {
        "report", "according", "official", "statement", "confirmed", "announced",
        "evidence", "study", "research", "data", "verified", "journal", "expert",
        "spokesperson", "agency", "source", "documented", "investigation", "analysis",
        "found", "discovered", "showed", "revealed", "indicated", "suggested",
        "university", "scientist", "professor", "researcher", "dr.", "phd",
        "published", "peer-reviewed", "data shows", "statistics", "survey",
        "officials say", "government", "department", "ministry", "representative",
        "authorities", "spokesman", "spokesperson", "statement released", "announced today",
        "court", "judge", "trial", "lawsuit", "ruled", "verdict"
    }

    def _score(self, text):
        """Improved scoring with multiple heuristics"""
        text_lower = str(text).lower()
        words = re.findall(r"[a-z']+", text_lower)
        word_count = len(words)
        
        # Keyword matching
        fake_hits = sum(word in self.fake_keywords for word in words)
        real_hits = sum(word in self.real_keywords for word in words)
        keyword_score = (real_hits - fake_hits) * 0.6
        
        # Sensationalism indicators
        exclamation_count = text.count('!') * 0.3
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text)) * 0.2
        multiple_caps = text.count('!!!') * 0.5 + text.count('??') * 0.3
        question_count = text.count('?') * 0.15
        sensationalism_score = -(exclamation_count + caps_words + multiple_caps + question_count)
        
        # Word repetition (fake news often repeats keywords)
        word_freq = {}
        for word in words:
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        repetition_penalty = -sum(1 for freq in word_freq.values() if freq > 3) * 0.25
        
        # Text length bias (longer, detailed text tends to be more credible)
        length_bonus = 0.4 if word_count > 100 else (0.2 if word_count > 50 else -0.3)
        
        # Quote markers (real news often has quotes)
        quote_bonus = text.count('"') * 0.1 + text.count("'") * 0.05
        
        # Balance towards center if no strong signals
        base_score = 0.1  # Slight real bias as default
        
        total_score = keyword_score + sensationalism_score + repetition_penalty + length_bonus + quote_bonus + base_score
        return total_score

    def predict_proba(self, texts):
        """Generate probabilities for fake/real classification"""
        probabilities = []
        for text in texts:
            score = self._score(text)
            # Use sigmoid to convert score to probability, but clamp to avoid extreme values
            real_prob = 1 / (1 + np.exp(-np.clip(score, -5, 5)))
            fake_prob = 1 - real_prob
            # Ensure valid probabilities
            fake_prob = max(0.01, min(0.99, fake_prob))
            real_prob = 1 - fake_prob
            probabilities.append([fake_prob, real_prob])
        return np.array(probabilities)

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

# ============ LOAD MODEL AND TOKENIZER ============
print("Loading Hugging Face model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = load_optional_model_dir('models/distilbert_fake_news')
tokenizer = None
model = None
model_mode = 'fallback'

if model_path is not None:
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
        model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
        model.to(device)
        model.eval()
        model_mode = 'distilbert'
    except Exception as e:
        print(f"✗ Error loading model bundle: {e}")

if model is None:
    model = FallbackNewsClassifier()

stats_path = BASE_DIR / 'models' / 'model_stats.pkl'
stats = joblib.load(stats_path) if stats_path.exists() else {
    "available": False,
    "model_type": "DistilBERT",
    "task": "Binary Text Classification",
    "max_sequence_length": 256,
    "classes": ["fake_news", "real_news"],
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "roc_auc": 0.0,
    "test_samples": 0,
    "confusion_matrix": [[0, 0], [0, 0]],
    "message": "Training artifacts are not present in this workspace yet."
}

if model_mode == 'distilbert' and tokenizer is not None:
    print("✓ Model and tokenizer loaded successfully!")
    print(f"  Device: {device}")
    print(f"  Classes: {stats['classes']}")
else:
    # Fallback: model works but DistilBERT artifacts are missing.
    # Preserve any existing metrics file's `available` flag so demo stats can be shown.
    stats["model_type"] = "Offline Fallback Text Classifier"
    if not stats.get("available", False):
        stats["available"] = False
    stats["message"] = "DistilBERT artifacts are unavailable in this workspace; using a lightweight offline classifier for demo predictions. If evaluation metrics exist they will be shown."
    print("! DistilBERT artifacts are unavailable; using offline fallback classifier so predictions still work.")

# ============ ROUTES ============

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', stats=stats)

@app.route('/dashboard')
def dashboard():
    """Dashboard with metrics"""
    return render_template('dashboard.html', stats=stats)

@app.route('/api/stats')
def get_stats():
    """Get model statistics"""
    return jsonify(stats)


def ensure_model_ready():
    if model is None:
        return jsonify({"success": False, "error": "Model is unavailable in this workspace."}), 503
    return None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if news is fake or real
    Expected JSON: {"text": "news content here..."}
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({"success": False, "error": "Text is empty"}), 400
        
        if model_mode == 'distilbert' and tokenizer is not None:
            # Tokenize
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors='pt'
            )

            # Move to device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Predict
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).item()
                probabilities = probs.cpu().numpy()[0]
        else:
            probabilities = model.predict_proba([text])[0]
            prediction = int(model.predict([text])[0])
        
        result = {
            "success": True,
            "prediction": int(prediction),
            "prediction_label": stats['classes'][prediction],
            "probability": {
                "fake_news": round(float(probabilities[0]), 4),
                "real_news": round(float(probabilities[1]), 4)
            },
            "confidence": round(float(max(probabilities)), 4),
            "text_length": len(text)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple texts
    Expected JSON: {"texts": ["text1", "text2", ...]}
    """
    try:
        data = request.get_json()
        
        if 'texts' not in data:
            return jsonify({"success": False, "error": "No texts provided"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"success": False, "error": "texts must be a list"}), 400
        
        results = []
        
        for text in texts:
            if not text or len(text.strip()) == 0:
                results.append({
                    "prediction": None,
                    "error": "Empty text"
                })
                continue
            
            try:
                if model_mode == 'distilbert' and tokenizer is not None:
                    # Tokenize
                    inputs = tokenizer(
                        text,
                        truncation=True,
                        max_length=256,
                        padding=True,
                        return_tensors='pt'
                    )

                    # Move to device
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    # Predict
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        prediction = torch.argmax(logits, dim=-1).item()
                        probabilities = probs.cpu().numpy()[0]
                else:
                    probabilities = model.predict_proba([text])[0]
                    prediction = int(model.predict([text])[0])
                
                results.append({
                    "prediction": int(prediction),
                    "prediction_label": stats['classes'][prediction],
                    "fake_prob": round(float(probabilities[0]), 4),
                    "real_prob": round(float(probabilities[1]), 4),
                    "confidence": round(float(max(probabilities)), 4)
                })
                
            except Exception as e:
                results.append({
                    "prediction": None,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "total": len(texts),
            "predictions": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/model/info')
def model_info():
    """Get model information"""
    return jsonify({
        "model_type": stats["model_type"],
        "task": stats["task"],
        "max_sequence_length": stats["max_sequence_length"],
        "classes": stats["classes"],
        "device": str(device),
        "test_accuracy": stats["accuracy"],
        "available": True,
        "backend": model_mode
    })

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "backend": model_mode,
        "device": str(device),
        "stats_available": stats is not None
    })

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ASSIGNMENT 3: Hugging Face Model Deployment - Flask Server")
    print("=" * 70)
    print(f"Starting server on http://localhost:5003")
    print(f"Dashboard: http://localhost:5003/dashboard")
    print(f"API Test: http://localhost:5003/test")
    print("=" * 70 + "\n")
    
    app.run(debug=True, port=5003)
