# Assignment 3: Hugging Face Model Deployment

## Overview
This assignment demonstrates the deployment of a pre-trained Hugging Face model (DistilBERT) fine-tuned for Fake News Detection. The project showcases transfer learning, model deployment, and REST API creation using Flask.

## Project Structure
```
assignment_3_hugging_face/
├── train_model.py               # Training script - fine-tune DistilBERT
├── app.py                       # Flask application
├── requirements.txt             # Dependencies
├── models/
│   ├── distilbert_fake_news/    # Fine-tuned model files
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── training_output/         # Training checkpoints
│   └── model_stats.pkl          # Model statistics
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── index.html
    └── dashboard.html
```

## Datasets

### Fake News Dataset
- **Task**: Binary Classification (Fake News / Real News)
- **Source**: Kaggle - Fake and Real News Dataset
- **Dataset ID**: clmentbisaillon/fake-and-real-news-dataset
- **Features Used**: Title + Content (combined text)
- **Classes**: 0 = Fake News, 1 = Real News
- **Training Samples**: 5,000 (subset for faster training)

## Installation & Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Important Notes
- PyTorch installation may vary based on your system (CPU/GPU)
- For GPU support: Install CUDA-compatible PyTorch version
- Transformers library will auto-download models from Hugging Face Hub

### Train Models
```bash
python train_model.py
```

**Expected Output:**
- Downloads Fake News dataset from Kaggle
- Fine-tunes DistilBERT for 3 epochs
- Evaluates on test set
- Saves model and tokenizer
- Saves metrics to `models/model_stats.pkl`

**Training Time:**
- CPU: ~15-30 minutes
- GPU: ~5-10 minutes

## Running the Application

### Start Flask Server
```bash
python app.py
```

### Access the Application
- **Home Page**: http://localhost:5003
- **Dashboard**: http://localhost:5003/dashboard
- **API Test**: http://localhost:5003/test

## Model Details

### DistilBERT
- **Base Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Type**: Transformer-based language model
- **Characteristics**:
  - 40% smaller than BERT
  - 60% faster than BERT
  - Retains 97% of BERT's performance
  - Pre-trained on large Wikipedia + BookCorpus

### Transfer Learning Approach
1. **Pre-training**: DistilBERT trained on masked language modeling
2. **Fine-tuning**: Task-specific training on Fake News classification
3. **Tokenization**: WordPiece tokenization with max length 256 tokens

## API Endpoints

### 1. Single Prediction
```
POST /predict
Content-Type: application/json

{
    "text": "news content here..."
}

Response:
{
    "success": true,
    "prediction": 1,
    "prediction_label": "Real News",
    "probability": {
        "fake_news": 0.1234,
        "real_news": 0.8766
    },
    "confidence": 0.8766,
    "text_length": 512
}
```

### 2. Batch Prediction
```
POST /predict/batch
Content-Type: application/json

{
    "texts": ["text1...", "text2...", "text3..."]
}

Response:
{
    "success": true,
    "total": 3,
    "predictions": [
        {
            "prediction": 1,
            "prediction_label": "Real News",
            "fake_prob": 0.1234,
            "real_prob": 0.8766,
            "confidence": 0.8766
        },
        ...
    ]
}
```

### 3. Get Model Statistics
```
GET /api/stats

Response: Complete model metrics and confusion matrix
```

### 4. Get Model Information
```
GET /model/info

Response:
{
    "model_type": "DistilBERT",
    "task": "Fake News Detection (Binary Classification)",
    "max_sequence_length": 256,
    "classes": ["Fake News", "Real News"],
    "device": "cuda" or "cpu",
    "test_accuracy": 0.xxxx
}
```

### 5. Test Connection
```
GET /test

Response:
{
    "status": "ok",
    "model_loaded": true,
    "tokenizer_loaded": true,
    "device": "cuda" or "cpu",
    "stats_available": true
}
```

## Evaluation Metrics

The model is evaluated on:
- **Accuracy**: Percentage of correct predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: 2×2 matrix showing TN, FP, FN, TP

## Expected Performance

### Typical Results on Fake News Dataset
| Metric | Value |
|--------|-------|
| Accuracy | 95-98% |
| Precision | 0.95-0.98 |
| Recall | 0.95-0.98 |
| F1-Score | 0.95-0.98 |
| ROC-AUC | 0.98-0.99 |

## Key Features

✅ **Pre-trained Model**: Leverages DistilBERT from Hugging Face  
✅ **Transfer Learning**: Fine-tuned on real Fake News dataset  
✅ **REST API**: Complete prediction endpoints  
✅ **Batch Processing**: Process multiple texts simultaneously  
✅ **Confidence Scores**: Probability for both classes  
✅ **GPU Support**: CUDA acceleration when available  
✅ **Interactive Dashboard**: Visualize metrics with Plotly  
✅ **Comprehensive Metrics**: Full evaluation statistics  
✅ **Model Persistence**: Save and load trained models  
✅ **Error Handling**: Graceful error responses  

## Assignment Requirements Checklist

- [x] Use pre-trained Hugging Face model
- [x] Fine-tune on custom dataset
- [x] Implement Flask deployment
- [x] Create REST API endpoints
- [x] Comprehensive model metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- [x] Interactive dashboard
- [x] Batch prediction support
- [x] Model comparison (HF vs Traditional ML)
- [x] Confusion matrix visualization
- [x] Full documentation

## Comparison with Traditional ML

### DistilBERT vs Logistic Regression

| Feature | DistilBERT | Logistic Regression |
|---------|-----------|-------------------|
| Accuracy | 95-98% | 80-85% |
| Context Understanding | Excellent | Limited |
| Feature Engineering | Automatic | Manual |
| Training Time | 5-30 min | <1 min |
| Model Size | ~260 MB | <1 MB |
| Inference Speed | 100-200 ms | <1 ms |

### Advantages of Hugging Face Models
1. **Contextual Understanding**: Understands word relationships and context
2. **Transfer Learning**: Leverages pre-trained knowledge
3. **Better Performance**: Superior accuracy on complex tasks
4. **Large Community**: Extensive pre-trained models available
5. **Production Ready**: Well-tested and optimized

### When to Use DistilBERT
- Complex text classification tasks
- When accuracy is critical
- Limited labeled data (transfer learning)
- Multi-lingual support needed
- Contextual understanding required

### When to Use Traditional ML
- Simple classification with clear patterns
- Real-time predictions required
- Limited computational resources
- Need for interpretability
- Small datasets with engineered features

## Troubleshooting

### Models not found
```
Error: FileNotFoundError - models/distilbert_fake_news not found
Solution: Run python train_model.py first
```

### CUDA out of memory
```
Error: RuntimeError: CUDA out of memory
Solutions:
- Reduce batch size in train_model.py
- Use CPU: Remove GPU code or use CPU-only PyTorch
```

### Slow inference
```
Solutions:
- Use GPU for faster inference
- Reduce max_sequence_length to 128
- Use batch processing for multiple texts
```

### Kaggle API error
```
Error: kagglehub.KaggleApiError
Solution: Ensure kagglehub is installed (pip install kagglehub)
```

## Performance Optimization

### For Faster Training
1. Reduce max_sequence_length from 256 to 128
2. Reduce number of training samples
3. Use GPU acceleration
4. Increase batch size (if memory permits)

### For Faster Inference
1. Use smaller max_sequence_length
2. Batch predictions together
3. Use GPU if available
4. Cache model in memory

## Technologies Used

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library
- **DistilBERT**: Pre-trained transformer model
- **Flask**: Web framework
- **scikit-learn**: Metrics and utilities
- **NumPy & Pandas**: Data processing
- **Plotly**: Interactive visualizations
- **kagglehub**: Dataset management

## Future Enhancements

- [ ] Add more pre-trained models (RoBERTa, ALBERT)
- [ ] Implement model comparison dashboard
- [ ] Add explainability (attention visualization)
- [ ] Support for multiple languages
- [ ] Fine-tuning interface (web UI)
- [ ] Export predictions to CSV
- [ ] Add caching for repeated predictions
- [ ] API rate limiting
- [ ] Model versioning
- [ ] A/B testing framework

## References

- Hugging Face Documentation: https://huggingface.co
- DistilBERT Paper: https://arxiv.org/abs/1910.01108
- BERT Paper: https://arxiv.org/abs/1810.04805
- PyTorch Documentation: https://pytorch.org
- Transformers Library: https://huggingface.co/docs/transformers

## License
Academic Assignment - ML Course

## Contact
For issues or questions, please contact the course instructor.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the Flask app
python app.py

# Open in browser
# http://localhost:5003
```
# fake-news-classifier-fallback
