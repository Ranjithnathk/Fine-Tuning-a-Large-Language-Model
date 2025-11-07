# Medical Specialty Classification with LoRA

Fine-tuning DistilBERT for automated clinical document routing across 40 medical specialties using Parameter-Efficient Fine-Tuning (LoRA).

## Project Overview

**Task:** Multi-class text classification of medical transcription notes

**Dataset:** galileo-ai/medical_transcription_40
- 40 medical specialties
- Implemented data cleaning and preprocessing
- 4,457 samples (3,371 train, 595 validation, 491 test)
- Real clinical transcriptions

**Model:** DistilBERT with LoRA adapters
- Base: distilbert-base-uncased (66M parameters)
- LoRA: 916K trainable parameters (98.65% reduction)
- Task: Sequence classification

---

## Setup Instructions

### Prerequisites
- Google Colab 
- T4 GPU runtime

### Installation

1. Open notebook file in Google Colab
2. Connect to GPU Runtime
3. Run Instal Libraries Cell
4. Run All Cells

---

## Project Structure
```
├── INFO_7375_Assignment3.ipynb    # Main notebook (all code)
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── label_mapping.json              # Label encodings (auto-generated)
├── train.csv                       # Training data (auto-generated)
├── validation.csv                  # Validation data (auto-generated)
├── test.csv                        # Test data (auto-generated)
├── best_model/                     # Trained LoRA model (auto-generated)
├── *.png                           # Visualizations (auto-generated)
└── *.txt                           # Reports (auto-generated)
```

---

## Key Components

### **1. Dataset Preparation**
- Loads medical transcription dataset from HuggingFace
- Preprocessing: text cleaning, remove duplicates
- Stratified train/val/test split (75.6%/13.3%/11.0%)
- Tokenization with MAX_LENGTH=512

### **2. Model Architecture**
- Base: DistilBERT (66M parameters)
- LoRA Configuration:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_lin, v_lin
  - Trainable: 916K params (1.35%)

### **3. Training**
- 3 hyperparameter configurations tested
- Learning rates optimized for LoRA: 1e-4, 3e-4, 5e-5
- Training time: ~3.4 minutes per config
- Evaluation metric: F1-macro (handles class imbalance)

### **4. Evaluation**
- Multiple baselines compared
- Comprehensive metrics: accuracy, F1-macro, F1-weighted, precision, recall
- Per-class performance analysis
- Confusion matrix visualization

### **5. Inference**
- MedicalSpecialtyClassifier class for easy usage
- Gradio web interface for interactive demo
- Batch processing support

---

## Results

**Best Model Performance (Config 2: LR=3e-4):**
- Test Accuracy: 54.18%
- F1-macro: 0.3872
- F1-weighted: 0.5330
- Improvement over baseline: 270x

**LoRA Efficiency:**
- Parameter reduction: 98.65%
- Training time: 3.4 minutes
- Inference latency: 29ms single, 57 docs/sec batch

**Key Findings:**
- LoRA requires 5-15x higher learning rates than standard fine-tuning
- Text length is primary error factor (longer texts have 2x error rate)
- Strong negative correlation (-0.313) between training samples and error rate

---

## Usage

### Running the Complete Pipeline
```python
# The notebook runs everything automatically
# Just: Runtime -> Run all

# Or run sections individually as needed
```

### Using the Trained Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=40
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./best_model")

# Tokenize and predict
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
inputs = tokenizer("Your medical text here", return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1)
```

### Using Gradio Interface
```python
# The notebook automatically launches Gradio at the end
# Or manually:
demo.launch(share=True)
```

---

## Technical Specifications

**Hardware Requirements:**
- GPU: NVIDIA T4 or equivalent (15GB VRAM)
- RAM: 12GB minimum
- Storage: 2GB for models and data

**Software Requirements:**
- Python 3.8+
- CUDA 11.x or 12.x
- See requirements.txt for package versions

**Training Time:**
- Dataset preparation: ~2 minutes
- Single config training: ~3.4 minutes  
- 3 configs + evaluation: ~1.5-2 hours total

---

## Files Generated

**Models:**
- `./best_model/` - Best performing LoRA model
- `./medical_specialty_model/` - Checkpoints

**Data:**
- `train.csv`, `validation.csv`, `test.csv` - Preprocessed splits
- `label_mapping.json` - Label-ID mappings

**Results:**
- `hyperparameter_results.csv` - Config comparison
- `baseline_comparison.csv` - Baseline vs fine-tuned
- `error_examples.csv` - Misclassified samples
- `classification_report.txt` - Detailed metrics

**Visualizations:**
- `hyperparameter_comparison.png` - Config results
- `confusion_matrix.png` - Top 15 classes
- `baseline_comparison.png` - Performance improvement

**Reports:**
- `hyperparameter_optimization_report.txt`
- `model_evaluation_report.txt`
- `training_config.json`
- `model_config.json`

---

## Troubleshooting

**Issue:** Out of memory during training
**Solution:** Reduce batch size to 8 or 16

**Issue:** LoRA model loading error
**Solution:** Use PeftModel.from_pretrained() not AutoModel

**Issue:** Gradio share link expired
**Solution:** Re-run Gradio cell to generate new link

**Issue:** Session disconnected during training
**Solution:** Keep browser tab active, don't let computer sleep

---

## Citation
```
Dataset: galileo-ai/medical_transcription_40
Model: distilbert/distilbert-base-uncased
PEFT Library: Hugging Face PEFT (LoRA)
```

## Author

**Course:** INFO 7375 - Prompt Engineering for Generative AI  
**Done by** Ranjithnath Karunanidhi
**Assignment:** Fine-Tuning a Large Language Model  

## License

Educational project - not for commercial use
```

---