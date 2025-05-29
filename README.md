# 🌍 Albanian-English-German Translation Dataset

A comprehensive multilingual translation dataset containing 500+ translation pairs across Albanian, English, and German languages, specifically designed for LLM fine-tuning and multilingual AI research.

## 📊 Dataset Overview

- **Languages**: Albanian (sq), English (en), German (de)
- **Total Entries**: 500+ translation triplets
- **Categories**: 20+ semantic categories including greetings, business, travel, education, healthcare, and more
- **Difficulty Levels**: Beginner, Intermediate, Advanced
- **Format**: Structured JSON with metadata for each translation

## 🎯 Key Features

### 📚 Rich Content Categories
- **Daily Life**: Greetings, common phrases, family, emotions
- **Professional**: Business, education, healthcare, technology
- **Travel & Commerce**: Directions, shopping, travel, food
- **Academic**: Numbers, time, colors, body parts, animals
- **Conversational**: Real-world dialogue scenarios

### 🔧 Advanced Preprocessing Tools
- **Multiple Export Formats**: OpenAI fine-tuning, HuggingFace datasets, CSV, JSONL
- **Data Augmentation**: Automatic generation of reverse translation pairs
- **Quality Analysis**: Token counting, grammar checking, consistency analysis
- **Interactive Visualization**: Comprehensive dashboard for dataset exploration

### 🤖 LLM-Ready Formats
- **Chat Format**: For instruction-tuning modern LLMs
- **Instruction Format**: Traditional instruction-response pairs
- **Completion Format**: Prompt-completion for older models

## 📁 Repository Structure

```
├── dataset.json                 # Main dataset file
├── dataset_visualizer.py        # Interactive visualization tool
├── llm_preprocessing.py         # Advanced preprocessing utilities
├── run_visualizer.py           # Quick start script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/bonin1/Al-En-Ger-opensourcedataset.git
cd Al-En-Ger-opensourcedataset
pip install -r requirements.txt
```

### 2. Explore the Dataset

```python
import json

# Load the dataset
with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# View a sample translation
sample = data['translations'][0]
print(f"Albanian: {sample['sq']}")
print(f"English: {sample['en']}")
print(f"German: {sample['de']}")
```

### 3. Launch Interactive Visualizer

```bash
python run_visualizer.py
```

### 4. Preprocess for LLM Training

```python
from llm_preprocessing import LLMPreprocessor, LLMConfig

# Configure preprocessing
config = LLMConfig(
    format_type="chat",
    augment_data=True,
    max_tokens=1024
)

# Process dataset
preprocessor = LLMPreprocessor("dataset.json", config)
splits = preprocessor.process_dataset()

# Export for training
preprocessor.export_for_openai(splits)
preprocessor.export_for_huggingface(splits)
```

## 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Translations | 500+ |
| Languages | 3 (Albanian, English, German) |
| Categories | 20+ |
| Difficulty Levels | 3 |
| Average Text Length | ~15-50 characters |
| Vocabulary Richness | High diversity across domains |

## 🎨 Visualization Features

The interactive dashboard provides:

- **Overview Dashboard**: Dataset statistics and distributions
- **Quality Analysis**: Text length, vocabulary richness, readability scores
- **Word Clouds**: Visual representation of most frequent terms
- **Translation Consistency**: Analysis of translation quality and consistency
- **Clustering Analysis**: Semantic grouping of translations
- **Export Tools**: Multiple format exports for different use cases

## 🔬 Research Applications

### Machine Learning
- **Neural Machine Translation (NMT)** model training
- **Multilingual BERT** fine-tuning
- **Cross-lingual transfer learning** research
- **Few-shot translation** experiments

### LLM Fine-tuning
- **Instruction tuning** for translation tasks
- **Chat model** specialization
- **Multilingual capability** enhancement
- **Domain adaptation** for Albanian language

### Linguistic Research
- **Translation quality assessment**
- **Cross-linguistic analysis**
- **Semantic similarity studies**
- **Cultural adaptation research**

## 📊 Data Quality Assurance

- ✅ **Human-validated translations** for accuracy
- ✅ **Balanced representation** across categories
- ✅ **Consistent formatting** and structure
- ✅ **Metadata enrichment** for analysis
- ✅ **Quality metrics** and analysis tools

## 🛠️ Advanced Features

### Data Augmentation
- Automatic reverse translation generation
- Context-aware prompt creation
- Difficulty-based sampling
- Category balancing

### Quality Analysis
- Token counting and filtering
- Grammar checking (where available)
- Translation consistency scoring
- Vocabulary richness metrics

### Export Formats

| Format | Use Case | Output |
|--------|----------|--------|
| OpenAI JSONL | GPT fine-tuning | `train_openai.jsonl` |
| HuggingFace | Transformers training | `hf_dataset/` |
| CSV | Data analysis | `train_data.csv` |
| JSON | Custom processing | `processed_data.json` |

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Add translations**: Expand the dataset with new translation pairs
2. **Improve quality**: Review and refine existing translations
3. **Add features**: Enhance visualization or preprocessing tools
4. **Report issues**: Identify and report data quality issues
5. **Documentation**: Improve documentation and examples

## 📋 Data Format

### Dataset Structure
```json
{
  "metadata": {
    "name": "Albanian-English-German Translation Dataset",
    "version": "1.0",
    "total_entries": 500,
    "languages": ["sq", "en", "de"],
    "categories": ["greetings", "common_phrases", ...]
  },
  "translations": [
    {
      "id": 1,
      "category": "greetings",
      "difficulty": "beginner",
      "sq": "Përshëndetje",
      "en": "Hello",
      "de": "Hallo"
    }
  ]
}
```

### Category Distribution
- 🗣️ **Conversations**: 15%
- 🏢 **Business**: 12%
- ✈️ **Travel**: 12%
- 🍽️ **Food**: 10%
- 📚 **Education**: 10%
- 🏥 **Healthcare**: 8%
- 🛍️ **Shopping**: 8%
- 🎯 **Common Phrases**: 25%

## 🔧 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- streamlit >= 1.28.0
- plotly >= 5.0.0
- scikit-learn >= 1.0.0
- tiktoken >= 0.4.0

---

**⭐ If you find this dataset useful, please consider giving it a star!**

**🔄 For updates and announcements, watch this repository**
