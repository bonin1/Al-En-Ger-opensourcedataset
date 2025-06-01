# 🌍 Albanian-English-German Translation Dataset

A comprehensive multilingual translation dataset containing 630+ translation pairs across Albanian, English, and German languages, specifically designed for LLM fine-tuning and multilingual AI research.

## 📊 Dataset Overview

- **Languages**: Albanian (sq), English (en), German (de)
- **Total Entries**: 630+ translation triplets
- **Categories**: 58 semantic categories including greetings, business, travel, education, healthcare, technology, and specialized domains
- **Difficulty Levels**: Beginner, Intermediate, Advanced, Expert
- **Format**: Structured JSON with metadata for each translation
- **Version**: 1.1.0 (Last updated: 2024-07-28)

## 🎯 Key Features

### 📚 Rich Content Categories
- **Daily Life**: Greetings, common phrases, family, emotions, daily routines
- **Professional**: Business, education, healthcare, technology, law, economics
- **Travel & Commerce**: Directions, shopping, travel, food, gastronomy
- **Academic**: Numbers, time, colors, body parts, animals, mathematics, science
- **Specialized Domains**: Artificial intelligence, cybersecurity, renewable energy, quantum physics
- **Advanced Topics**: Philosophy, psychology, sociology, cultural nuances, ethics

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
| Total Translations | 630 |
| Languages | 3 (Albanian, English, German) |
| Categories | 58 |
| Difficulty Levels | 4 (Beginner, Intermediate, Advanced, Expert) |
| Average Text Length | ~20-80 characters |
| Vocabulary Richness | High diversity across domains and specializations |

### Complete Category List
The dataset covers 58 categories including:
- **Basic Communication**: greetings, common_phrases, conversations, emotions
- **Everyday Life**: family, body_parts, colors, animals, food, clothing, weather, time, numbers
- **Professional Domains**: business, education, healthcare, sports, professions, shopping, directions, travel
- **Advanced Topics**: technology, science, politics, culture, philosophy, environment, psychology, economics, law, sociology, history, mathematics
- **Specialized Fields**: artificial_intelligence, cybersecurity, renewable_energy, urban_planning, marine_biology, space_exploration, quantum_physics, ethics, gastronomy, social_media
- **Complex Language**: idioms, formal_language, literature, complex_conversations, abstract_concepts, technical_terminology, metaphorical_language, conditional_statements, cultural_nuances, emotional_complexity, academic_discourse, professional_jargon, temporal_complexity

### Difficulty Distribution
- **Beginner**: Basic vocabulary and simple phrases
- **Intermediate**: Common expressions and moderate complexity
- **Advanced**: Complex sentences and specialized terminology
- **Expert**: Highly technical and sophisticated language

## 🎨 Visualization Features

The interactive dashboard provides:

- **Overview Dashboard**: Dataset statistics and distributions across all 630 entries
- **Quality Analysis**: Text length analysis, vocabulary richness across 58 categories
- **Word Clouds**: Visual representation of most frequent terms in all three languages
- **Translation Consistency**: Analysis of translation quality across difficulty levels
- **Category Analysis**: Distribution and complexity analysis across specialized domains
- **Export Tools**: Multiple format exports optimized for different ML frameworks

## 🔬 Research Applications

### Machine Learning
- **Neural Machine Translation (NMT)** model training with Albanian support
- **Multilingual BERT** fine-tuning for Albanian-English-German
- **Cross-lingual transfer learning** research with low-resource Albanian
- **Few-shot translation** experiments across difficulty levels

### LLM Fine-tuning
- **Instruction tuning** for translation tasks with specialized domains
- **Chat model** specialization for Albanian language support
- **Multilingual capability** enhancement with technical terminology
- **Domain adaptation** spanning from basic to expert-level content

### Linguistic Research
- **Translation quality assessment** across difficulty levels
- **Cross-linguistic analysis** of Albanian, English, and German
- **Semantic similarity studies** in specialized domains
- **Cultural adaptation research** with Albanian-specific content

## 📊 Data Quality Assurance

- ✅ **Human-validated translations** for accuracy across all 630 entries
- ✅ **Balanced representation** across 58 categories
- ✅ **Progressive difficulty** from beginner to expert levels
- ✅ **Consistent formatting** and comprehensive metadata
- ✅ **Specialized domain coverage** including AI, science, and technology
- ✅ **Quality metrics** and analysis tools for all entries

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
- 🗣️ **Communication & Daily Life**: 25%
- 🏢 **Professional & Business**: 20%
- 🔬 **Science & Technology**: 15%
- ✈️ **Travel & Commerce**: 12%
- 📚 **Education & Academia**: 10%
- 🏥 **Healthcare & Medicine**: 8%
- 🌍 **Specialized Domains**: 10%

---

**⭐ If you find this dataset useful, please consider giving it a star!**

**🔄 For updates and announcements, watch this repository**
