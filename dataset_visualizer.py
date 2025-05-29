import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import streamlit as st
from collections import Counter, defaultdict
import re
import unicodedata
from textstat import flesch_reading_ease, flesch_kincaid_grade
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MultilingualDatasetVisualizer:
    """Advanced visualization and preprocessing tool for multilingual translation datasets"""
    
    def __init__(self, dataset_path):
        """Initialize the visualizer with the dataset"""
        self.dataset_path = dataset_path
        self.data = self.load_dataset()
        self.df = self.create_dataframe()
        self.setup_language_tools()
        
    def load_dataset(self):
        """Load the JSON dataset"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_dataframe(self):
        """Convert JSON to pandas DataFrame for easier manipulation"""
        translations = self.data['translations']
        df = pd.DataFrame(translations)
        
        # Add text length features
        df['sq_length'] = df['sq'].str.len()
        df['en_length'] = df['en'].str.len()
        df['de_length'] = df['de'].str.len()
        
        # Add word count features
        df['sq_words'] = df['sq'].str.split().str.len()
        df['en_words'] = df['en'].str.split().str.len()
        df['de_words'] = df['de'].str.split().str.len()
        
        # Add sentence complexity metrics
        df['avg_word_length_sq'] = df['sq'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        df['avg_word_length_en'] = df['en'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        df['avg_word_length_de'] = df['de'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        
        return df
    
    def setup_language_tools(self):
        """Setup language processing tools"""
        try:
            self.lang_tool_en = language_tool_python.LanguageTool('en-US')
            self.lang_tool_de = language_tool_python.LanguageTool('de-DE')
        except:
            print("Warning: LanguageTool not available. Grammar checking disabled.")
            self.lang_tool_en = None
            self.lang_tool_de = None
    
    def create_overview_dashboard(self):
        """Create comprehensive dataset overview dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Dataset Overview', 'Category Distribution', 'Difficulty Distribution',
                'Text Length Distribution', 'Word Count Distribution', 'Language Complexity',
                'Translation Quality Matrix', 'Vocabulary Richness', 'Data Quality Score'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "pie"}, {"type": "pie"}],
                [{"type": "violin"}, {"type": "box"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # Dataset Overview
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=len(self.df),
                title={"text": "Total Translations"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # Category Distribution
        category_counts = self.df['category'].value_counts()
        fig.add_trace(
            go.Pie(labels=category_counts.index, values=category_counts.values, name="Categories"),
            row=1, col=2
        )
        
        # Difficulty Distribution
        difficulty_counts = self.df['difficulty'].value_counts()
        fig.add_trace(
            go.Pie(labels=difficulty_counts.index, values=difficulty_counts.values, name="Difficulty"),
            row=1, col=3
        )
        
        # Text Length Distribution
        for lang, color in zip(['sq', 'en', 'de'], ['blue', 'red', 'green']):
            fig.add_trace(
                go.Violin(y=self.df[f'{lang}_length'], name=f'{lang.upper()}', 
                         line_color=color, fillcolor=color, opacity=0.6),
                row=2, col=1
            )
        
        # Word Count Distribution
        for lang, color in zip(['sq', 'en', 'de'], ['blue', 'red', 'green']):
            fig.add_trace(
                go.Box(y=self.df[f'{lang}_words'], name=f'{lang.upper()} Words',
                      marker_color=color),
                row=2, col=2
            )
        
        # Language Complexity Scatter
        fig.add_trace(
            go.Scatter(
                x=self.df['en_words'], y=self.df['avg_word_length_en'],
                mode='markers', name='EN Complexity',
                marker=dict(color=self.df['difficulty'].map({'beginner': 1, 'intermediate': 2, 'advanced': 3}),
                           colorscale='Viridis', showscale=True)
            ),
            row=2, col=3
        )
        
        fig.update_layout(height=1200, showlegend=True, title_text="Multilingual Dataset Dashboard")
        return fig
    
    def analyze_text_quality(self):
        """Analyze text quality across languages"""
        quality_metrics = {
            'language': [],
            'avg_length': [],
            'vocab_richness': [],
            'readability_score': [],
            'grammar_errors': []
        }
        
        for lang in ['sq', 'en', 'de']:
            texts = self.df[lang].tolist()
            
            # Average length
            avg_length = np.mean([len(text) for text in texts])
            
            # Vocabulary richness (unique words / total words)
            all_words = ' '.join(texts).split()
            vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0
            
            # Readability (for English only)
            if lang == 'en':
                readability = np.mean([flesch_reading_ease(text) for text in texts if len(text) > 10])
            else:
                readability = None
            
            # Grammar errors (for EN and DE if available)
            grammar_errors = 0
            if lang == 'en' and self.lang_tool_en:
                for text in texts[:50]:  # Sample for performance
                    errors = self.lang_tool_en.check(text)
                    grammar_errors += len(errors)
            elif lang == 'de' and self.lang_tool_de:
                for text in texts[:50]:  # Sample for performance
                    errors = self.lang_tool_de.check(text)
                    grammar_errors += len(errors)
            
            quality_metrics['language'].append(lang.upper())
            quality_metrics['avg_length'].append(avg_length)
            quality_metrics['vocab_richness'].append(vocab_richness)
            quality_metrics['readability_score'].append(readability)
            quality_metrics['grammar_errors'].append(grammar_errors)
        
        return pd.DataFrame(quality_metrics)
    
    def create_word_clouds(self):
        """Create word clouds for each language"""
        word_clouds = {}
        
        for lang in ['sq', 'en', 'de']:
            text = ' '.join(self.df[lang].tolist())
            
            # Clean text for word cloud
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            word_clouds[lang] = wordcloud
        
        return word_clouds
    
    def analyze_translation_consistency(self):
        """Analyze consistency across translations"""
        consistency_data = []
        
        for _, row in self.df.iterrows():
            # Length ratios
            en_sq_ratio = row['en_length'] / row['sq_length'] if row['sq_length'] > 0 else 0
            en_de_ratio = row['en_length'] / row['de_length'] if row['de_length'] > 0 else 0
            sq_de_ratio = row['sq_length'] / row['de_length'] if row['de_length'] > 0 else 0
            
            # Word count ratios
            en_sq_words = row['en_words'] / row['sq_words'] if row['sq_words'] > 0 else 0
            en_de_words = row['en_words'] / row['de_words'] if row['de_words'] > 0 else 0
            
            consistency_data.append({
                'id': row['id'],
                'category': row['category'],
                'difficulty': row['difficulty'],
                'en_sq_length_ratio': en_sq_ratio,
                'en_de_length_ratio': en_de_ratio,
                'sq_de_length_ratio': sq_de_ratio,
                'en_sq_word_ratio': en_sq_words,
                'en_de_word_ratio': en_de_words,
                'length_variance': np.var([row['sq_length'], row['en_length'], row['de_length']]),
                'word_variance': np.var([row['sq_words'], row['en_words'], row['de_words']])
            })
        
        return pd.DataFrame(consistency_data)
    
    def preprocessing_for_llm(self, format_type='conversational'):
        """Preprocess data for LLM fine-tuning"""
        processed_data = []
        
        for _, row in self.df.iterrows():
            if format_type == 'conversational':
                # Chat format for instruction tuning
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are a multilingual translator specializing in Albanian, English, and German."},
                        {"role": "user", "content": f"Translate '{row['sq']}' from Albanian to English and German."},
                        {"role": "assistant", "content": f"English: {row['en']}\nGerman: {row['de']}"}
                    ],
                    "metadata": {
                        "category": row['category'],
                        "difficulty": row['difficulty'],
                        "id": row['id']
                    }
                }
            elif format_type == 'instruction':
                # Instruction-response format
                entry = {
                    "instruction": f"Translate the following Albanian text to English and German: '{row['sq']}'",
                    "response": f"English: {row['en']}\nGerman: {row['de']}",
                    "category": row['category'],
                    "difficulty": row['difficulty']
                }
            elif format_type == 'prompt_completion':
                # Simple prompt-completion format
                entry = {
                    "prompt": f"Albanian: {row['sq']}\nEnglish:",
                    "completion": f" {row['en']}\nGerman: {row['de']}",
                    "metadata": {
                        "category": row['category'],
                        "difficulty": row['difficulty']
                    }
                }
            
            processed_data.append(entry)
        
        return processed_data
    
    def create_clustering_analysis(self):
        """Perform clustering analysis on translations"""
        # Combine all texts for vectorization
        all_texts = []
        labels = []
        
        for _, row in self.df.iterrows():
            for lang in ['sq', 'en', 'de']:
                all_texts.append(row[lang])
                labels.append(f"{lang}_{row['category']}")
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(tfidf_matrix.toarray())
        
        # K-means clustering
        kmeans = KMeans(n_clusters=8, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Create DataFrame for visualization
        cluster_df = pd.DataFrame({
            'x': pca_result[:, 0],
            'y': pca_result[:, 1],
            'text': all_texts,
            'label': labels,
            'cluster': clusters
        })
        
        return cluster_df, vectorizer, pca, kmeans
    
    def export_for_training(self, output_format='jsonl', train_split=0.8):
        """Export processed data for training"""
        processed_data = self.preprocessing_for_llm()
        
        # Split data
        split_idx = int(len(processed_data) * train_split)
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        
        if output_format == 'jsonl':
            # Export as JSONL
            with open('train_data.jsonl', 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            with open('val_data.jsonl', 'w', encoding='utf-8') as f:
                for item in val_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        elif output_format == 'csv':
            # Export as CSV
            train_df = pd.DataFrame(train_data)
            val_df = pd.DataFrame(val_data)
            train_df.to_csv('train_data.csv', index=False)
            val_df.to_csv('val_data.csv', index=False)
        
        return train_data, val_data
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        report = {
            'dataset_overview': {
                'total_entries': len(self.df),
                'categories': self.df['category'].nunique(),
                'difficulties': self.df['difficulty'].nunique(),
                'languages': 3
            },
            'length_statistics': {
                'sq': {
                    'mean_length': self.df['sq_length'].mean(),
                    'mean_words': self.df['sq_words'].mean(),
                    'std_length': self.df['sq_length'].std()
                },
                'en': {
                    'mean_length': self.df['en_length'].mean(),
                    'mean_words': self.df['en_words'].mean(),
                    'std_length': self.df['en_length'].std()
                },
                'de': {
                    'mean_length': self.df['de_length'].mean(),
                    'mean_words': self.df['de_words'].mean(),
                    'std_length': self.df['de_length'].std()
                }
            },
            'balance_analysis': {
                'category_distribution': self.df['category'].value_counts().to_dict(),
                'difficulty_distribution': self.df['difficulty'].value_counts().to_dict()
            }
        }
        
        return report

def main():
    """Main function to run the visualization tool"""
    st.title("🌍 Multilingual Dataset Visualizer & Preprocessor")
    st.sidebar.title("Navigation")
    
    # Initialize visualizer
    if 'visualizer' not in st.session_state:
        try:
            st.session_state.visualizer = MultilingualDatasetVisualizer('dataset.json')
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return
    
    visualizer = st.session_state.visualizer
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview Dashboard", "Text Quality Analysis", "Word Clouds", 
         "Translation Consistency", "Clustering Analysis", "LLM Preprocessing", 
         "Quality Report", "Export Data"]
    )
    
    if page == "Overview Dashboard":
        st.header("📊 Dataset Overview Dashboard")
        fig = visualizer.create_overview_dashboard()
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Translations", len(visualizer.df))
        with col2:
            st.metric("Categories", visualizer.df['category'].nunique())
        with col3:
            st.metric("Difficulty Levels", visualizer.df['difficulty'].nunique())
    
    elif page == "Text Quality Analysis":
        st.header("📝 Text Quality Analysis")
        quality_df = visualizer.analyze_text_quality()
        st.dataframe(quality_df)
        
        # Quality visualization
        fig = px.bar(quality_df, x='language', y='vocab_richness', 
                    title='Vocabulary Richness by Language')
        st.plotly_chart(fig)
    
    elif page == "Word Clouds":
        st.header("☁️ Word Clouds")
        word_clouds = visualizer.create_word_clouds()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Albanian")
            st.image(word_clouds['sq'].to_array())
        with col2:
            st.subheader("English")
            st.image(word_clouds['en'].to_array())
        with col3:
            st.subheader("German")
            st.image(word_clouds['de'].to_array())
    
    elif page == "Translation Consistency":
        st.header("🔍 Translation Consistency Analysis")
        consistency_df = visualizer.analyze_translation_consistency()
        
        fig = px.scatter(consistency_df, x='en_sq_length_ratio', y='en_de_length_ratio',
                        color='category', title='Translation Length Consistency')
        st.plotly_chart(fig)
        
        st.subheader("Consistency Statistics")
        st.dataframe(consistency_df.describe())
    
    elif page == "Clustering Analysis":
        st.header("🎯 Clustering Analysis")
        cluster_df, vectorizer, pca, kmeans = visualizer.create_clustering_analysis()
        
        fig = px.scatter(cluster_df, x='x', y='y', color='cluster',
                        hover_data=['text'], title='Text Clustering Visualization')
        st.plotly_chart(fig)
    
    elif page == "LLM Preprocessing":
        st.header("🤖 LLM Preprocessing")
        
        format_type = st.selectbox(
            "Choose preprocessing format:",
            ["conversational", "instruction", "prompt_completion"]
        )
        
        processed_data = visualizer.preprocessing_for_llm(format_type)
        
        st.subheader("Sample Processed Entry")
        st.json(processed_data[0])
        
        st.subheader("Processing Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Processed Entries", len(processed_data))
        with col2:
            st.metric("Format Type", format_type.title())
    
    elif page == "Quality Report":
        st.header("📋 Quality Report")
        report = visualizer.generate_quality_report()
        
        st.subheader("Dataset Overview")
        st.json(report['dataset_overview'])
        
        st.subheader("Length Statistics")
        st.json(report['length_statistics'])
        
        st.subheader("Balance Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Category Distribution")
            st.bar_chart(report['balance_analysis']['category_distribution'])
        with col2:
            st.write("Difficulty Distribution")
            st.bar_chart(report['balance_analysis']['difficulty_distribution'])
    
    elif page == "Export Data":
        st.header("💾 Export Data")
        
        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox("Output Format", ["jsonl", "csv"])
        with col2:
            train_split = st.slider("Training Split", 0.5, 0.9, 0.8)
        
        if st.button("Export for Training"):
            train_data, val_data = visualizer.export_for_training(output_format, train_split)
            st.success(f"Exported {len(train_data)} training and {len(val_data)} validation samples")
            
            # Download buttons would go here in a real Streamlit app
            st.info("Files saved as train_data and val_data with the selected format")

if __name__ == "__main__":
    main()
