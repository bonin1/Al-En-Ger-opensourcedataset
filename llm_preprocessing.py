
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import random
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import tiktoken

@dataclass
class LLMConfig:
    """Configuration for LLM preprocessing"""
    max_tokens: int = 2048
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augment_data: bool = True
    include_context: bool = True
    format_type: str = "chat"  # "chat", "instruction", "completion"

class LLMPreprocessor:
    """Advanced preprocessing for LLM fine-tuning"""
    
    def __init__(self, dataset_path: str, config: LLMConfig = None):
        self.dataset_path = dataset_path
        self.config = config or LLMConfig()
        self.data = self.load_dataset()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
    def load_dataset(self) -> Dict:
        """Load the JSON dataset"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def create_chat_format(self, entry: Dict) -> Dict:
        """Create chat format for instruction tuning"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert multilingual translator specializing in Albanian, English, and German. You provide accurate translations while preserving context, tone, and cultural nuances. Category: {entry['category']}, Difficulty: {entry['difficulty']}"
                },
                {
                    "role": "user", 
                    "content": f"Translate the following text from Albanian to both English and German:\n\nAlbanian: {entry['sq']}"
                },
                {
                    "role": "assistant",
                    "content": f"English: {entry['en']}\nGerman: {entry['de']}"
                }
            ],
            "metadata": {
                "id": entry["id"],
                "category": entry["category"],
                "difficulty": entry["difficulty"],
                "token_count": 0  # Will be calculated
            }
        }
    
    def create_instruction_format(self, entry: Dict) -> Dict:
        """Create instruction-response format"""
        context = ""
        if self.config.include_context:
            context = f"[Category: {entry['category']}, Difficulty: {entry['difficulty']}] "
        
        return {
            "instruction": f"{context}Translate the following Albanian text to English and German: '{entry['sq']}'",
            "input": "",
            "output": f"English: {entry['en']}\nGerman: {entry['de']}",
            "metadata": {
                "id": entry["id"],
                "category": entry["category"],
                "difficulty": entry["difficulty"]
            }
        }
    
    def create_completion_format(self, entry: Dict) -> Dict:
        """Create prompt-completion format"""
        prompt = f"Albanian: {entry['sq']}\nEnglish: {entry['en']}\nGerman:"
        completion = f" {entry['de']}"
        
        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "id": entry["id"],
                "category": entry["category"],
                "difficulty": entry["difficulty"]
            }
        }
    
    def augment_entry(self, entry: Dict) -> List[Dict]:
        """Create augmented versions of an entry"""
        augmented = [entry]  # Original
        
        # Reverse translation pairs
        if entry['category'] not in ['numbers', 'colors']:  # Skip for simple categories
            # English to Albanian + German
            aug1 = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert multilingual translator. Category: {entry['category']}, Difficulty: {entry['difficulty']}"
                    },
                    {
                        "role": "user",
                        "content": f"Translate from English to Albanian and German:\n\nEnglish: {entry['en']}"
                    },
                    {
                        "role": "assistant",
                        "content": f"Albanian: {entry['sq']}\nGerman: {entry['de']}"
                    }
                ],
                "metadata": entry.get("metadata", {})
            }
            augmented.append(aug1)
            
            # German to Albanian + English
            aug2 = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert multilingual translator. Category: {entry['category']}, Difficulty: {entry['difficulty']}"
                    },
                    {
                        "role": "user",
                        "content": f"Translate from German to Albanian and English:\n\nGerman: {entry['de']}"
                    },
                    {
                        "role": "assistant",
                        "content": f"Albanian: {entry['sq']}\nEnglish: {entry['en']}"
                    }
                ],
                "metadata": entry.get("metadata", {})
            }
            augmented.append(aug2)
        
        return augmented
    
    def filter_by_token_count(self, entries: List[Dict]) -> List[Dict]:
        """Filter entries that exceed token limit"""
        filtered = []
        
        for entry in entries:
            if self.config.format_type == "chat":
                # Calculate total tokens for chat format
                total_text = ""
                for message in entry["messages"]:
                    total_text += message["content"] + " "
                token_count = self.count_tokens(total_text)
            else:
                # For other formats
                if "instruction" in entry:
                    total_text = entry["instruction"] + entry["output"]
                else:
                    total_text = entry["prompt"] + entry["completion"]
                token_count = self.count_tokens(total_text)
            
            if token_count <= self.config.max_tokens:
                if "metadata" in entry:
                    entry["metadata"]["token_count"] = token_count
                filtered.append(entry)
        
        return filtered
    
    def balance_dataset(self, entries: List[Dict]) -> List[Dict]:
        """Balance dataset across categories and difficulties"""
        # Group by category and difficulty
        groups = {}
        for entry in entries:
            metadata = entry.get("metadata", {})
            key = (metadata.get("category", "unknown"), metadata.get("difficulty", "unknown"))
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)
        
        # Find minimum group size
        min_size = min(len(group) for group in groups.values())
        
        # Sample equally from each group
        balanced = []
        for group in groups.values():
            balanced.extend(random.sample(group, min(len(group), min_size)))
        
        random.shuffle(balanced)
        return balanced
    
    def create_training_splits(self, entries: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/validation/test splits"""
        # First split: train vs temp
        train_data, temp_data = train_test_split(
            entries, 
            test_size=(1 - self.config.train_split),
            random_state=42,
            stratify=[entry.get("metadata", {}).get("category", "unknown") for entry in entries]
        )
        
        # Second split: validation vs test
        val_size = self.config.val_split / (self.config.val_split + self.config.test_split)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=42,
            stratify=[entry.get("metadata", {}).get("category", "unknown") for entry in temp_data]
        )
        
        return train_data, val_data, test_data
    
    def process_dataset(self) -> Dict[str, List[Dict]]:
        """Process the entire dataset for LLM training"""
        print("🔄 Processing dataset for LLM fine-tuning...")
        
        processed_entries = []
        
        for entry in self.data["translations"]:
            # Create entry in specified format
            if self.config.format_type == "chat":
                formatted_entry = self.create_chat_format(entry)
            elif self.config.format_type == "instruction":
                formatted_entry = self.create_instruction_format(entry)
            else:  # completion
                formatted_entry = self.create_completion_format(entry)
            
            # Add to processed entries
            if self.config.augment_data:
                augmented = self.augment_entry(formatted_entry)
                processed_entries.extend(augmented)
            else:
                processed_entries.append(formatted_entry)
        
        print(f"📊 Generated {len(processed_entries)} entries (including augmentation)")
        
        # Filter by token count
        processed_entries = self.filter_by_token_count(processed_entries)
        print(f"🔍 Filtered to {len(processed_entries)} entries within token limit")
        
        # Balance dataset
        if len(processed_entries) > 1000:  # Only balance if we have enough data
            processed_entries = self.balance_dataset(processed_entries)
            print(f"⚖️ Balanced dataset to {len(processed_entries)} entries")
        
        # Create splits
        train_data, val_data, test_data = self.create_training_splits(processed_entries)
        
        print(f"✅ Created splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
    
    def export_for_openai(self, splits: Dict[str, List[Dict]], output_dir: str = "."):
        """Export in OpenAI fine-tuning format"""
        for split_name, data in splits.items():
            filename = f"{output_dir}/{split_name}_openai.jsonl"
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in data:
                    # Remove metadata for OpenAI format
                    clean_entry = {k: v for k, v in entry.items() if k != "metadata"}
                    f.write(json.dumps(clean_entry, ensure_ascii=False) + '\n')
            print(f"💾 Exported {len(data)} entries to {filename}")
    
    def export_for_huggingface(self, splits: Dict[str, List[Dict]], output_dir: str = "."):
        """Export in HuggingFace datasets format"""
        from datasets import Dataset, DatasetDict
        
        hf_splits = {}
        for split_name, data in splits.items():
            # Flatten the data structure for HuggingFace
            flattened = []
            for entry in data:
                if self.config.format_type == "chat":
                    flat_entry = {
                        "system": entry["messages"][0]["content"],
                        "user": entry["messages"][1]["content"],
                        "assistant": entry["messages"][2]["content"],
                        **entry["metadata"]
                    }
                else:
                    flat_entry = {**entry, **entry.get("metadata", {})}
                flattened.append(flat_entry)
            
            hf_splits[split_name] = Dataset.from_list(flattened)
        
        dataset_dict = DatasetDict(hf_splits)
        dataset_dict.save_to_disk(f"{output_dir}/hf_dataset")
        print(f"💾 Exported HuggingFace dataset to {output_dir}/hf_dataset")
    
    def generate_training_config(self) -> Dict:
        """Generate recommended training configuration"""
        total_samples = len(self.data["translations"])
        
        config = {
            "model_config": {
                "base_model": "microsoft/DialoGPT-medium",  # or other multilingual model
                "max_length": self.config.max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "training_config": {
                "learning_rate": 5e-5,
                "batch_size": 4 if total_samples < 1000 else 8,
                "num_epochs": 10 if total_samples < 500 else 5,
                "warmup_steps": min(100, total_samples // 10),
                "save_steps": max(50, total_samples // 20),
                "eval_steps": max(25, total_samples // 40),
                "gradient_accumulation_steps": 2
            },
            "dataset_info": {
                "total_entries": total_samples,
                "languages": ["Albanian", "English", "German"],
                "categories": len(set(entry["category"] for entry in self.data["translations"])),
                "estimated_training_time": f"{total_samples * 0.1:.1f} minutes"
            }
        }
        
        return config

def main():
    """Main function to run preprocessing"""
    # Configuration
    config = LLMConfig(
        max_tokens=1024,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        augment_data=True,
        include_context=True,
        format_type="chat"  # or "instruction", "completion"
    )
    
    # Initialize preprocessor
    preprocessor = LLMPreprocessor("dataset.json", config)
    
    # Process dataset
    splits = preprocessor.process_dataset()
    
    # Export in different formats
    preprocessor.export_for_openai(splits)
    preprocessor.export_for_huggingface(splits)
    
    # Generate training config
    training_config = preprocessor.generate_training_config()
    with open("training_config.json", 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
    
    print("🎉 Preprocessing complete! Ready for LLM fine-tuning.")

if __name__ == "__main__":
    main()
