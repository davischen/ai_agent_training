# -*- coding: utf-8 -*-
"""
PII Recognition Trainer for ATLAS System
Improved version with better error handling and integration
"""

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score, classification_report
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Setup logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"pii_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class PIITrainer:
    """PII Recognition Model Trainer"""
    
    def __init__(self, 
                 model_name: str = "bert-base-cased",
                 dataset_name: str = "ai4privacy/pii-masking-200k",
                 output_dir: str = None,
                 seed: int = 42):
        """
        Initialize PII Trainer
        
        Args:
            model_name: Pre-trained model name
            dataset_name: Training dataset name
            output_dir: Output directory
            seed: Random seed
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir or f"./models/pii-ner-model-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.seed = seed
        self.logger = setup_logging()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.label_list = []
        
        # Set random seed
        self._set_seed()
    
    def _set_seed(self):
        """Set random seed for reproducibility"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.logger.info(f"Random seed set to: {self.seed}")
    
    def load_dataset(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Load and preprocess dataset
        
        Args:
            max_samples: Maximum number of samples for testing
            
        Returns:
            Processed dataset information
        """
        try:
            self.logger.info(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)["train"]
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                self.logger.info(f"Limited samples to: {max_samples}")
            
            self.logger.info(f"Original dataset size: {len(dataset)}")
            
            # Filter necessary fields
            filtered_dataset = dataset.filter(
                lambda x: x["mbert_text_tokens"] is not None and x["mbert_bio_labels"] is not None
            )
            self.logger.info(f"Filtered dataset size: {len(filtered_dataset)}")
            
            # Build label mapping
            self._build_label_mapping(filtered_dataset)
            
            # Encode labels
            encoded_dataset = filtered_dataset.map(self._encode_labels)
            
            return {
                'dataset': encoded_dataset,
                'original_size': len(dataset),
                'filtered_size': len(filtered_dataset),
                'num_labels': len(self.label_list)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _build_label_mapping(self, dataset):
        """Build label mapping"""
        self.logger.info("Building label mapping...")
        unique_labels = sorted({label for seq in dataset["mbert_bio_labels"] for label in seq})
        
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.label_list = unique_labels
        
        self.logger.info(f"Number of labels: {len(unique_labels)}")
        self.logger.info(f"Label list: {unique_labels[:10]}...")  # Show only first 10
    
    def _encode_labels(self, example):
        """Encode labels"""
        example["ner_tags"] = [self.label2id[label] for label in example["mbert_bio_labels"]]
        example["tokens"] = example["mbert_text_tokens"]
        return example
    
    def prepare_model_and_tokenizer(self):
        """Prepare model and tokenizer"""
        try:
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.logger.info("Loading pre-trained model...")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
            self.model.to(self.device)
            
            self.logger.info("Model and tokenizer ready")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare model: {e}")
            raise
    
    def tokenize_and_align_labels(self, dataset):
        """Tokenize and align labels"""
        def tokenize_and_align(example):
            tokenized = self.tokenizer(
                example["tokens"], 
                is_split_into_words=True, 
                truncation=True, 
                max_length=512
            )
            
            word_ids = tokenized.word_ids()
            aligned_labels = []
            prev_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != prev_word_idx:
                    aligned_labels.append(example["ner_tags"][word_idx])
                else:
                    # Handle subword tokens
                    label = example["ner_tags"][word_idx]
                    # If B- tag, subsequent subwords use I- tag
                    if label % 2 == 1 and label < len(self.label_list) - 1:
                        aligned_labels.append(label + 1)
                    else:
                        aligned_labels.append(label)
                prev_word_idx = word_idx
            
            tokenized["labels"] = aligned_labels
            return tokenized
        
        self.logger.info("Tokenizing data and aligning labels...")
        tokenized_dataset = dataset.map(
            tokenize_and_align,
            batched=False,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def split_dataset(self, dataset, test_size: float = 0.1):
        """Split dataset"""
        self.logger.info("Splitting train and validation sets...")
        split = dataset.train_test_split(test_size=test_size, seed=self.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        
        self.logger.info(f"Training set size: {len(train_dataset)}")
        self.logger.info(f"Validation set size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, p):
        """Compute evaluation metrics"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Convert labels back to original label strings
        true_labels = [
            [self.label_list[l] for l in label_seq if l != -100]
            for label_seq in labels
        ]
        pred_labels = [
            [self.label_list[p] for p, l in zip(pred_seq, label_seq) if l != -100]
            for pred_seq, label_seq in zip(predictions, labels)
        ]
        
        # Calculate metrics
        try:
            report_dict = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            f1 = f1_score(true_labels, pred_labels)
            
            return {
                "f1": f1,
                "precision": report_dict.get("macro avg", {}).get("precision", 0.0),
                "recall": report_dict.get("macro avg", {}).get("recall", 0.0),
            }
        except Exception as e:
            self.logger.warning(f"Error computing metrics: {e}")
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    def train(self, 
              epochs: int = 1,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500,
              max_samples: Optional[int] = None,
              save_steps: int = 500,
              eval_steps: int = 500) -> Dict[str, Any]:
        """
        Train model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            max_samples: Maximum samples (for testing)
            save_steps: Save interval
            eval_steps: Evaluation interval
            
        Returns:
            Training results
        """
        try:
            # Load dataset
            dataset_info = self.load_dataset(max_samples)
            dataset = dataset_info['dataset']
            
            # Prepare model
            self.prepare_model_and_tokenizer()
            
            # Tokenize
            tokenized_dataset = self.tokenize_and_align_labels(dataset)
            
            # Split dataset
            train_dataset, eval_dataset = self.split_dataset(tokenized_dataset)
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_labels': len(self.label_list),
                'label2id': self.label2id,
                'id2label': self.id2label,
                'dataset_info': dataset_info
            }
            
            with open(os.path.join(self.output_dir, 'training_config.json'), 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                num_train_epochs=epochs,
                warmup_steps=warmup_steps,
                logging_dir=f"{self.output_dir}/logs",
                logging_steps=100,
                save_steps=save_steps,
                eval_steps=eval_steps,
                save_total_limit=3,
                evaluation_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                report_to=None,  # Disable wandb etc.
                dataloader_num_workers=2 if torch.cuda.is_available() else 0,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForTokenClassification(self.tokenizer),
                compute_metrics=self.compute_metrics
            )
            
            # Start training
            self.logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Final evaluation
            eval_result = trainer.evaluate()
            
            # Save training results
            results = {
                'train_result': train_result.metrics,
                'eval_result': eval_result,
                'model_path': self.output_dir,
                'config': config
            }
            
            with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Training completed! Model saved to: {self.output_dir}")
            self.logger.info(f"Final F1 score: {eval_result.get('eval_f1', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

# Utility functions
def create_pii_trainer(model_name: str = "bert-base-cased",
                      dataset_name: str = "ai4privacy/pii-masking-200k",
                      output_dir: str = None) -> PIITrainer:
    """Create PII trainer instance"""
    return PIITrainer(model_name, dataset_name, output_dir)

def quick_train_pii(epochs: int = 1, 
                   batch_size: int = 16, 
                   max_samples: int = 1000,
                   model_name: str = "bert-base-cased") -> Dict[str, Any]:
    """Quick train PII model (for testing)"""
    trainer = create_pii_trainer(model_name=model_name)
    return trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        max_samples=max_samples
    )

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PII Recognition Model Trainer")
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples (for testing)')
    parser.add_argument('--model-name', type=str, default="bert-base-cased", help='Pre-trained model name')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("🧪 Quick test mode")
        results = quick_train_pii(
            epochs=1, 
            batch_size=8, 
            max_samples=100,
            model_name=args.model_name
        )
    else:
        print("🚀 Full training mode")
        trainer = create_pii_trainer(
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples
        )
    
    print("✅ Training completed!")
    print(f"Model saved to: {results['model_path']}")
    if 'eval_result' in results:
        print(f"F1 score: {results['eval_result'].get('eval_f1', 'N/A')}")