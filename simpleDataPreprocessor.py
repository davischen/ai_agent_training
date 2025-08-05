# -*- coding: utf-8 -*-
"""
Simple Data Preprocessor for ATLAS Agent
Matches the original preprocessing logic exactly
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class SimpleDataPreprocessor:
    """Simple data preprocessor that matches the original logic"""
    
    def __init__(self):
        self.label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
    
    def load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess data exactly like the original code"""
        
        # Load email dataset
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from {csv_path}")
        
        # Ensure correct column names and drop NaN values
        df.columns = df.columns.str.strip()
        
        # Create label mapping
        df['Label'] = df['Email Type'].map(self.label_mapping)
        
        # Keep original text (no actual preprocessing applied)
        df['preprocessed_text_combined'] = df['Email Text']
        
        # Rename columns to standard format
        df = df.rename(columns={'preprocessed_text_combined': 'text', 'Label': 'label'})
        
        # Drop NaN values
        df = df.dropna(subset=['text', 'label'])
        
        logger.info(f"After preprocessing: {len(df)} records")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        
        # Convert text and labels exactly like original
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label']
        )
        
        logger.info(f"Train Set: {len(train_df)} emails, Test Set: {len(test_df)} emails")
        
        return train_df, test_df
    
    def prepare_training_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Prepare training data in the exact format as original"""
        
        # Convert to lists exactly like original code
        train_emails = train_df['text'].astype(str).tolist()
        test_emails = test_df['text'].astype(str).tolist()
        train_labels = train_df['label'].astype(int).tolist()
        test_labels = test_df['label'].astype(int).tolist()
        
        return train_emails, test_emails, train_labels, test_labels
    
    def process_complete_pipeline(self, csv_path: str, 
                                 test_size: float = 0.2, 
                                 random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Complete processing pipeline that matches original exactly"""
        
        # Step 1: Load and preprocess
        df = self.load_and_preprocess(csv_path)
        
        # Step 2: Split data
        train_df, test_df = self.split_data(df, test_size, random_state)
        
        # Step 3: Prepare training data
        train_emails, test_emails, train_labels, test_labels = self.prepare_training_data(train_df, test_df)
        
        print(f"Train Set: {len(train_emails)} emails, Test Set: {len(test_emails)} emails")
        
        return train_emails, test_emails, train_labels, test_labels
    
    def get_dataframe_after_processing(self, csv_path: str) -> pd.DataFrame:
        """Get the processed DataFrame for inspection"""
        return self.load_and_preprocess(csv_path)

# Standalone function that exactly matches your original code
def preprocess_phishing_data(csv_path: str = "Phishing_Email.csv"):
    """
    Standalone function that exactly replicates your original preprocessing code
    """
    
    # Load email dataset
    df = pd.read_csv(csv_path)

    # Ensure correct column names and drop NaN values
    df.columns = df.columns.str.strip()
    df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df['preprocessed_text_combined'] = df['Email Text']  # No actual text preprocessing
    df = df.rename(columns={'preprocessed_text_combined': 'text', 'Label': 'label'})
    df = df.dropna(subset=['text', 'label'])

    # Convert text and labels
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_emails = train_df['text'].astype(str).tolist()
    test_emails = test_df['text'].astype(str).tolist()
    train_labels = train_df['label'].astype(int).tolist()
    test_labels = test_df['label'].astype(int).tolist()

    print(f"Train Set: {len(train_emails)} emails, Test Set: {len(test_emails)} emails")
    
    return train_emails, test_emails, train_labels, test_labels, train_df, test_df

# Enhanced version with optional text preprocessing
def preprocess_phishing_data_enhanced(csv_path: str = "Phishing_Email.csv", 
                                    apply_text_cleaning: bool = False):
    """
    Enhanced version with optional text preprocessing
    """
    
    # Load email dataset
    df = pd.read_csv(csv_path)

    # Ensure correct column names and drop NaN values
    df.columns = df.columns.str.strip()
    df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    
    # Optional text preprocessing
    if apply_text_cleaning:
        from data_preprocessor import TextPreprocessor
        text_processor = TextPreprocessor()
        
        df['preprocessed_text_combined'] = df['Email Text'].apply(
            lambda x: text_processor.process_text(x, clean=True, remove_stopwords=True)
        )
        print("Applied text cleaning and preprocessing")
    else:
        df['preprocessed_text_combined'] = df['Email Text']  # Keep original
        print("Keeping original text without preprocessing")
    
    df = df.rename(columns={'preprocessed_text_combined': 'text', 'Label': 'label'})
    df = df.dropna(subset=['text', 'label'])

    # Filter out empty texts that might result from aggressive preprocessing
    if apply_text_cleaning:
        initial_count = len(df)
        df = df[df['text'].str.strip().str.len() > 0]
        final_count = len(df)
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} empty texts after preprocessing")

    # Convert text and labels
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_emails = train_df['text'].astype(str).tolist()
    test_emails = test_df['text'].astype(str).tolist()
    train_labels = train_df['label'].astype(int).tolist()
    test_labels = test_df['label'].astype(int).tolist()

    print(f"Train Set: {len(train_emails)} emails, Test Set: {len(test_emails)} emails")
    
    return train_emails, test_emails, train_labels, test_labels, train_df, test_df

# Utility function to compare preprocessing results
def compare_preprocessing_methods(csv_path: str = "Phishing_Email.csv"):
    """Compare different preprocessing methods"""
    
    print("=" * 60)
    print("COMPARING PREPROCESSING METHODS")
    print("=" * 60)
    
    # Method 1: Original (no text preprocessing)
    print("\n1. ORIGINAL METHOD (No text preprocessing):")
    train_emails_orig, test_emails_orig, train_labels_orig, test_labels_orig, _, _ = preprocess_phishing_data_enhanced(
        csv_path, apply_text_cleaning=False
    )
    
    # Method 2: With text preprocessing
    print("\n2. ENHANCED METHOD (With text preprocessing):")
    try:
        train_emails_clean, test_emails_clean, train_labels_clean, test_labels_clean, _, _ = preprocess_phishing_data_enhanced(
            csv_path, apply_text_cleaning=True
        )
        
        print("\nSample comparison:")
        print("Original text:", train_emails_orig[0][:100] + "...")
        print("Cleaned text: ", train_emails_clean[0][:100] + "...")
        
    except Exception as e:
        print(f"Enhanced preprocessing failed: {e}")
        print("Using original method only.")
    
    return train_emails_orig, test_emails_orig, train_labels_orig, test_labels_orig

if __name__ == "__main__":
    # Test the simple preprocessor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Method 1: Using class-based approach
    print("Testing SimpleDataPreprocessor class:")
    preprocessor = SimpleDataPreprocessor()
    
    try:
        train_emails, test_emails, train_labels, test_labels = preprocessor.process_complete_pipeline("Phishing_Email.csv")
        print("✅ Class-based preprocessing successful")
    except FileNotFoundError:
        print("⚠️ Phishing_Email.csv not found. Creating sample data...")
        
        # Create sample data for testing
        sample_data = {
            'Email Text': [
                "Urgent: Your account will be suspended unless you verify immediately!",
                "Hi team, please find the quarterly report attached for your review.",
                "WINNER! You've won $1,000,000! Click here to claim your prize now!",
                "Meeting reminder: Our weekly team sync is scheduled for tomorrow at 2 PM.",
                "Your PayPal account has been limited. Verify your identity to restore access."
            ],
            'Email Type': [
                'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email', 'Phishing Email'
            ]
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv('Sample_Phishing_Email.csv', index=False)
        
        # Test with sample data
        train_emails, test_emails, train_labels, test_labels = preprocessor.process_complete_pipeline("Sample_Phishing_Email.csv")
        print("✅ Sample data preprocessing successful")
    
    # Method 2: Using standalone function
    print("\nTesting standalone function:")
    try:
        train_emails, test_emails, train_labels, test_labels, train_df, test_df = preprocess_phishing_data("Phishing_Email.csv")
        print("✅ Standalone function successful")
    except FileNotFoundError:
        train_emails, test_emails, train_labels, test_labels, train_df, test_df = preprocess_phishing_data("Sample_Phishing_Email.csv")
        print("✅ Standalone function with sample data successful")
