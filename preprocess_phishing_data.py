#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact Phishing Data Preprocessing Script
This file contains the exact preprocessing code you provided
"""

import pandas as pd
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_phishing_dataset(csv_file_path: str = "Phishing_Email.csv"):
    """
    Exact preprocessing function that matches your provided code
    """
    
    # Load email dataset
    df = pd.read_csv(csv_file_path)
    print(f"Loaded dataset: {len(df)} records")

    # Ensure correct column names and drop NaN values
    df.columns = df.columns.str.strip()
    df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df['preprocessed_text_combined'] = df['Email Text']  # Keeping raw text
    df = df.rename(columns={'preprocessed_text_combined': 'text', 'Label': 'label'})

    # Drop rows where 'text' is NaN or contains only whitespace
    df = df.dropna(subset=['text', 'label'])

    # Apply stricter filtering to remove:
    # - Empty or whitespace-only texts
    # - Texts that contain the word "empty" (case insensitive)
    df = df[df['text'].astype(str).apply(lambda x: isinstance(x, str) and x.strip() != "" and "empty" not in x.lower())]

    print(f"After filtering: {len(df)} records")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Convert text and labels
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_emails = train_df['text'].astype(str).tolist()
    test_emails = test_df['text'].astype(str).tolist()
    train_labels = train_df['label'].astype(int).tolist()
    test_labels = test_df['label'].astype(int).tolist()

    print(f"Train Set: {len(train_emails)} emails, Test Set: {len(test_emails)} emails")

    # Identify remaining emails that still contain "empty" (for debugging)
    remaining_empty_indices = [i for i, email in enumerate(test_emails) if "empty" in email.lower()]
    if remaining_empty_indices:
        print("\n⚠️ Warning: Emails containing 'empty' still exist at indices:", remaining_empty_indices)
    else:
        print("\n✅ No emails containing 'empty' found.")
    
    return train_emails, test_emails, train_labels, test_labels, train_df, test_df

def create_training_examples(train_emails, train_labels):
    """
    Create training examples for sentence transformer training
    """
    train_examples = []
    for email, label in zip(train_emails, train_labels):
        # Find positive example from same category
        positive_examples = [e for e, l in zip(train_emails, train_labels) if l == label and e != email]
        if positive_examples:
            positive_example = random.choice(positive_examples)
            train_examples.append(InputExample(texts=[email, positive_example]))
    
    print(f"Created {len(train_examples)} training examples")
    return train_examples

def prepare_model_training(train_examples, batch_size=16):
    """
    Prepare model training components
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    return model, train_dataloader, train_loss

def train_model(model, train_dataloader, train_loss, epochs=5, output_path="finetuned_email_BERT_v3"):
    """
    Train the sentence transformer model
    """
    print(f"Starting training for {epochs} epochs...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True,
        output_path=output_path
    )

    model.save(output_path)
    print(f"Model saved to: {output_path}")
    
    return model

def evaluate_model(model, test_emails, test_labels):
    """
    Evaluate the trained model
    """
    print("Evaluating model...")
    
    # Generate embeddings
    test_embeddings = model.encode(test_emails, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    
    # Create query-to-document ranking
    query_id_to_ranked_doc_ids = {}
    for idx, query_embedding in enumerate(test_embeddings):
        similarities = cosine_similarity([query_embedding], test_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]  # Rank documents by similarity
        query_id_to_ranked_doc_ids[idx] = ranked_indices.tolist()

    print(f"Example ranking for query 0 (within test set): {query_id_to_ranked_doc_ids[0][:10]}")
    
    return query_id_to_ranked_doc_ids, test_embeddings

def calculate_performance_metrics(test_labels, query_id_to_ranked_doc_ids):
    """
    Calculate performance metrics
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Simple prediction based on top-1 ranking
    y_pred = [test_labels[query_id_to_ranked_doc_ids[i][0]] for i in range(len(test_labels))]
    
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred))
    
    conf_matrix = confusion_matrix(test_labels, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    return y_pred, conf_matrix, accuracy

def analyze_misclassified_examples(test_emails, test_labels, y_pred, max_examples=10):
    """
    Analyze misclassified examples
    """
    print("\nMisclassified Examples:")

    misclassified_indices = [
        i for i in range(len(test_labels))
        if test_labels[i] != y_pred[i] and test_emails[i].strip()  # Exclude empty text
    ]

    if not misclassified_indices:
        print("No misclassified examples found with non-empty text.")
    else:
        for idx in misclassified_indices[:max_examples]:  # Print only first N misclassified samples
            true_label = "Safe Email" if test_labels[idx] == 0 else "Phishing Email"
            predicted_label = "Safe Email" if y_pred[idx] == 0 else "Phishing Email"

            print(f"🔹 Index {idx}:")
            print(f"   ✅ True Label     : {true_label}")
            print(f"   ❌ Predicted Label: {predicted_label}")
            print(f"   📧 Email Text:\n   {test_emails[idx][:200]}...\n")

def complete_pipeline(csv_file_path: str = "Phishing_Email.csv", 
                     epochs: int = 5, 
                     batch_size: int = 16,
                     output_model_path: str = "finetuned_email_BERT_v3"):
    """
    Complete training pipeline
    """
    print("🚀 Starting Complete Phishing Email Detection Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Preprocess data
        print("\n📊 Step 1: Preprocessing data...")
        train_emails, test_emails, train_labels, test_labels, train_df, test_df = preprocess_phishing_dataset(csv_file_path)
        
        # Step 2: Create training examples
        print("\n🎯 Step 2: Creating training examples...")
        train_examples = create_training_examples(train_emails, train_labels)
        
        # Step 3: Prepare model training
        print("\n🧠 Step 3: Preparing model training...")
        model, train_dataloader, train_loss = prepare_model_training(train_examples, batch_size)
        
        # Step 4: Train model
        print("\n🎓 Step 4: Training model...")
        trained_model = train_model(model, train_dataloader, train_loss, epochs, output_model_path)
        
        # Step 5: Evaluate model
        print("\n📈 Step 5: Evaluating model...")
        query_rankings, test_embeddings = evaluate_model(trained_model, test_emails, test_labels)
        
        # Step 6: Calculate metrics
        print("\n📊 Step 6: Calculating performance metrics...")
        y_pred, conf_matrix, accuracy = calculate_performance_metrics(test_labels, query_rankings)
        
        # Step 7: Analyze misclassified examples
        print("\n🔍 Step 7: Analyzing misclassified examples...")
        analyze_misclassified_examples(test_emails, test_labels, y_pred)
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Model saved to: {output_model_path}")
        
        return {
            'model': trained_model,
            'train_data': (train_emails, train_labels),
            'test_data': (test_emails, test_labels),
            'predictions': y_pred,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_sample_data_if_needed():
    """
    Create sample data if Phishing_Email.csv doesn't exist
    """
    if not os.path.exists("Phishing_Email.csv"):
        print("Creating sample Phishing_Email.csv for testing...")
        
        sample_data = {
            'Email Text': [
                "URGENT: Your account will be suspended unless you verify immediately at secure-bank-verify.com",
                "Hi team, please find the quarterly financial report attached for your review.",
                "WINNER! You've won $1,000,000! Click here to claim your prize now!",
                "Meeting reminder: Our weekly team sync is scheduled for tomorrow at 2 PM.",
                "Your PayPal account has been limited. Verify your identity to restore access.",
                "Thank you for your purchase. Your order #12345 has been shipped.",
                "SECURITY ALERT: Suspicious login detected. Confirm your identity immediately.",
                "Project update: The new feature development is on track for next week's release.",
                "Your credit card will expire soon. Update your information to avoid service interruption.",
                "System maintenance scheduled: Our servers will be down this weekend for upgrades.",
                "CONGRATULATIONS! You have been selected for a special offer. Click now!",
                "Document review needed: Please check the attached contract by Friday.",
                "Account verification required: Your account access will be restricted soon.",
                "Team lunch: Join us at the new restaurant downtown next Tuesday.",
                "FINAL NOTICE: Your subscription will be cancelled unless you act now.",
                "Budget meeting: Please prepare your Q4 budget proposals for Monday's meeting.",
                "Security breach detected: Update your password immediately to protect your account.",
                "Conference call: Dial-in details for tomorrow's client presentation attached.",
                "Prize notification: You've won a free vacation! Claim within 24 hours.",
                "Server maintenance: Brief outage scheduled for Sunday night maintenance window."
            ],
            'Email Type': [
                'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email', 'Phishing Email',
                'Safe Email', 'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email',
                'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email', 'Phishing Email',
                'Safe Email', 'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('Phishing_Email.csv', index=False)
        print("✅ Sample data created: Phishing_Email.csv")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample data if needed
    create_sample_data_if_needed()
    
    # Run the complete pipeline
    result = complete_pipeline(
        csv_file_path="Phishing_Email.csv",
        epochs=3,  # Reduced for faster testing
        batch_size=8,  # Smaller batch for testing
        output_model_path="test_finetuned_model"
    )
    
    if result:
        print(f"\n✅ Training completed with accuracy: {result['accuracy']:.4f}")
    else:
        print("\n❌ Training failed")
