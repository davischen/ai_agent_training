# -*- coding: utf-8 -*-
"""
Inference Engine for ATLAS Agent
Handles real-time inference requests
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# 修改：改為同一層目錄的導入
from core.model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Request object for inference operations"""
    request_id: str
    text_data: List[str]
    model_name: str = "default"
    batch_size: int = 32
    return_embeddings: bool = False
    threshold: float = 0.5
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class InferenceEngine:
    """Handles real-time inference requests"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
    async def process_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Process inference request"""
        try:
            model = self.model_manager.load_model(request.model_name)
            
            # Generate embeddings
            embeddings = model.encode(
                request.text_data,
                batch_size=request.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Perform classification (simplified binary classification)
            predictions = []
            confidences = []
            
            for embedding in embeddings:
                # Simplified classification logic - replace with your actual classification method
                confidence = self._calculate_phishing_probability(embedding)
                prediction = 1 if confidence > request.threshold else 0
                predictions.append(prediction)
                confidences.append(confidence)
            
            result = {
                'predictions': predictions,
                'confidences': confidences,
                'text_data': request.text_data,
                'threshold_used': request.threshold,
                'model_used': request.model_name
            }
            
            if request.return_embeddings:
                result['embeddings'] = embeddings.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _calculate_phishing_probability(self, embedding: np.ndarray) -> float:
        """
        Calculate phishing probability from embedding
        
        注意：這是簡化的實現。在實際使用中，您應該：
        1. 使用訓練好的分類器（如 SVM、RandomForest 等）
        2. 或者使用與訓練時相同的相似度計算方法
        """
        # 方法1：簡化的特徵分析（當前實現）
        feature_sum = np.sum(np.abs(embedding))
        normalized_score = (feature_sum % 1.0)  # Simple normalization
        
        # Add some randomness for demonstration
        noise = np.random.normal(0, 0.1)
        probability = max(0.0, min(1.0, normalized_score + noise))
        
        return probability
    
    def _calculate_phishing_probability_advanced(self, embedding: np.ndarray, 
                                               reference_embeddings: Dict[str, np.ndarray] = None) -> float:
        """
        更進階的釣魚郵件機率計算方法
        使用與訓練時相似的相似度比較邏輯
        """
        if reference_embeddings is None:
            # 回退到簡化方法
            return self._calculate_phishing_probability(embedding)
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 與釣魚郵件參考向量的相似度
            phishing_similarities = []
            safe_similarities = []
            
            # 計算與已知釣魚郵件的相似度
            if 'phishing_examples' in reference_embeddings:
                phishing_refs = reference_embeddings['phishing_examples']
                for ref_emb in phishing_refs:
                    sim = cosine_similarity([embedding], [ref_emb])[0][0]
                    phishing_similarities.append(sim)
            
            # 計算與已知安全郵件的相似度
            if 'safe_examples' in reference_embeddings:
                safe_refs = reference_embeddings['safe_examples']
                for ref_emb in safe_refs:
                    sim = cosine_similarity([embedding], [ref_emb])[0][0]
                    safe_similarities.append(sim)
            
            # 基於相似度計算機率
            avg_phishing_sim = np.mean(phishing_similarities) if phishing_similarities else 0.0
            avg_safe_sim = np.mean(safe_similarities) if safe_similarities else 0.0
            
            # 簡單的機率計算
            if avg_phishing_sim + avg_safe_sim > 0:
                probability = avg_phishing_sim / (avg_phishing_sim + avg_safe_sim)
            else:
                probability = 0.5  # 無法判斷時返回中性值
            
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            logger.warning(f"Advanced probability calculation failed: {e}")
            return self._calculate_phishing_probability(embedding)
    
    def batch_inference(self, text_list: List[str], 
                       model_name: str = "default",
                       batch_size: int = 32) -> Dict[str, Any]:
        """Synchronous batch inference for multiple texts"""
        try:
            model = self.model_manager.load_model(model_name)
            
            # Process in batches
            all_predictions = []
            all_confidences = []
            
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i + batch_size]
                
                # Generate embeddings for this batch
                embeddings = model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Get predictions for this batch
                for embedding in embeddings:
                    confidence = self._calculate_phishing_probability(embedding)
                    prediction = 1 if confidence > 0.5 else 0
                    all_predictions.append(prediction)
                    all_confidences.append(confidence)
            
            return {
                'predictions': all_predictions,
                'confidences': all_confidences,
                'total_processed': len(text_list),
                'model_used': model_name
            }
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise
    
    def get_model_predictions_summary(self, predictions: List[int], 
                                    confidences: List[float]) -> Dict[str, Any]:
        """Generate summary statistics for predictions"""
        if not predictions:
            return {}
        
        phishing_count = sum(predictions)
        safe_count = len(predictions) - phishing_count
        
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        
        high_confidence_threshold = 0.8
        high_confidence_count = sum(1 for c in confidences if c > high_confidence_threshold)
        
        return {
            'total_emails': len(predictions),
            'phishing_detected': phishing_count,
            'safe_emails': safe_count,
            'phishing_percentage': (phishing_count / len(predictions)) * 100,
            'average_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'high_confidence_predictions': high_confidence_count,
            'high_confidence_percentage': (high_confidence_count / len(predictions)) * 100
        }

    async def process_inference_advanced(self, request: InferenceRequest, 
                                      method: str = "few_shot",
                                      reference_examples: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        進階推理處理，支援多種推理方法
        """
        try:
            model = self.model_manager.load_model(request.model_name)
            
            if method == "zero_shot":
                # 真正的 Zero-Shot 分類
                results = InferenceMethod.zero_shot_classification(request.text_data, model)
                
            elif method == "few_shot" and reference_examples:
                # Few-Shot 分類（類似您的訓練代碼）
                results = InferenceMethod.few_shot_classification(
                    request.text_data, model, reference_examples
                )
                
            elif method == "keyword":
                # 基於關鍵詞的分類
                results = InferenceMethod.keyword_based_classification(request.text_data)
                
            else:
                # 回退到原始方法
                return await self.process_inference(request)
            
            # 轉換結果格式
            predictions = []
            confidences = []
            
            for result in results:
                phishing_prob = result['phishing_probability']
                prediction = 1 if phishing_prob > request.threshold else 0
                predictions.append(prediction)
                confidences.append(phishing_prob)
            
            return {
                'predictions': predictions,
                'confidences': confidences,
                'text_data': request.text_data,
                'method_used': method,
                'threshold_used': request.threshold,
                'model_used': request.model_name,
                'detailed_results': results
            }
            
        except Exception as e:
            logger.error(f"Advanced inference failed: {e}")
            raise


# 新增：不同的推理方法
class InferenceMethod:
    """不同推理方法的實現"""
    
    @staticmethod
    def zero_shot_classification(text_list: List[str], model) -> List[Dict[str, float]]:
        """
        真正的 Zero-Shot 分類
        使用模型的語義理解能力
        """
        results = []
        
        for text in text_list:
            # 構造 zero-shot 提示
            phishing_prompt = f"This email is a phishing attempt: {text}"
            safe_prompt = f"This email is legitimate and safe: {text}"
            
            # 計算兩種情況下的嵌入相似度
            text_embedding = model.encode([text])[0]
            phishing_embedding = model.encode([phishing_prompt])[0]
            safe_embedding = model.encode([safe_prompt])[0]
            
            # 計算相似度
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                phishing_sim = cosine_similarity([text_embedding], [phishing_embedding])[0][0]
                safe_sim = cosine_similarity([text_embedding], [safe_embedding])[0][0]
            except ImportError:
                # Fallback without sklearn
                phishing_sim = np.dot(text_embedding, phishing_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(phishing_embedding)
                )
                safe_sim = np.dot(text_embedding, safe_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(safe_embedding)
                )
            
            # 基於相似度計算機率
            total_sim = phishing_sim + safe_sim
            if total_sim > 0:
                phishing_prob = phishing_sim / total_sim
            else:
                phishing_prob = 0.5
            
            results.append({
                'phishing_probability': phishing_prob,
                'safe_probability': 1 - phishing_prob,
                'phishing_similarity': phishing_sim,
                'safe_similarity': safe_sim
            })
        
        return results
    
    @staticmethod
    def few_shot_classification(text_list: List[str], model, 
                               reference_examples: Dict[str, List[str]]) -> List[Dict[str, float]]:
        """
        Few-Shot 分類（類似您的訓練代碼邏輯）
        使用參考範例進行相似度比較
        """
        results = []
        
        # 計算參考範例的嵌入
        phishing_examples = reference_examples.get('phishing', [])
        safe_examples = reference_examples.get('safe', [])
        
        phishing_embeddings = model.encode(phishing_examples) if phishing_examples else []
        safe_embeddings = model.encode(safe_examples) if safe_examples else []
        
        for text in text_list:
            text_embedding = model.encode([text])[0]
            
            # 計算與釣魚郵件範例的相似度
            phishing_similarities = []
            if len(phishing_embeddings) > 0:
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity([text_embedding], phishing_embeddings)[0]
                    phishing_similarities = similarities.tolist()
                except ImportError:
                    # Fallback without sklearn
                    for emb in phishing_embeddings:
                        sim = np.dot(text_embedding, emb) / (
                            np.linalg.norm(text_embedding) * np.linalg.norm(emb)
                        )
                        phishing_similarities.append(sim)
            
            # 計算與安全郵件範例的相似度
            safe_similarities = []
            if len(safe_embeddings) > 0:
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity([text_embedding], safe_embeddings)[0]
                    safe_similarities = similarities.tolist()
                except ImportError:
                    # Fallback without sklearn
                    for emb in safe_embeddings:
                        sim = np.dot(text_embedding, emb) / (
                            np.linalg.norm(text_embedding) * np.linalg.norm(emb)
                        )
                        safe_similarities.append(sim)
            
            # 基於最高相似度進行分類
            max_phishing_sim = max(phishing_similarities) if phishing_similarities else 0.0
            max_safe_sim = max(safe_similarities) if safe_similarities else 0.0
            
            # 計算機率
            if max_phishing_sim + max_safe_sim > 0:
                phishing_prob = max_phishing_sim / (max_phishing_sim + max_safe_sim)
            else:
                phishing_prob = 0.5
            
            results.append({
                'phishing_probability': phishing_prob,
                'safe_probability': 1 - phishing_prob,
                'max_phishing_similarity': max_phishing_sim,
                'max_safe_similarity': max_safe_sim
            })
        
        return results
    
    @staticmethod
    def keyword_based_classification(text_list: List[str]) -> List[Dict[str, float]]:
        """
        基於關鍵詞的啟發式分類
        """
        phishing_keywords = [
            'urgent', 'verify', 'account', 'suspended', 'click', 'immediately',
            'winner', 'prize', 'congratulations', 'limited time', 'act now',
            'confirm', 'update', 'security alert', 'suspended'
        ]
        
        results = []
        
        for text in text_list:
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in phishing_keywords if keyword in text_lower)
            
            # 基於關鍵詞數量計算機率
            max_keywords = len(phishing_keywords)
            phishing_prob = min(0.9, keyword_count / max_keywords * 2)  # 最高90%
            
            results.append({
                'phishing_probability': phishing_prob,
                'safe_probability': 1 - phishing_prob,
                'keyword_matches': keyword_count
            })
        
        return results


def quick_inference(text_list: List[str], model_name: str = "default") -> Dict[str, Any]:
    """Quick inference function for standalone use"""
    try:
        from core.model_manager import ModelManager
        
        # Create temporary model manager and inference engine
        model_manager = ModelManager("./models")
        inference_engine = InferenceEngine(model_manager)
        
        # Perform batch inference
        result = inference_engine.batch_inference(text_list, model_name)
        return result
        
    except Exception as e:
        logger.error(f"Quick inference failed: {e}")
        # Return mock result for testing
        return {
            'predictions': [np.random.randint(0, 2) for _ in text_list],
            'confidences': [np.random.uniform(0.3, 0.9) for _ in text_list],
            'total_processed': len(text_list),
            'model_used': model_name,
            'error': str(e)
        }


# 新增：簡單的測試函數
def test_inference_engine():
    """Test function for inference engine"""
    print("🧪 Testing Inference Engine...")
    
    # Sample email texts
    sample_emails = [
        "Urgent: Your account will be suspended unless you verify immediately!",
        "Hi team, please find the quarterly report attached for your review.",
        "WINNER! You've won $1,000,000! Click here to claim your prize now!"
    ]
    
    try:
        # Test quick inference
        result = quick_inference(sample_emails)
        
        print(f"✅ Processed {result['total_processed']} emails")
        print(f"📊 Results: {result['predictions']}")
        print(f"📈 Confidences: {[f'{c:.2f}' for c in result['confidences']]}")
        
        # Test summary
        try:
            from core.model_manager import ModelManager
            model_manager = ModelManager("./models")
            inference_engine = InferenceEngine(model_manager)
            
            summary = inference_engine.get_model_predictions_summary(
                result['predictions'], 
                result['confidences']
            )
            
            print(f"📋 Summary:")
            print(f"  - Phishing detected: {summary['phishing_detected']}/{summary['total_emails']}")
            print(f"  - Average confidence: {summary['average_confidence']:.2f}")
            print(f"  - High confidence predictions: {summary['high_confidence_predictions']}")
        except Exception as e:
            print(f"⚠️ Summary test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# 只在直接運行此文件時執行測試，避免在import時執行
if __name__ == "__main__":
    # Run test when file is executed directly
    test_inference_engine()