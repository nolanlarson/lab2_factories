from typing import Dict, Any, Optional
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email
import numpy as np

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
    
    def classify_email(self, email: Email, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Classify an email into topics using generated features
        
        Args:
            email: Email to classify
            similarity_threshold: Minimum similarity score to find similar stored emails (0-1)
        """
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        # Step 2: Classify using features and get topic score
        predicted_topic, topic_score = self.model.predict_with_score(features)
        topic_scores = self.model.get_topic_scores(features)
        
        # Step 3: Find most similar email from stored emails
        email_embedding = features.get("email_embeddings_average_embedding", None)
        similar_email = None
        if email_embedding is not None:
            if isinstance(email_embedding, list):
                email_embedding = np.array(email_embedding)
            similar_email = self.model.find_most_similar_email(email_embedding, similarity_threshold)
        
        # Step 4: Compare classifiers and get the winner
        comparison_result = self.model.compare_classifiers(predicted_topic, topic_score, similar_email)
        
        # Return comprehensive results
        return {
            "predicted_topic": comparison_result["predicted_topic"],
            "confidence_score": comparison_result["confidence_score"],
            "classifier_winner": comparison_result["winner"],
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email,
            "most_similar_email": comparison_result.get("similar_email"),
            "topic_similarity_score": topic_score if comparison_result["winner"] == "email" else None,
            "email_similarity_score": comparison_result.get("email_similarity_score") if comparison_result["winner"] == "topic" else None
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }