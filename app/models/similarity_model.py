import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

class EmailClassifierModel:
    """Email classifier model using embedding similarity"""

    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())

        # Load sentence transformer model (same model as feature generator)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-compute embeddings for all topic descriptions
        self.topic_embeddings = self._compute_topic_embeddings()
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)

    def _compute_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all topic descriptions"""
        topic_embeddings = {}
        for topic, data in self.topic_data.items():
            description = data['description']
            embedding = self.model.encode(description, convert_to_numpy=True)
            topic_embeddings[topic] = embedding
        return topic_embeddings
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature similarity"""
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        return max(scores, key=scores.get)
    
    def predict_with_score(self, features: Dict[str, Any]) -> tuple[str, float]:
        """Classify email into one of the topics and return both topic and its score
        
        Returns:
            Tuple of (predicted_topic, score)
        """
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        predicted_topic = max(scores, key=scores.get)
        return predicted_topic, float(scores[predicted_topic])
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate cosine similarity between email and topic embeddings"""
        # Get email embedding from features (now a list/array)
        email_embedding = features.get("email_embeddings_average_embedding", None)

        if email_embedding is None:
            return 0.0

        # Convert to numpy array if it's a list
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        # Get pre-computed topic embedding
        topic_embedding = self.topic_embeddings[topic]

        # Calculate cosine similarity
        # cosine_similarity = dot(A, B) / (||A|| * ||B||)
        dot_product = np.dot(email_embedding, topic_embedding)
        email_norm = np.linalg.norm(email_embedding)
        topic_norm = np.linalg.norm(topic_embedding)

        if email_norm == 0 or topic_norm == 0:
            return 0.0

        cosine_similarity = dot_product / (email_norm * topic_norm)

        # Cosine similarity is between -1 and 1, but for text it's usually positive
        # Normalize to 0-1 range for better interpretability
        normalized_score = (cosine_similarity + 1) / 2

        return float(normalized_score)
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}
    
    def compare_classifiers(self, predicted_topic: str, topic_score: float, similar_email: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare topic classification score with email body similarity score and return the higher confidence result
        
        Args:
            predicted_topic: The predicted topic from topic classifier
            topic_score: The similarity score for the predicted topic
            similar_email: The most similar stored email dict (or None if below threshold)
            
        Returns:
            Dictionary indicating which classifier won and the result details
        """
        if similar_email is None:
            # No similar email found, use topic classification
            return {
                "winner": "topic",
                "predicted_topic": predicted_topic,
                "confidence_score": float(topic_score),
                "similar_email": None
            }
        
        if not similar_email.get("ground_truth", False):
            # Similar email is not verified/ground truth, rely on topic classification
            return {
                "winner": "topic",
                "predicted_topic": predicted_topic,
                "confidence_score": float(topic_score),
                "similar_email": None
            }
        
        email_similarity_score = similar_email["similarity_score"]
        similar_email_topic = similar_email.get("topic")
        
        if email_similarity_score > topic_score:
            # Email similarity is higher and has ground truth, use the similar email's topic if available
            final_topic = similar_email_topic if similar_email_topic else predicted_topic
            return {
                "winner": "email",
                "predicted_topic": final_topic,
                "confidence_score": float(email_similarity_score),
                "similar_email": similar_email,
                "topic_score": float(topic_score)
            }
        else:
            # Topic classification is higher or equal, use topic classification
            return {
                "winner": "topic",
                "predicted_topic": predicted_topic,
                "confidence_score": float(topic_score),
                "similar_email": similar_email,
                "email_similarity_score": float(email_similarity_score)
            }
    
    def find_most_similar_email(self, email_embedding: np.ndarray, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Find the most similar email from stored emails above a threshold using embedding similarity
        
        Args:
            email_embedding: The embedding of the email to classify
            threshold: Minimum similarity score (0-1) to consider an email as similar for ground truth
        """
        try:
            # Load stored emails
            data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
            with open(data_file, 'r') as f:
                stored_emails = json.load(f)
            
            if not stored_emails:
                return None
            
            max_similarity = -1
            most_similar_email = None
            
            # Compute embeddings for stored emails and find max similarity
            for stored_email in stored_emails:
                # Create combined text from subject and body
                combined_text = f"{stored_email['subject']} {stored_email['body']}"
                stored_embedding = self.model.encode(combined_text, convert_to_numpy=True)
                
                # Calculate cosine similarity
                similarity = self._compute_cosine_similarity(email_embedding, stored_embedding)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_email = stored_email
            
            # Return if above threshold
            if max_similarity >= threshold and most_similar_email is not None:
                return {
                    "id": most_similar_email["id"],
                    "subject": most_similar_email["subject"],
                    "body": most_similar_email["body"],
                    "ground_truth": most_similar_email.get("ground_truth"),
                    "similarity_score": float(max_similarity)
                }
            
            return None
            
        except Exception as e:
            # If there's an error loading/processing emails, return None gracefully
            return None
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))