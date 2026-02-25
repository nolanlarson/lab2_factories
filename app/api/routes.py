from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email

import json
import os

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    confidence_score: float
    classifier_winner: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]
    most_similar_email: Optional[Dict[str, Any]] = None
    topic_similarity_score: Optional[float] = None
    email_similarity_score: Optional[float] = None

class EmailAddResponse(BaseModel):
    message: str
    email_id: int
    
class EmailAddTopic(BaseModel):
    topic: str
    description: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    topic: Optional[str] = None
    ground_truth: Optional[bool] = None

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            confidence_score=result["confidence_score"],
            classifier_winner=result["classifier_winner"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"],
            most_similar_email=result["most_similar_email"],
            topic_similarity_score=result["topic_similarity_score"],
            email_similarity_score=result["email_similarity_score"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

### Adding emails store endpoint.

@router.post("/emails/store", response_model=EmailAddResponse)
async def store_email(request: EmailStoreRequest):
    """Store an email with optional ground truth label"""
    try:
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
        
        # Load existing emails
        with open(data_file, "r") as f:
            emails = json.load(f)
        
        # Create new email entry with auto-incrementing ID
        email_id = max([e.get("id", 0) for e in emails], default=0) + 1
        
        email_entry = {
            "id": email_id,
            "subject": request.subject,
            "body": request.body,
            "topic": request.topic,
            "ground_truth": request.ground_truth
        }
        
        emails.append(email_entry)
        
        # Save updated emails
        with open(data_file, "w") as f:
            json.dump(emails, f, indent=2)
        
        return EmailAddResponse(
            message="Email stored successfully",
            email_id=email_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
### Adding possiblity to post topics    
@router.post("/topics")
async def add_topic(request: EmailAddTopic):
    """Add a new email topic with description"""
    try:
        
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        
        with open(data_file, "r") as f:
            topics = json.load(f)

        
        if request.topic in topics:
            raise HTTPException(status_code=400, detail="Topic already exists")

        
        topics[request.topic] = {
            "description": request.description
        }

    
        with open(data_file, "w") as f:
            json.dump(topics, f, indent=2)

        return {
            "message": "Topic added successfully",
            "topic": request.topic
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

