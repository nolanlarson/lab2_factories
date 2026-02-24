from dataclasses import dataclass
from typing import Optional

@dataclass
class Email:
    """Dataclass representing an email with subject and body"""
    subject: str
    body: str
    ground_truth: Optional[str] = None