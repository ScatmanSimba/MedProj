"""Temporal utility functions for response attribution system."""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class TemporalInfo:
    """Container for temporal information about a medication."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration: Optional[str] = None
    confidence: float = 1.0
    status: str = "unknown"
    conflicting_mentions: list = None
    
    def __post_init__(self):
        if self.conflicting_mentions is None:
            self.conflicting_mentions = []

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime object.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Datetime object or None if parsing fails
    """
    # Common date formats
    formats = [
        "%Y-%m-%d",  # 2024-03-15
        "%m/%d/%Y",  # 03/15/2024
        "%d/%m/%Y",  # 15/03/2024
        "%B %d, %Y",  # March 15, 2024
        "%b %d, %Y",  # Mar 15, 2024
    ]
    
    # Try each format
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    # Try relative dates
    relative_patterns = {
        r"(\d+)\s+days?\s+ago": lambda x: datetime.now() - timedelta(days=int(x)),
        r"(\d+)\s+weeks?\s+ago": lambda x: datetime.now() - timedelta(weeks=int(x)),
        r"(\d+)\s+months?\s+ago": lambda x: datetime.now() - timedelta(days=int(x) * 30),
        r"(\d+)\s+years?\s+ago": lambda x: datetime.now() - timedelta(days=int(x) * 365),
    }
    
    for pattern, converter in relative_patterns.items():
        match = re.match(pattern, date_str.lower())
        if match:
            try:
                return converter(match.group(1))
            except ValueError:
                continue
    
    return None

def parse_duration(duration_str: str) -> Optional[Tuple[int, str]]:
    """Parse duration string to (amount, unit) tuple.
    
    Args:
        duration_str: Duration string (e.g., "2 weeks", "3 months")
        
    Returns:
        Tuple of (amount, unit) or None if parsing fails
    """
    try:
        # Split into amount and unit
        parts = duration_str.lower().split()
        if len(parts) != 2:
            return None
        
        amount = float(parts[0])
        unit = parts[1].rstrip('s')  # Remove plural
        
        # Validate unit
        if unit not in ['day', 'week', 'month', 'year']:
            return None
        
        return (int(amount), unit)
    except (ValueError, IndexError):
        return None

def calculate_recency_weight(temporal_info: TemporalInfo, 
                           reference_date: Optional[datetime] = None,
                           weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate recency weight from temporal information.
    
    Args:
        temporal_info: Temporal information about medication
        reference_date: Reference date for calculations (defaults to now)
        weights: Optional custom weights for different time periods
        
    Returns:
        Recency weight between 0 and 1
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'current': 1.0,    # Within a week
            'recent': 0.8,     # Within a month
            'past': 0.5,       # Within 6 months
            'distant': 0.2     # More than 6 months
        }
    
    # Calculate days from reference date
    days = None
    
    if temporal_info.start_date:
        days = (reference_date - temporal_info.start_date).days
    elif temporal_info.end_date:
        days = (reference_date - temporal_info.end_date).days
    elif temporal_info.duration:
        parsed = parse_duration(temporal_info.duration)
        if parsed:
            amount, unit = parsed
            if unit == 'day':
                days = amount
            elif unit == 'week':
                days = amount * 7
            elif unit == 'month':
                days = amount * 30
            elif unit == 'year':
                days = amount * 365
    
    if days is None:
        return 0.0
    
    # Calculate base weight from days
    if days <= 7:
        base_weight = weights['current']
    elif days <= 30:
        base_weight = weights['recent']
    elif days <= 180:
        base_weight = weights['past']
    else:
        base_weight = weights['distant']
    
    # Apply confidence adjustment
    confidence = temporal_info.confidence
    
    # Reduce confidence if there are conflicting mentions
    if temporal_info.conflicting_mentions:
        confidence *= 0.8 ** len(temporal_info.conflicting_mentions)
    
    # Calculate final weight
    final_weight = base_weight * confidence
    
    # Apply status-specific adjustments
    if temporal_info.status == "current":
        final_weight *= 1.2
    elif temporal_info.status == "past":
        final_weight *= 0.8
    
    return max(0.0, min(1.0, final_weight))

def process_temporal_mentions(mentions: list) -> Dict[str, TemporalInfo]:
    """Process multiple temporal mentions for a medication.
    
    Args:
        mentions: List of temporal mention dictionaries
        
    Returns:
        Dictionary mapping medications to processed temporal info
    """
    processed = {}
    
    for mention in mentions:
        med = mention['medication']
        
        # Initialize or get existing info
        if med not in processed:
            processed[med] = TemporalInfo()
        
        # Update with new information
        if 'start_date' in mention:
            date = parse_date(mention['start_date'])
            if date:
                processed[med].start_date = date
        
        if 'end_date' in mention:
            date = parse_date(mention['end_date'])
            if date:
                processed[med].end_date = date
        
        if 'duration' in mention:
            processed[med].duration = mention['duration']
        
        if 'confidence' in mention:
            processed[med].confidence = mention['confidence']
        
        if 'status' in mention:
            processed[med].status = mention['status']
        
        # Track conflicting mentions
        if processed[med].status != mention.get('status', 'unknown'):
            processed[med].conflicting_mentions.append(mention)
    
    return processed 