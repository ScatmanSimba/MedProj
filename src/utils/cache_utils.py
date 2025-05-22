"""Enhanced cache utilities for the medication response prediction system.

This module provides advanced caching implementations with features like:
- Hit/miss statistics tracking
- Cache warming
- Compression for large items
- Sophisticated eviction policies
"""

from typing import Any, Dict, Optional, Tuple, List
from collections import OrderedDict
import threading
import zlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache statistics container."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compressions: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compressions = 0
        self.last_reset = datetime.now()

class EnhancedLRUCache:
    """Enhanced LRU cache with statistics, compression, and frequency-based eviction."""
    
    def __init__(
        self,
        maxsize: int = 1000,
        compression_threshold: int = 1024,  # 1KB
        warmup_size: int = 100,
        eviction_policy: str = "lru_freq"
    ):
        """Initialize the enhanced LRU cache.
        
        Args:
            maxsize: Maximum number of items in cache
            compression_threshold: Size in bytes above which items are compressed
            warmup_size: Number of items to keep in warmup cache
            eviction_policy: Cache eviction policy ('lru' or 'lru_freq')
        """
        self.maxsize = maxsize
        self.compression_threshold = compression_threshold
        self.warmup_size = warmup_size
        self.eviction_policy = eviction_policy
        
        # Main cache storage
        self._cache: OrderedDict[str, Any] = OrderedDict()
        
        # Frequency tracking
        self._freq: Dict[str, int] = {}
        
        # Warmup cache for frequently accessed items
        self._warmup: OrderedDict[str, Any] = OrderedDict()
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"Initialized EnhancedLRUCache with maxsize={maxsize}, "
            f"compression_threshold={compression_threshold}, "
            f"warmup_size={warmup_size}, eviction_policy={eviction_policy}"
        )
    
    def _compress(self, value: Any) -> Tuple[bytes, bool]:
        """Compress value if it exceeds threshold.
        
        Args:
            value: Value to potentially compress
            
        Returns:
            Tuple of (compressed value, was_compressed)
        """
        try:
            # Convert to JSON string
            json_str = json.dumps(value)
            
            # Check if compression needed
            if len(json_str.encode()) <= self.compression_threshold:
                return value, False
            
            # Compress
            compressed = zlib.compress(json_str.encode())
            self.stats.compressions += 1
            return compressed, True
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value, False
    
    def _decompress(self, value: Any) -> Any:
        """Decompress value if it's compressed.
        
        Args:
            value: Potentially compressed value
            
        Returns:
            Decompressed value
        """
        try:
            if isinstance(value, bytes):
                decompressed = zlib.decompress(value)
                return json.loads(decompressed.decode())
            return value
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return value
    
    def _update_frequency(self, key: str) -> None:
        """Update access frequency for key.
        
        Args:
            key: Cache key
        """
        self._freq[key] = self._freq.get(key, 0) + 1
        
        # Update warmup cache if frequency is high
        if self._freq[key] >= 3 and key not in self._warmup:
            if len(self._warmup) >= self.warmup_size:
                self._warmup.popitem(last=False)
            self._warmup[key] = self._cache[key]
    
    def _evict(self) -> None:
        """Evict least valuable item based on policy."""
        if self.eviction_policy == "lru_freq":
            # Find least valuable item based on frequency and recency
            min_score = float('inf')
            min_key = None
            
            for key in self._cache:
                # Score combines frequency and recency
                freq = self._freq.get(key, 0)
                recency = list(self._cache.keys()).index(key)
                score = recency / (freq + 1)  # Add 1 to avoid division by zero
                
                if score < min_score:
                    min_score = score
                    min_key = key
            
            if min_key:
                del self._cache[min_key]
                del self._freq[min_key]
                self.stats.evictions += 1
        else:
            # Simple LRU eviction
            key, _ = self._cache.popitem(last=False)
            del self._freq[key]
            self.stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            # Check warmup cache first
            if key in self._warmup:
                self.stats.hits += 1
                self._update_frequency(key)
                return self._decompress(self._warmup[key])
            
            # Check main cache
            if key in self._cache:
                self.stats.hits += 1
                value = self._cache.pop(key)
                self._cache[key] = value  # Move to end (most recent)
                self._update_frequency(key)
                return self._decompress(value)
            
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Compress if needed
            value, was_compressed = self._compress(value)
            
            # Evict if needed
            if len(self._cache) >= self.maxsize:
                self._evict()
            
            # Store in cache
            self._cache[key] = value
            self._freq[key] = self._freq.get(key, 0)
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._warmup.clear()
            self._freq.clear()
            self.stats.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                'size': len(self._cache),
                'maxsize': self.maxsize,
                'warmup_size': len(self._warmup),
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': self.stats.hit_rate(),
                'evictions': self.stats.evictions,
                'compressions': self.stats.compressions,
                'last_reset': self.stats.last_reset.isoformat()
            }
    
    def warmup(self, items: List[Tuple[str, Any]]) -> None:
        """Warm up cache with frequently used items.
        
        Args:
            items: List of (key, value) tuples to warm up
        """
        with self._lock:
            for key, value in items:
                if len(self._warmup) >= self.warmup_size:
                    break
                self._warmup[key] = value
                self._freq[key] = 3  # Start with high frequency 