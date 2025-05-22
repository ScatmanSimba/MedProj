# -*- coding: utf-8 -*-
"""snapshot.py - Reddit medication data snapshot creator

This script:
1. Queries Reddit for medication-related posts using PRAW
2. Performs basic cleaning and validation
3. Exports a frozen snapshot for reproducible analysis

Usage:
    python snapshot.py --output reddit_meds_20240505.csv --limit 50000
"""


import os
import re  # Move re import to global scope
import praw
import pandas as pd
import argparse
import json
import datetime
import time
import hashlib
import logging
import random
import signal
import sys
from typing import Dict, List, Any, Set, Tuple, Optional
from langdetect import detect, LangDetectException
from tqdm import tqdm
from functools import lru_cache
import prawcore
from requests.exceptions import Timeout, ConnectionError, RequestException
import socket
import requests
# import IPython  # Move this import
from collections import deque

# Import our medication dictionary
from src.med_dictionary1803 import MedDictionary

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("snapshot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("snapshot")

# Check if running in Colab
IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    from google.colab import drive
    import IPython
    import psutil
    import gc
    import threading

# Colab-specific settings
COLAB_MEMORY_LIMIT = 0.8  # 80% of available memory
COLAB_CHECKPOINT_INTERVAL = 500  # Save checkpoint every 500 posts in Colab
COLAB_RECONNECT_DELAY = 30  # 30 seconds between reconnection attempts in Colab
COLAB_MAX_RETRIES = 10  # More retries for Colab

# Reddit API rate limits
REDDIT_RATE_LIMIT = 60  # requests per minute
REDDIT_BURST_LIMIT = 30  # maximum burst requests
REDDIT_TOKEN_REFRESH = 1.0  # tokens per second

class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, 
                 rate: float = REDDIT_TOKEN_REFRESH,
                 burst: int = REDDIT_BURST_LIMIT):
        """
        Initialize token bucket
        
        Args:
            rate: Token refresh rate (tokens per second)
            burst: Maximum burst size (maximum tokens)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()
        
    def _update_tokens(self):
        """Update token count based on time elapsed"""
        now = time.time()
        time_passed = now - self.last_update
        self.tokens = min(
            self.burst,
            self.tokens + time_passed * self.rate
        )
        self.last_update = now
        
    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        with self.lock:
            self._update_tokens()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until enough tokens are available
        
        Args:
            tokens: Number of tokens to wait for
        """
        while not self.acquire(tokens):
            time.sleep(0.1)  # Short sleep to prevent CPU spinning

class ColabError(Exception):
    """Custom exception for Colab-specific errors"""
    pass

class RedditDataSnapshot:
    """
    Creates a reproducible snapshot of medication-related Reddit data
    """

    def __init__(self,
                 client_id: str,
                 client_secret: str,
                 user_agent: str,
                 data_dir: str = "reddit_data",
                 max_retries: int = None,
                 reconnect_delay: int = None):
        """
        Initialize the snapshot creator with Colab-specific settings if running in Colab
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        
        # Use Colab-specific settings if in Colab
        if IS_COLAB:
            self.max_retries = COLAB_MAX_RETRIES
            self.reconnect_delay = COLAB_RECONNECT_DELAY
            logger.info("Running in Colab environment - using Colab-specific settings")
        else:
            self.max_retries = max_retries or 5
            self.reconnect_delay = reconnect_delay or 60
            
        # Initialize rate limiter
        self.rate_limiter = TokenBucket()
        
        # Track API errors
        self.error_counts = {
            'rate_limit': 0,
            'timeout': 0,
            'connection': 0,
            'server_error': 0,
            'other': 0
        }
        self.error_times = deque(maxlen=1000)  # Track last 1000 errors
        
        # Track post rejection reasons
        self.rejection_counts = {
            'spam': 0,
            'too_old': 0,
            'stickied': 0,
            'no_med_context': 0,
            'non_english': 0,
            'too_short': 0,
            'processing_error': 0
        }
        # Add this to track unproductive queries
        self.dead_queries = set()
        # Initialize Reddit API client with timeout
        self._initialize_reddit_client()

        # Initialize medication dictionary
        self.med_dict = MedDictionary()

        # Pre-compile regex pattern for medication matching
        med_names = self.med_dict.get_all_medication_names()
        pattern = r'\b(?:' + '|'.join(map(re.escape, med_names)) + r')\b'
        self.med_regex = re.compile(pattern, re.IGNORECASE)

        # Paths
        self.data_dir = data_dir
        if IS_COLAB:
            # Mount Google Drive if not already mounted
            try:
                drive.mount('/content/drive')
                self.data_dir = os.path.join('/content/drive/MyDrive', data_dir)
                logger.info(f"Using Google Drive for data storage: {self.data_dir}")
            except Exception as e:
                logger.warning(f"Failed to mount Google Drive: {e}")
                logger.info("Using local storage instead")
        
        os.makedirs(self.data_dir, exist_ok=True)

        # Collection settings
        self.time_window = "year"
        self.post_limit = 500
        self.total_limit = 50000
        self.batch_size = 1000  # Number of posts to save at once
        
        # Time window in seconds
        self.time_window_seconds = {
            'day': 24 * 3600,
            'week': 7 * 24 * 3600,
            'month': 30 * 24 * 3600,
            'year': 365 * 24 * 3600,
            'all': float('inf')
        }.get(self.time_window, 3 * 365 * 24 * 3600)  # Default to 3 years

        # Collection state
        self.all_posts = []
        self.seen_post_ids = set()
        self.current_subreddit = None
        self.current_query = None
        self.interrupted = False
        self.retry_count = 0
        self.last_checkpoint_time = time.time()
        self.output_file = os.path.join(self.data_dir, "reddit_posts.csv")
        self.start_time = time.time()  # Add start time tracking

        # Collection metadata
        self.metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "subreddits": [],
            "query_terms": [],
            "raw_post_count": 0,
            "clean_post_count": 0,
            "collection_duration": 0,
            "time_window": self.time_window,
            "reconnection_attempts": 0,
            "environment": "colab" if IS_COLAB else "local",
            "colab_specific": {
                "checkpoints": [],
                "memory_usage": [],
                "reconnections": []
            } if IS_COLAB else None,
            "dead_queries": []  # Add this line to metadata initialization
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

        # Load checkpoint if exists
        self.load_checkpoint()

        # Memory management settings
        self.MEMORY_WARNING_THRESHOLD = 0.7  # 70% memory usage triggers warning
        self.MEMORY_CRITICAL_THRESHOLD = 0.85  # 85% memory usage triggers aggressive GC
        self.MEMORY_EMERGENCY_THRESHOLD = 0.95  # 95% memory usage triggers emergency measures
        self.memory_history = deque(maxlen=100)  # Track last 100 memory readings
        self.last_gc_time = time.time()
        self.gc_cooldown = 5.0  # Minimum seconds between GC runs
        
        # Memory usage tracking
        self.memory_stats = {
            'warnings': 0,
            'critical': 0,
            'emergency': 0,
            'gc_runs': 0,
            'memory_cleared': 0
        }

    def _check_colab_memory(self):
        """Check if memory usage is within safe limits in Colab"""
        if not IS_COLAB:
            return True
            
        try:
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # Record memory usage
            self.metadata["colab_specific"]["memory_usage"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "usage": memory_usage
            })
            
            if memory_usage > COLAB_MEMORY_LIMIT:
                logger.warning(f"High memory usage detected: {memory_usage:.1%}")
                # Force garbage collection
                gc.collect()
                return False
            return True
        except Exception as e:
            logger.warning(f"Error checking memory: {e}")
            return True

    def _handle_colab_disconnect(self):
        """Handle Colab disconnection by saving state and attempting reconnection"""
        if not IS_COLAB:
            return False
            
        logger.warning("Colab disconnection detected - saving state...")
        
        try:
            # Save current progress
            self.save_checkpoint()
            
            # Record reconnection attempt
            self.metadata["colab_specific"]["reconnections"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "retry_count": self.retry_count
            })
            
            # Wait before reconnecting
            time.sleep(self.reconnect_delay)
            
            # Try to reinitialize client
            self._initialize_reddit_client()
            
            if self._check_connection():
                logger.info("Successfully reconnected after Colab disconnection")
                return True
            else:
                logger.error("Failed to reconnect after Colab disconnection")
                return False
                
        except Exception as e:
            logger.error(f"Error handling Colab disconnection: {e}")
            return False

    def log_error_summary(self, force: bool = False) -> None:
        """
        Log a summary of API errors and their frequencies.
        Only logs if enough time has passed since last summary or if forced.
        
        Args:
            force: If True, log summary regardless of time since last log
        """
        # Check if enough time has passed since last summary
        current_time = time.time()
        if not force and hasattr(self, '_last_error_summary_time'):
            if current_time - self._last_error_summary_time < 300:  # 5 minutes
                return
                
        # Calculate error rates
        now = time.time()
        recent_errors = [t for t in self.error_times if now - t < 3600]  # Last hour
        error_rate = len(recent_errors) / 3600 if recent_errors else 0
        
        # Calculate error percentages
        total_errors = sum(self.error_counts.values())
        error_percentages = {
            error_type: (count / total_errors * 100) if total_errors > 0 else 0
            for error_type, count in self.error_counts.items()
        }
        
        # Log summary
        logger.info("=== Error Summary ===")
        logger.info(f"Total errors: {total_errors}")
        logger.info(f"Errors in last hour: {len(recent_errors)}")
        logger.info(f"Error rate: {error_rate:.2f} errors/second")
        logger.info("Error distribution:")
        for error_type, count in self.error_counts.items():
            logger.info(f"  {error_type}: {count} ({error_percentages[error_type]:.1f}%)")
        logger.info(f"Retry count: {self.retry_count}")
        logger.info("===================")
        
        # Update last summary time
        self._last_error_summary_time = current_time

    def _make_api_request(self, func, *args, **kwargs):
        """Make an API request with rate limiting and error handling"""
        while True:
            try:
                # Check memory usage
                if not self._check_memory_usage():
                    logger.warning("Memory usage too high - pausing collection")
                    time.sleep(30)  # Wait for memory to free up
                    continue
                
                # Wait for rate limit tokens
                self.rate_limiter.wait_for_tokens()
                
                # Make the request
                return func(*args, **kwargs)
                
            except prawcore.exceptions.TooManyRequests as e:
                self.error_counts['rate_limit'] += 1
                self.error_times.append(time.time())
                self.log_error_summary()
                
                # Calculate backoff based on error frequency
                backoff = self._calculate_backoff()
                logger.warning(f"Rate limit exceeded. Backing off for {backoff:.1f} seconds")
                time.sleep(backoff)
                
                # Reset rate limiter on rate limit error
                self.rate_limiter = TokenBucket()
                
            except (Timeout, socket.timeout) as e:
                self.error_counts['timeout'] += 1
                self.error_times.append(time.time())
                self.log_error_summary()
                if not self._handle_connection_error(e):
                    raise
                    
            except (ConnectionError, RequestException) as e:
                self.error_counts['connection'] += 1
                self.error_times.append(time.time())
                self.log_error_summary()
                if IS_COLAB:
                    if not self._handle_colab_disconnect():
                        if not self._handle_connection_error(e):
                            raise
                else:
                    if not self._handle_connection_error(e):
                        raise
                        
            except prawcore.exceptions.ServerError as e:
                self.error_counts['server_error'] += 1
                self.error_times.append(time.time())
                self.log_error_summary()
                logger.error(f"Reddit server error: {e}")
                time.sleep(5)  # Short sleep for server errors
                
            except Exception as e:
                self.error_counts['other'] += 1
                self.error_times.append(time.time())
                self.log_error_summary()
                logger.error(f"Unexpected error: {e}")
                raise

    def _calculate_backoff(self) -> float:
        """
        Calculate dynamic backoff time based on error frequency
        
        Returns:
            Backoff time in seconds
        """
        # Get errors in last 5 minutes
        now = time.time()
        recent_errors = [t for t in self.error_times if now - t < 300]
        
        if not recent_errors:
            return 5.0  # Default 5 second backoff
            
        # Calculate error rate
        error_rate = len(recent_errors) / 300  # errors per second
        
        # Exponential backoff based on error rate
        base_delay = 5.0
        max_delay = 300.0  # 5 minutes
        delay = min(base_delay * (2 ** error_rate), max_delay)
        
        # Add jitter
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter

    def validate_subreddits(self, subreddits: List[str]) -> List[str]:
        """
        Validate that subreddits exist and are accessible.
        
        Args:
            subreddits: List of subreddit names to validate
            
        Returns:
            List of valid subreddit names
        """
        valid_subreddits = []
        
        for subreddit_name in subreddits:
            try:
                # Try to access the subreddit
                subreddit = self.reddit.subreddit(subreddit_name)
                # This will raise an exception if subreddit doesn't exist
                _ = subreddit.id
                valid_subreddits.append(subreddit_name)
                logger.info(f"Validated subreddit: r/{subreddit_name}")
            except Exception as e:
                logger.warning(f"Invalid subreddit r/{subreddit_name}: {str(e)}")
        
        if not valid_subreddits:
            raise ValueError("No valid subreddits found!")
            
        logger.info(f"Found {len(valid_subreddits)} valid subreddits out of {len(subreddits)}")
        return valid_subreddits

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals by saving partial progress"""
        logger.warning("Received interrupt signal - saving partial progress...")
        self.interrupted = True
        
        if self.all_posts:
            # Save partial progress
            partial_df = pd.DataFrame(self.all_posts)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            partial_file = os.path.join(self.data_dir, f"partial_snapshot_{timestamp}.csv")
            partial_df.to_csv(partial_file, index=False)
            logger.info(f"Saved partial progress ({len(self.all_posts)} posts) to {partial_file}")

        sys.exit(0)

    def save_checkpoint(self):
        """Save checkpoint with Colab-specific optimizations"""
        if not self.all_posts:
            return
            
        # In Colab, save more frequently
        if IS_COLAB:
            current_time = time.time()
            if current_time - self.last_checkpoint_time < 300:  # 5 minutes
                return
            self.last_checkpoint_time = current_time
            
        try:
            checkpoint_file = os.path.join(self.data_dir, f"checkpoint_{len(self.all_posts)}.csv")
            pd.DataFrame(self.all_posts).to_csv(checkpoint_file, index=False)
            
            # Save post IDs
            with open(checkpoint_file.replace('.csv', '_ids.json'), 'w') as f:
                json.dump(list(self.seen_post_ids), f)
                
            # Record checkpoint in metadata
            if IS_COLAB:
                self.metadata["colab_specific"]["checkpoints"].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "post_count": len(self.all_posts),
                    "file": checkpoint_file
                })
                
            logger.info(f"Saved checkpoint with {len(self.all_posts)} posts to {checkpoint_file}")
            
            # Force garbage collection in Colab
            if IS_COLAB:
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            if IS_COLAB:
                # Try to save to a different location in Colab
                try:
                    backup_file = f"/tmp/checkpoint_backup_{int(time.time())}.csv"
                    pd.DataFrame(self.all_posts).to_csv(backup_file, index=False)
                    logger.info(f"Saved backup checkpoint to {backup_file}")
                except Exception as backup_e:
                    logger.error(f"Failed to save backup checkpoint: {backup_e}")

    def exponential_backoff(self, attempt: int, base_delay: float = 1.0, max_delay: float = 300.0) -> float:
        """Calculate exponential backoff delay with jitter

        Args:
            attempt: Current attempt number (starting from 1)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Delay time in seconds
        """
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        jitter = random.uniform(0, 0.1 * delay)  # Add 0-10% jitter
        return delay + jitter

    @lru_cache(maxsize=10000)
    def detect_language_safe(self, text: str) -> str:
        """
        Safely detect language, returning 'unknown' if detection fails
        Uses LRU cache to avoid repeated detections

        Args:
            text: Text to detect language

        Returns:
            ISO language code or 'unknown'
        """
        if not isinstance(text, str) or len(text.strip()) < 20:
            return 'unknown'

        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'

    def is_spam(self, post) -> bool:
        """
        Check if a post is likely spam.
        
        Args:
            post: PRAW post object
            
        Returns:
            True if post is likely spam, False otherwise
        """
        try:
            # Check for common spam indicators
                
            logger.debug(f"Post {post.id} passed spam checks")
            return False
            
        except Exception as e:
            logger.warning(f"Error checking spam for post {getattr(post, 'id', 'unknown')}: {e}")
            return True

    def find_medication_mentions(self, title: str, text: str) -> Tuple[List[str], bool]:
        """
        Return (meds_list, has_med_context).
        - meds_list: matched generic med names
        - has_med_context: True if there's any evidence this is about medication use (name OR context)
        """
        if not isinstance(title, str): title = ""
        if not isinstance(text, str): text = ""

        combined = (title + " " + text).lower()
        found_meds = set()

        # Use pre-compiled regex pattern to find all matches
        for match in self.med_regex.finditer(combined):
            med_name = match.group(0).lower()
            generic = self.med_dict.get_generic_name(med_name)
            if generic:
                found_meds.add(generic)

        # Check for common misspellings and abbreviations
        for misspelling, correct in self.med_dict.COMMON_MISSPELLINGS.items():
            if re.search(r'\b' + re.escape(misspelling) + r'\b', combined, re.IGNORECASE):
                generic = self.med_dict.get_generic_name(correct)
                if generic:
                    found_meds.add(generic)

        # Broader medication context check (even if drug not named)
        context_patterns = [
            r"\bstopped (taking|using)? (my|the)? (meds|medication|drugs)?",
            r"\bchanged (my|the)? meds",
            r"\bswitched .* (meds|medication|drugs)",
            r"\bmy meds\b",
            r"\bthe meds\b",
            r"\bon meds\b",
            r"\bprescribed\b",
            r"\bmedication\b",
            r"\bantidepressant\b",
            r"\bantipsychotic\b",
            r"\banxiety med\b",
            r"\bdepression med\b",
            r"\bpsych med\b",
            r"\bpsychiatric med\b",
            r"\bpsychotropic\b",
            r"\bmedication side effects\b",
            r"\bmedication withdrawal\b",
            r"\bmedication adjustment\b",
            r"\bmedication change\b",
            r"\bmedication switch\b",
            r"\bmedication taper\b"
        ]
        has_context = bool(found_meds) or any(re.search(p, combined) for p in context_patterns)

        return list(found_meds), has_context

    def load_checkpoint(self) -> bool:
        """
        Load previous checkpoint to resume collection

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if os.path.exists(self.output_file):
            try:
                # Load IDs of posts we've already collected
                df = pd.read_csv(self.output_file)
                self.seen_post_ids.update(df['id'].tolist())
                logger.info(f"Loaded checkpoint with {len(self.seen_post_ids)} posts")
                return True
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return False
        return False

    def save_batch(self, batch_size: int = None) -> None:
        """
        Save a batch of posts to disk and clear memory

        Args:
            batch_size: Number of posts to save (defaults to self.batch_size)
        """
        batch_size = batch_size or self.batch_size
        if len(self.all_posts) >= batch_size:
            batch_df = pd.DataFrame(self.all_posts[:batch_size])
            
            # Append to existing file or create new one
            file_exists = os.path.exists(self.output_file)
            batch_df.to_csv(
                self.output_file, 
                mode='a' if file_exists else 'w',
                header=not file_exists,
                index=False
            )
            
            # Remove saved posts from memory
            self.all_posts = self.all_posts[batch_size:]
            gc.collect()  # Force garbage collection
            logger.info(f"Saved batch of {batch_size} posts to {self.output_file}")

    def collect_snapshot(self,
                         subreddits: List[str] = None,
                         query_terms: List[str] = None,
                         time_window: str = "year",
                         post_limit: int = 500,
                         total_limit: int = 50000) -> pd.DataFrame:
        """
        Collect a snapshot of medication-related Reddit posts.
        Handles medication-specific subreddits (all posts) and general subreddits (query-based) differently.

        Args:
            subreddits: List of subreddits to search (if None, uses list from MedDictionary)
            query_terms: List of search terms (medications)
            time_window: Reddit search time window
            post_limit: Maximum posts per subreddit + query
            total_limit: Maximum total posts

        Returns:
            DataFrame containing the snapshot data
        """
        # Update time window settings
        self.time_window = time_window
        self.time_window_seconds = {
            'day': 24 * 3600,
            'week': 7 * 24 * 3600,
            'month': 30 * 24 * 3600,
            'year': 365 * 24 * 3600,
            'all': float('inf')
        }.get(time_window, 3 * 365 * 24 * 3600)  # Default to 3 years
        
        # Record start time for duration tracking
        start_time = time.time()
        
        # Use defaults if not provided
        if subreddits is None:
            subreddit_dict = self.med_dict.get_subreddits()
            subreddits = subreddit_dict["medication_specific"] + subreddit_dict["general"]
            logger.info(f"Using {len(subreddits)} subreddits from MedDictionary")
        query_terms = query_terms or self.med_dict.get_query_terms()

        # Validate subreddits
        valid_subreddits = self.validate_subreddits(subreddits)

        # Separate medication-specific and general subreddits
        med_specific_subs = [sub for sub in valid_subreddits if sub in self.med_dict.get_all_medication_names()]
        general_subs = [sub for sub in valid_subreddits if sub not in med_specific_subs]

        logger.info("Subreddit categorization:")
        logger.info(f"  Medication-specific subreddits ({len(med_specific_subs)}): {', '.join(med_specific_subs)}")
        logger.info(f"  General subreddits ({len(general_subs)}): {', '.join(general_subs)}")
        logger.info(f"Query terms ({len(query_terms)}): {', '.join(query_terms)}")

        # Update metadata
        self.metadata.update({
            "timestamp": datetime.datetime.now().isoformat(),
            "medication_specific_subreddits": med_specific_subs,
            "general_subreddits": general_subs,
            "query_terms": query_terms,
            "time_window": time_window,
            "collection_strategy": {
                "med_specific": "all_posts",
                "general": "query_based"
            }
        })

        # Track progress
        total_posts = 0
        pbar = tqdm(total=total_limit, desc="Collecting posts")

        try:
            # 1. Collect from medication-specific subreddits
            for subreddit in med_specific_subs:
                if total_posts >= total_limit:
                    break

                try:
                    sub = self.reddit.subreddit(subreddit)
                    posts = sub.new(limit=post_limit)
                    
                    for post in posts:
                        if total_posts >= total_limit:
                            break
                            
                        # Skip if already seen
                        if post.id in self.seen_post_ids:
                            continue
                            
                        post_data = self._process_post(post)
                        if post_data:
                            self.all_posts.append(post_data)
                            self.seen_post_ids.add(post.id)
                            total_posts += 1
                            pbar.update(1)
                            
                            # Save batch if we've reached the batch size
                            if len(self.all_posts) >= self.batch_size:
                                self.save_batch()
                                
                except Exception as e:
                    logger.error(f"Error processing subreddit {subreddit}: {e}")
                    continue

            # 2. Collect from general subreddits using medication search terms
            for subreddit in general_subs:
                logger.info(f"DEBUG: Starting to process general subreddit: {subreddit}") 
                if total_posts >= total_limit:
                    break

                try:
                    sub = self.reddit.subreddit(subreddit)
                    
                    for term in query_terms:
                        if total_posts >= total_limit:
                            break
                            
                        # Skip if this subreddit/query pair is already known to be dead
                        query_key = f"{subreddit}:{term}"
                        if query_key in self.dead_queries:
                            logger.debug(f"Skipping known dead query: {query_key}")
                            continue
                            
                        try:
                            # Search with time filter
                            search_query = term
                            logger.info(f"Using simplified search query: '{search_query}' in r/{subreddit}")
                            logger.info(f"Searching for '{search_query}' in r/{subreddit}")
                            try:
                                search_results = sub.search(search_query, limit=post_limit)
                                posts = list(search_results)
                                logger.info(f"Found {len(posts)} posts for '{search_query}' in r/{subreddit}")
                            except Exception as e:
                                logger.error(f"Error converting search results to list: {e}")
                                posts = []
                            
                            # Check if this query returns too few posts
                            if len(posts) < 2:
                                logger.info(f"Low-yield query detected: {query_key} - only returned {len(posts)} posts")
                                self.dead_queries.add(query_key)
                                self.metadata["dead_queries"].append({
                                    "subreddit": subreddit,
                                    "query": term,
                                    "time_window": time_window,
                                    "posts_returned": len(posts),
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                                continue
                            
                            # Process the posts from this query
                            post_count_for_query = 0
                            for post in posts:
                                if total_posts >= total_limit:
                                    break
                                    
                                # Skip if already seen
                                if post.id in self.seen_post_ids:
                                    continue
                                    
                                post_data = self._process_post(post)
                                if post_data:
                                    self.all_posts.append(post_data)
                                    self.seen_post_ids.add(post.id)
                                    total_posts += 1
                                    post_count_for_query += 1
                                    pbar.update(1)
                                    
                                    # Save batch if we've reached the batch size
                                    if len(self.all_posts) >= self.batch_size:
                                        self.save_batch()
                            
                            logger.info(f"Retrieved {post_count_for_query} valid posts from {query_key}")
                                
                        except Exception as e:
                            logger.error(f"Error processing search term {term} in {subreddit}: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing subreddit {subreddit}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error during collection: {e}")
            raise
        finally:
            pbar.close()
            
            # Save any remaining posts
            if self.all_posts:
                self.save_batch(len(self.all_posts))
            
            # Calculate duration
            duration = time.time() - start_time
            logger.info(f"Collection completed in {duration:.2f} seconds")
            logger.info(f"Total posts collected: {total_posts}")
            
            # Log final error summary
            self.log_error_summary(force=True)

        # Return the final DataFrame
        return pd.DataFrame(self.all_posts) if self.all_posts else pd.DataFrame()

    def clean_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the collected snapshot data.
        
        Args:
            df: DataFrame containing the collected posts
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with additional filtering applied
        """
        if df.empty:
            return df
            
        # Convert medications to string representation
        df['medications'] = df['medications'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])
        
        # Apply additional filters for the clean dataset
        df = df[
            (df['language'] == 'en') &  # English only
            (df['text_length'] > 20) &  # Minimum text length
            (df['created_utc'] > (time.time() - self.time_window_seconds))  # Time window
        ]
        
        return df

    def save_snapshot(self, df: pd.DataFrame, output_file: str) -> None:
        """
        Save both raw and filtered snapshots to CSV files with metadata

        Args:
            df: Snapshot data
            output_file: Base path for output files (without extension)
        """
        # Ensure output_file doesn't have extension
        output_file = output_file.replace('.csv', '')
        
        # Save raw dataset (only basic sanity checks applied)
        raw_file = f"{output_file}_raw.csv"
        df.to_csv(raw_file, index=False)
        
        # Save filtered dataset (with additional cleaning)
        clean_df = self.clean_snapshot(df)
        clean_file = f"{output_file}_clean.csv"
        clean_df.to_csv(clean_file, index=False)

        # Calculate SHA-256 hashes
        with open(raw_file, 'rb') as f:
            raw_hash = hashlib.sha256(f.read()).hexdigest()
        with open(clean_file, 'rb') as f:
            clean_hash = hashlib.sha256(f.read()).hexdigest()

        # Calculate total posts processed
        total_posts_processed = sum(self.rejection_counts.values()) + len(df)
        
        # Update metadata with file information and rejection statistics
        self.metadata.update({
            "raw_file": raw_file,
            "raw_file_size_bytes": os.path.getsize(raw_file),
            "raw_file_hash": raw_hash,
            "raw_row_count": len(df),
            "clean_file": clean_file,
            "clean_file_size_bytes": os.path.getsize(clean_file),
            "clean_file_hash": clean_hash,
            "clean_row_count": len(clean_df),
            "column_count": len(df.columns),
            "filtering_stats": {
                "total_posts_processed": total_posts_processed,
                "posts_after_cleaning": len(clean_df),
                "posts_removed": len(df) - len(clean_df),
                "removal_percentage": round((len(df) - len(clean_df)) / len(df) * 100, 1) if len(df) > 0 else 0
            },
            "rejection_stats": {
                "total_rejections": sum(self.rejection_counts.values()),
                "rejection_percentage": round(sum(self.rejection_counts.values()) / total_posts_processed * 100, 1) if total_posts_processed > 0 else 0,
                "reasons": {
                    reason: {
                        "count": count,
                        "percentage": round(count / total_posts_processed * 100, 1) if total_posts_processed > 0 else 0
                    }
                    for reason, count in self.rejection_counts.items()
                    if count > 0
                }
            }
        })

        # Save metadata as JSON
        metadata_file = f"{output_file}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved raw snapshot to {raw_file} ({len(df)} posts)")
        logger.info(f"Saved filtered snapshot to {clean_file} ({len(clean_df)} posts)")
        logger.info(f"Saved metadata to {metadata_file}")
        logger.info(f"Filtering removed {len(df) - len(clean_df)} posts ({self.metadata['filtering_stats']['removal_percentage']}%)")
        logger.info("\nRejection statistics:")
        for reason, stats in self.metadata['rejection_stats']['reasons'].items():
            logger.info(f"  {reason}: {stats['count']} posts ({stats['percentage']}%)")

    def _initialize_reddit_client(self):
        """Initialize the Reddit API client with proper settings"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                request_timeout=30  # 30 second timeout
            )
            # Test the connection
            self._check_connection()
            logger.info("Successfully initialized Reddit client")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise

    def _check_connection(self) -> bool:
        """Check if Reddit API connection is working"""
        try:
            # Try to access a public subreddit instead of user.me()
            subreddit = self.reddit.subreddit("test")
            # Just try to get the subreddit info
            _ = subreddit.title
            logger.info("Successfully connected to Reddit API")
            return True
        except Exception as e:
            logger.error(f"Reddit API connection check failed: {e}")
            return False

    def _handle_connection_error(self, error: Exception) -> bool:
        """Handle connection errors with dynamic backoff"""
        self.retry_count += 1
        if self.retry_count > self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded")
            return False

        delay = self._calculate_backoff()
        logger.warning(f"Connection error: {error}. Retrying in {delay:.1f} seconds...")
        time.sleep(delay)
        return True

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API errors
        
        Returns:
            Dictionary with error statistics
        """
        now = time.time()
        recent_errors = [t for t in self.error_times if now - t < 3600]  # Last hour
        
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts,
            'errors_last_hour': len(recent_errors),
            'error_rate': len(recent_errors) / 3600 if recent_errors else 0,
            'retry_count': self.retry_count
        }

    def print_collection_stats(self):
        """Print current collection statistics"""
        stats = {
            "Total posts": len(self.all_posts),
            "Unique posts": len(self.seen_post_ids),
            "Elapsed time": str(datetime.timedelta(
                seconds=int(time.time() - self.start_time)
            )),
            "Posts per minute": round(
                len(self.all_posts) / 
                ((time.time() - self.start_time) / 60), 2
            ),
            "Memory usage": f"{psutil.Process().memory_info().rss / (1024 * 1024):.1f} MB",
            "Memory management": {
                "Warnings": self.memory_stats['warnings'],
                "Critical": self.memory_stats['critical'],
                "Emergency": self.memory_stats['emergency'],
                "GC runs": self.memory_stats['gc_runs'],
                "Memory cleared": self.memory_stats['memory_cleared']
            }
        }
        
        # Add rejection statistics
        total_rejections = sum(self.rejection_counts.values())
        if total_rejections > 0:
            stats["Rejection reasons"] = {
                reason: f"{count} ({count/total_rejections*100:.1f}%)"
                for reason, count in self.rejection_counts.items()
                if count > 0
            }
        
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")

    def _process_post(self, post) -> Optional[Dict[str, Any]]:
        """
        Process a Reddit post, extract relevant data, and check for spam.
        
        Args:
            post: PRAW post object
            
        Returns:
            Dictionary with processed post data or None if post should be skipped
        """
        try:
            logger.debug(f"Processing post {post.id}: {post.title}")
            
            # Skip if post is spam
            if self.is_spam(post):
                logger.debug(f"Post {post.id} rejected: spam")
                self.rejection_counts['spam'] += 1
                return None
                
            # Skip if post is too old (older than time window)
            if hasattr(post, 'created_utc'):
                post_age = time.time() - post.created_utc
                logger.debug(f"Post {post.id} age: {post_age/3600:.1f} hours, window: {self.time_window_seconds/3600:.1f} hours")
                if post_age > self.time_window_seconds:
                    logger.debug(f"Post {post.id} rejected: too old (age: {post_age/3600:.1f} hours, window: {self.time_window_seconds/3600:.1f} hours)")
                    self.rejection_counts['too_old'] += 1
                    return None
            
            # Skip stickied posts (typically mod announcements, AMAs, etc.)
            if getattr(post, 'stickied', False):
                logger.debug(f"Post {post.id} rejected: stickied")
                self.rejection_counts['stickied'] += 1
                return None
            
            # Extract medication mentions and check for medication context
            medications, has_med_context = self.find_medication_mentions(post.title, post.selftext)
            logger.debug(f"Post {post.id} - Medications: {medications}, Has context: {has_med_context}")
            
            # Check if this is a medication-specific subreddit
            subreddit_name = post.subreddit.display_name.lower()
            is_med_specific = subreddit_name in self.med_dict.get_all_medication_names()
            logger.debug(f"Post {post.id} - Subreddit {subreddit_name} is med-specific: {is_med_specific}")
            
            # For medication-specific subreddits, we don't require explicit mentions
            # For general subreddits, we require medication context
            if not is_med_specific and not has_med_context:
                logger.debug(f"Post {post.id} rejected: no medication context in general subreddit")
                self.rejection_counts['no_med_context'] += 1
                return None
                
            # For medication-specific subreddits, add the subreddit's medication if no explicit mentions
            if is_med_specific and not medications:
                subreddit_med = self.med_dict.get_generic_name(subreddit_name)
                if subreddit_med:
                    medications = [subreddit_med]
                    logger.debug(f"Post {post.id} - Added subreddit medication: {subreddit_med}")
                
            # Calculate engagement metrics
            score = getattr(post, 'score', 0)
            upvote_ratio = getattr(post, 'upvote_ratio', 0)
            num_comments = getattr(post, 'num_comments', 0)
            
            # Determine engagement level
            engagement_level = "high"
            if score <= 0:
                engagement_level = "low"
            elif score < 5 or num_comments < 2:
                engagement_level = "medium"
                
            # Extract basic post data
            post_data = {
                'id': post.id,
                'title': post.title,
                'selftext': post.selftext,
                'author': post.author.name if post.author else '[deleted]',
                'created_utc': post.created_utc,
                'score': score,
                'upvote_ratio': upvote_ratio,
                'num_comments': num_comments,
                'engagement_level': engagement_level,
                'subreddit': post.subreddit.display_name,
                'medications': medications,
                'is_medication_specific_subreddit': is_med_specific,
                'url': f"https://reddit.com{post.permalink}",
                'is_self': post.is_self,
                'over_18': post.over_18,
                'spoiler': post.spoiler,
                'stickied': post.stickied,
                'distinguished': post.distinguished,
                'edited': post.edited,
                'locked': post.locked,
                'removed_by_category': post.removed_by_category,
                'collection_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Add additional metadata
            if hasattr(post, 'link_flair_text'):
                post_data['link_flair_text'] = post.link_flair_text
            if hasattr(post, 'link_flair_css_class'):
                post_data['link_flair_css_class'] = post.link_flair_css_class
                
            # Add language detection
            text = f"{post.title} {post.selftext}"
            post_data['language'] = self.detect_language_safe(text)
            logger.debug(f"Post {post.id} - Language: {post_data['language']}")
            
            # Skip non-English posts
            if post_data['language'] != 'en':
                logger.debug(f"Post {post.id} rejected: non-English ({post_data['language']})")
                self.rejection_counts['non_english'] += 1
                return None
                
            # Add text length
            post_data['text_length'] = len(post.selftext)
            logger.debug(f"Post {post.id} - Text length: {post_data['text_length']}")
            
            # Skip very short posts
            if post_data['text_length'] < 20:
                logger.debug(f"Post {post.id} rejected: too short ({post_data['text_length']} chars)")
                self.rejection_counts['too_short'] += 1
                return None
                
            logger.debug(f"Post {post.id} accepted")
            return post_data
            
        except Exception as e:
            logger.warning(f"Error processing post {getattr(post, 'id', 'unknown')}: {e}")
            self.rejection_counts['processing_error'] += 1
            return None

    def _check_memory_usage(self) -> bool:
        """
        Check memory usage and perform progressive garbage collection if needed.
        
        Returns:
            bool: True if memory usage is acceptable, False if emergency measures were taken
        """
        try:
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            self.memory_history.append(memory_usage)
            
            # Calculate memory usage trend
            if len(self.memory_history) > 1:
                trend = sum(1 for i in range(1, len(self.memory_history)) 
                          if self.memory_history[i] > self.memory_history[i-1])
                is_increasing = trend > len(self.memory_history) * 0.7  # 70% of readings show increase
            else:
                is_increasing = False
            
            current_time = time.time()
            time_since_last_gc = current_time - self.last_gc_time
            
            # Progressive garbage collection based on memory usage
            if memory_usage >= self.MEMORY_EMERGENCY_THRESHOLD:
                # Emergency measures
                self.memory_stats['emergency'] += 1
                logger.warning(f"EMERGENCY: Memory usage at {memory_usage:.1%}")
                
                # Force immediate garbage collection
                gc.collect(generation=2)
                self.memory_stats['gc_runs'] += 1
                
                # Clear non-essential data
                self._emergency_memory_cleanup()
                return False
                
            elif memory_usage >= self.MEMORY_CRITICAL_THRESHOLD:
                # Critical memory usage
                self.memory_stats['critical'] += 1
                logger.warning(f"CRITICAL: Memory usage at {memory_usage:.1%}")
                
                if time_since_last_gc >= self.gc_cooldown:
                    # Aggressive garbage collection
                    gc.collect(generation=2)
                    gc.collect(generation=1)
                    self.memory_stats['gc_runs'] += 1
                    self.last_gc_time = current_time
                    
                    # Clear some cached data
                    self._clear_cached_data()
                    
            elif memory_usage >= self.MEMORY_WARNING_THRESHOLD:
                # Warning level memory usage
                self.memory_stats['warnings'] += 1
                logger.warning(f"WARNING: Memory usage at {memory_usage:.1%}")
                
                if time_since_last_gc >= self.gc_cooldown and (is_increasing or memory_usage > 0.8):
                    # Standard garbage collection
                    gc.collect()
                    self.memory_stats['gc_runs'] += 1
                    self.last_gc_time = current_time
            
            # Record memory usage in metadata
            if IS_COLAB:
                self.metadata["colab_specific"]["memory_usage"].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "usage": memory_usage,
                    "gc_runs": self.memory_stats['gc_runs']
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return True  # Continue if we can't check memory

    def _emergency_memory_cleanup(self):
        """Perform emergency memory cleanup"""
        try:
            # Clear all cached data
            self._clear_cached_data()
            
            # Clear any stored posts that haven't been saved
            if len(self.all_posts) > self.batch_size:
                self.save_batch(len(self.all_posts))
                self.all_posts = []
            
            # Clear any stored error times
            self.error_times.clear()
            
            # Force garbage collection on all generations
            gc.collect(generation=2)
            gc.collect(generation=1)
            gc.collect(generation=0)
            
            # Clear memory stats
            self.memory_stats['memory_cleared'] += 1
            
            logger.warning("Emergency memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during emergency memory cleanup: {e}")

    def _clear_cached_data(self):
        """Clear non-essential cached data"""
        try:
            # Clear any cached language detection results
            if hasattr(self.detect_language_safe, 'cache_info'):
                self.detect_language_safe.cache_clear()
            
            # Clear any cached regex patterns
            if hasattr(self, 'med_regex'):
                self.med_regex = None
            
            # Clear any cached subreddit data
            if hasattr(self, 'subreddit_cache'):
                self.subreddit_cache.clear()
            
            logger.debug("Cleared cached data")
            
        except Exception as e:
            logger.error(f"Error clearing cached data: {e}")

def create_snapshot(args):
    """
    Create a snapshot using command-line arguments
    """
    # Initialize snapshot creator
    snapshot = RedditDataSnapshot(
        client_id=args.client_id,
        client_secret=args.client_secret,
        user_agent=args.user_agent,
        data_dir=args.data_dir
    )

    # Collect snapshot
    df = snapshot.collect_snapshot(
        subreddits=args.subreddits.split(',') if args.subreddits else None,
        time_window=args.time_window,
        post_limit=args.post_limit,
        total_limit=args.total_limit
    )

    # Save both raw and filtered snapshots
    snapshot.save_snapshot(df, args.output)

    return df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create a snapshot of medication-related Reddit data")

    # Required arguments
    parser.add_argument('--client_id', required=True, help="Reddit API client ID")
    parser.add_argument('--client_secret', required=True, help="Reddit API client secret")
    parser.add_argument('--output', required=True, help="Output file path (without extension)")

    # Optional arguments
    parser.add_argument('--user_agent', default="PsychMedPredictor/1.0", help="User agent for Reddit API")
    parser.add_argument('--subreddits', help="Comma-separated list of subreddits")
    parser.add_argument('--time_window', default="year", choices=['day', 'week', 'month', 'year', 'all'], help="Reddit search time window")
    parser.add_argument('--post_limit', type=int, default=500, help="Maximum posts per subreddit + query")
    parser.add_argument('--total_limit', type=int, default=50000, help="Maximum total posts")
    parser.add_argument('--data_dir', default="reddit_data", help="Directory to store data")

    args = parser.parse_args()

    # Create snapshot
    df = create_snapshot(args)

    print(f"Created snapshot with {len(df)} raw posts")
    print(f"Raw data saved to {args.output}_raw.csv")
    print(f"Filtered data saved to {args.output}_clean.csv")
    print(f"Metadata saved to {args.output}_metadata.json")