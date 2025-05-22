import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import datetime
import time
import json
import os
from src.snapshot import RedditDataSnapshot, TokenBucket

class TestRedditDataSnapshot(unittest.TestCase):
    """Test cases for RedditDataSnapshot class"""

    def setUp(self):
        """Set up test fixtures"""
        self.snapshot = RedditDataSnapshot(
            client_id="test_client_id",
            client_secret="test_client_secret",
            user_agent="test_user_agent"
        )
        
        # Mock Reddit client
        self.mock_reddit = Mock()
        self.snapshot.reddit = self.mock_reddit
        
        # Sample medication dictionary data
        self.snapshot.med_dict = Mock()
        self.snapshot.med_dict.get_all_medication_names.return_value = [
            "prozac", "fluoxetine", "zoloft", "sertraline"
        ]
        self.snapshot.med_dict.get_generic_name.side_effect = lambda x: {
            "prozac": "fluoxetine",
            "fluoxetine": "fluoxetine",
            "zoloft": "sertraline",
            "sertraline": "sertraline"
        }.get(x.lower())
        
        # Mock rate limiter
        self.snapshot.rate_limiter = Mock(spec=TokenBucket)
        self.snapshot.rate_limiter.acquire.return_value = True
        
        # Mock language detection
        self.snapshot.detect_language_safe = Mock(return_value='en')
        
        # Setup test data directory
        self.test_data_dir = "test_data"
        self.snapshot.data_dir = self.test_data_dir
        os.makedirs(self.test_data_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove test data directory
        if os.path.exists(self.test_data_dir):
            for file in os.listdir(self.test_data_dir):
                os.remove(os.path.join(self.test_data_dir, file))
            os.rmdir(self.test_data_dir)

    @patch('time.time')
    @patch('time.sleep')
    def test_token_bucket(self, mock_sleep, mock_time):
        """Test token bucket rate limiting with mocked time"""
        # Initialize time at 0
        mock_time.return_value = 0.0
        bucket = TokenBucket(rate=1.0, burst=2)
        
        # Test initial state
        self.assertTrue(bucket.acquire())
        self.assertTrue(bucket.acquire())
        self.assertFalse(bucket.acquire())  # Should be empty
        
        # Test token refresh - advance time by 1.1 seconds
        mock_time.return_value = 1.1
        self.assertTrue(bucket.acquire())
        
        # Test burst limit
        bucket.tokens = bucket.burst
        self.assertTrue(bucket.acquire(bucket.burst))
        self.assertFalse(bucket.acquire(1))
        
        # Verify sleep was never called
        mock_sleep.assert_not_called()

    def test_token_bucket_timing_tolerance(self):
        """Test token bucket with timing tolerance"""
        bucket = TokenBucket(rate=1.0, burst=2)
        
        # Test initial state
        self.assertTrue(bucket.acquire())
        self.assertTrue(bucket.acquire())
        self.assertFalse(bucket.acquire())  # Should be empty
        
        # Test token refresh with tolerance
        time.sleep(1.1)  # Slightly longer than rate to account for system timing
        self.assertTrue(bucket.acquire())
        
        # Test burst limit
        bucket.tokens = bucket.burst
        self.assertTrue(bucket.acquire(bucket.burst))
        self.assertFalse(bucket.acquire(1))

    @patch('time.time')
    def test_process_post_timing(self, mock_time):
        """Test post processing with mocked time"""
        # Set initial time
        mock_time.return_value = 1000.0
        
        # Create mock post
        mock_post = Mock(
            id="123",
            title="Test Post",
            selftext="Test content",
            author=Mock(name="test_user"),
            created_utc=999.0,  # 1 second ago
            score=10,
            upvote_ratio=0.8,
            num_comments=5,
            subreddit=Mock(display_name="test_subreddit"),
            permalink="/r/test/123",
            is_self=True,
            over_18=False,
            spoiler=False,
            stickied=False,
            distinguished=None,
            edited=False,
            locked=False,
            removed_by_category=None
        )
        
        # Process post
        result = self.snapshot._process_post(mock_post)
        
        # Verify timing calculations
        self.assertEqual(result['created_utc'], 999.0)
        self.assertEqual(result['age_days'], 1.0/86400)  # 1 second in days

    def test_find_medication_mentions(self):
        """Test basic medication mention detection"""
        test_cases = [
            # Basic detection
            (
                "I'm taking Prozac",
                "I've been on it for a month",
                ["fluoxetine"],
                "Basic mention"
            ),
            (
                "Started Zoloft yesterday",
                "Feeling better already",
                ["sertraline"],
                "Brand name mention"
            ),
            (
                "On Lexapro and Wellbutrin",
                "Both working well",
                ["escitalopram", "bupropion"],
                "Multiple medications"
            ),
            # Case insensitivity
            (
                "PROZAC is helping",
                "prozac is great",
                ["fluoxetine"],
                "Case insensitive"
            ),
            # No medications
            (
                "Feeling better today",
                "No medications mentioned",
                [],
                "No medications"
            )
        ]
        
        for title, text, expected, description in test_cases:
            with self.subTest(description=description):
                result = self.snapshot.find_medication_mentions(title, text)
                self.assertEqual(sorted(result), sorted(expected))

    def test_batch_saving(self):
        """Test batch saving functionality"""
        # Create test posts
        test_posts = [
            {
                'id': '1',
                'title': 'Test Post 1',
                'selftext': 'Content 1',
                'medications': ['fluoxetine']
            },
            {
                'id': '2',
                'title': 'Test Post 2',
                'selftext': 'Content 2',
                'medications': ['sertraline']
            }
        ]
        
        # Set batch size
        self.snapshot.batch_size = 1
        
        # Add posts and trigger batch save
        self.snapshot.all_posts = test_posts.copy()
        self.snapshot.save_batch()
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.snapshot.output_file))
        
        # Verify content
        df = pd.read_csv(self.snapshot.output_file)
        self.assertEqual(len(df), 1)  # Only first batch should be saved
        self.assertEqual(df.iloc[0]['id'], '1')

    def test_checkpoint_handling(self):
        """Test checkpoint saving and loading"""
        # Create test data
        test_posts = [
            {
                'id': '1',
                'title': 'Test Post 1',
                'selftext': 'Content 1',
                'medications': ['fluoxetine']
            }
        ]
        
        # Save checkpoint
        self.snapshot.all_posts = test_posts
        self.snapshot.seen_post_ids = {'1'}
        self.snapshot.save_checkpoint()
        
        # Verify checkpoint files
        checkpoint_file = os.path.join(self.test_data_dir, f"checkpoint_{len(test_posts)}.csv")
        self.assertTrue(os.path.exists(checkpoint_file))
        
        # Clear data
        self.snapshot.all_posts = []
        self.snapshot.seen_post_ids = set()
        
        # Load checkpoint
        self.snapshot.load_checkpoint()
        self.assertEqual(len(self.snapshot.seen_post_ids), 1)
        self.assertEqual(list(self.snapshot.seen_post_ids)[0], '1')

    def test_error_handling(self):
        """Test error handling and rate limiting"""
        # Test rate limit handling
        with patch.object(self.snapshot, '_make_api_request') as mock_request:
            mock_request.side_effect = [
                Exception("Rate limit"),
                "Success"
            ]
            
            # Should retry and eventually succeed
            result = self.snapshot._make_api_request(lambda: "test")
            self.assertEqual(result, "Success")
            
        # Test connection error handling
        with patch.object(self.snapshot, '_make_api_request') as mock_request:
            mock_request.side_effect = ConnectionError()
            
            # Should raise after max retries
            with self.assertRaises(ConnectionError):
                self.snapshot._make_api_request(lambda: "test")
                
        # Test error statistics
        self.snapshot.error_counts['rate_limit'] = 5
        self.snapshot.error_counts['connection'] = 3
        stats = self.snapshot.get_error_stats()
        self.assertEqual(stats['total_errors'], 8)
        self.assertEqual(stats['error_counts']['rate_limit'], 5)

    def test_colab_specific(self):
        """Test Colab-specific functionality"""
        # Mock Colab environment
        with patch('src.snapshot.IS_COLAB', True):
            # Test memory check
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 90  # High memory usage
                self.assertFalse(self.snapshot._check_colab_memory())
                
                mock_memory.return_value.percent = 50  # Normal memory usage
                self.assertTrue(self.snapshot._check_colab_memory())
            
            # Test Colab disconnection handling
            with patch.object(self.snapshot, '_check_connection', return_value=False):
                self.assertFalse(self.snapshot._handle_colab_disconnect())

    def test_process_post(self):
        """Test post processing with various scenarios"""
        # Create base mock post
        def create_mock_post(**kwargs):
            base_post = Mock(
                id="123",
                title="Test Post",
                selftext="Test content",
                author=Mock(name="test_user"),
                created_utc=time.time() - 86400,  # 1 day ago
                score=10,
                upvote_ratio=0.8,
                num_comments=5,
                subreddit=Mock(display_name="test_subreddit"),
                permalink="/r/test/123",
                is_self=True,
                over_18=False,
                spoiler=False,
                stickied=False,
                distinguished=None,
                edited=False,
                locked=False,
                removed_by_category=None
            )
            for key, value in kwargs.items():
                setattr(base_post, key, value)
            return base_post

        test_cases = [
            # (post_kwargs, expected_result, description)
            (
                {
                    'title': 'Started Prozac',
                    'selftext': 'Taking 20mg daily',
                    'created_utc': time.time() - 86400  # 1 day ago
                },
                True,
                "Valid post with medication"
            ),
            (
                {
                    'selftext': '[removed]',
                    'title': 'Removed Post'
                },
                False,
                "Removed post"
            ),
            (
                {
                    'selftext': '[deleted]',
                    'title': 'Deleted Post'
                },
                False,
                "Deleted post"
            ),
            (
                {
                    'title': 'No medication post',
                    'selftext': 'Just a regular post'
                },
                False,
                "No medication mention"
            ),
            (
                {
                    'title': 'Prozac post',
                    'selftext': 'Corto contenido',  # Spanish text
                    'language': 'es'
                },
                False,
                "Non-English post"
            ),
            (
                {
                    'title': 'Short post',
                    'selftext': 'Hi',
                    'created_utc': time.time() - 86400
                },
                False,
                "Too short post"
            ),
            (
                {
                    'title': 'Old post',
                    'selftext': 'Taking Prozac',
                    'created_utc': time.time() - (366 * 24 * 3600)  # > 1 year
                },
                False,
                "Too old post"
            ),
            (
                {
                    'title': 'Spam post',
                    'selftext': 'Check out my website www.spam.com',
                    'created_utc': time.time() - 86400
                },
                False,
                "Spam post with URL"
            ),
            (
                {
                    'title': 'Multiple URLs',
                    'selftext': 'Here are some links:\nhttp://link1.com\nhttp://link2.com',
                    'created_utc': time.time() - 86400
                },
                False,
                "Post with multiple URLs"
            ),
            (
                {
                    'title': 'Stickied post',
                    'selftext': 'Important announcement',
                    'stickied': True
                },
                False,
                "Stickied post"
            ),
            (
                {
                    'title': 'Locked post',
                    'selftext': 'Taking Prozac',
                    'locked': True
                },
                False,
                "Locked post"
            )
        ]

        for post_kwargs, expected_result, description in test_cases:
            with self.subTest(description=description):
                post = create_mock_post(**post_kwargs)
                result = self.snapshot._process_post(post)
                if expected_result:
                    self.assertIsNotNone(result, f"Expected valid result for: {description}")
                    self.assertEqual(result['id'], post.id)
                    # Verify all required fields are present
                    required_fields = ['id', 'title', 'selftext', 'author', 'created_utc', 
                                     'score', 'upvote_ratio', 'num_comments', 'subreddit',
                                     'permalink', 'medications']
                    for field in required_fields:
                        self.assertIn(field, result, f"Missing required field: {field}")
                else:
                    self.assertIsNone(result, f"Expected None for: {description}")

    def test_clean_snapshot(self):
        """Test snapshot cleaning with various scenarios"""
        # Mock medication dictionary
        self.snapshot.med_dict.get_all_medication_names.return_value = [
            'fluoxetine', 'sertraline', 'escitalopram', 'citalopram'
        ]
        
        test_data = pd.DataFrame([
            # Valid post
            {
                'id': '1',
                'title': 'Started Prozac',
                'selftext': 'Taking 20mg daily for depression',
                'language': 'en',
                'text_length': 100,
                'medications': ['fluoxetine'],
                'score': 10,
                'num_comments': 5,
                'upvote_ratio': 0.8,
                'created_utc': time.time() - 86400
            },
            # Non-English post
            {
                'id': '2',
                'title': 'Started Zoloft',
                'selftext': 'Tomando 50mg diarios',
                'language': 'es',
                'text_length': 100,
                'medications': ['sertraline'],
                'score': 8,
                'num_comments': 3,
                'upvote_ratio': 0.7,
                'created_utc': time.time() - 86400
            },
            # Short post
            {
                'id': '3',
                'title': 'Short post',
                'selftext': 'Hi',
                'language': 'en',
                'text_length': 2,
                'medications': ['fluoxetine'],
                'score': 1,
                'num_comments': 0,
                'upvote_ratio': 0.5,
                'created_utc': time.time() - 86400
            },
            # Duplicate ID
            {
                'id': '1',
                'title': 'Duplicate post',
                'selftext': 'This is a duplicate',
                'language': 'en',
                'text_length': 100,
                'medications': ['fluoxetine'],
                'score': 5,
                'num_comments': 2,
                'upvote_ratio': 0.6,
                'created_utc': time.time() - 86400
            },
            # Low engagement
            {
                'id': '4',
                'title': 'Low engagement',
                'selftext': 'No comments or upvotes',
                'language': 'en',
                'text_length': 100,
                'medications': ['sertraline'],
                'score': 0,
                'num_comments': 0,
                'upvote_ratio': 0.3,
                'created_utc': time.time() - 86400
            },
            # Old post
            {
                'id': '5',
                'title': 'Old post',
                'selftext': 'Taking Prozac',
                'language': 'en',
                'text_length': 100,
                'medications': ['fluoxetine'],
                'score': 10,
                'num_comments': 5,
                'upvote_ratio': 0.8,
                'created_utc': time.time() - (366 * 24 * 3600)  # > 1 year
            },
            # Unknown medication
            {
                'id': '6',
                'title': 'Unknown med',
                'selftext': 'Taking experimental medication',
                'language': 'en',
                'text_length': 100,
                'medications': ['experimental_med'],
                'score': 10,
                'num_comments': 5,
                'upvote_ratio': 0.8,
                'created_utc': time.time() - 86400
            },
            # Multiple medications
            {
                'id': '7',
                'title': 'Multiple meds',
                'selftext': 'Taking both Prozac and Zoloft',
                'language': 'en',
                'text_length': 100,
                'medications': ['fluoxetine', 'sertraline'],
                'score': 10,
                'num_comments': 5,
                'upvote_ratio': 0.8,
                'created_utc': time.time() - 86400
            },
            # Mixed known/unknown medications
            {
                'id': '8',
                'title': 'Mixed meds',
                'selftext': 'Taking Prozac and experimental med',
                'language': 'en',
                'text_length': 100,
                'medications': ['fluoxetine', 'experimental_med'],
                'score': 10,
                'num_comments': 5,
                'upvote_ratio': 0.8,
                'created_utc': time.time() - 86400
            }
        ])
        
        # Clean the snapshot
        cleaned = self.snapshot.clean_snapshot(test_data)
        
        # Verify results
        self.assertEqual(len(cleaned), 4)  # Posts that pass basic filters (1, 6, 7, 8)
        self.assertEqual(set(cleaned['id']), {'1', '6', '7', '8'})
        
        # Verify medications are stored as strings
        for _, row in cleaned.iterrows():
            self.assertIsInstance(row['medications'], str)
            if row['id'] == '1':
                self.assertEqual(row['medications'], 'fluoxetine')
            elif row['id'] == '6':
                self.assertEqual(row['medications'], 'experimental_med')
            elif row['id'] == '7':
                self.assertEqual(row['medications'], 'fluoxetine,sertraline')
            elif row['id'] == '8':
                self.assertEqual(row['medications'], 'fluoxetine,experimental_med')
        
        # Verify all required fields are present
        required_fields = ['id', 'title', 'selftext', 'language', 'text_length', 
                         'medications', 'score', 'num_comments', 'upvote_ratio', 
                         'created_utc']
        for field in required_fields:
            self.assertIn(field, cleaned.columns, f"Missing required field: {field}")
            
        # Verify filtering criteria
        for _, row in cleaned.iterrows():
            # Language
            self.assertEqual(row['language'], 'en')
            # Text length
            self.assertTrue(row['text_length'] > 50)
            # Engagement
            self.assertTrue(row['score'] > 0 or row['num_comments'] > 0)
            # Time window
            self.assertTrue(time.time() - row['created_utc'] <= self.snapshot.time_window_seconds)

    def test_multi_post_processing(self):
        """Test processing posts from different subreddit types"""
        # Mock subreddit responses
        med_specific_sub = Mock()
        med_specific_sub.display_name = "Prozac"
        med_specific_sub.new.return_value = [
            Mock(
                id="1",
                title="Prozac experience",
                selftext="Taking 20mg",
                author=Mock(name="user1"),
                created_utc=time.time() - 86400,
                score=10,
                upvote_ratio=0.8,
                num_comments=5,
                subreddit=med_specific_sub,
                permalink="/r/Prozac/1",
                is_self=True
            )
        ]
        
        general_sub = Mock()
        general_sub.display_name = "depression"
        general_sub.search.return_value = [
            Mock(
                id="2",
                title="Medication question",
                selftext="Thinking about Prozac",
                author=Mock(name="user2"),
                created_utc=time.time() - 86400,
                score=8,
                upvote_ratio=0.7,
                num_comments=3,
                subreddit=general_sub,
                permalink="/r/depression/2",
                is_self=True
            )
        ]
        
        # Mock subreddit dictionary
        self.snapshot.med_dict.get_subreddits.return_value = {
            "medication_specific": ["Prozac"],
            "general": ["depression"]
        }
        
        # Mock subreddit access
        self.mock_reddit.subreddit.side_effect = lambda x: {
            "Prozac": med_specific_sub,
            "depression": general_sub
        }.get(x)
        
        # Collect snapshot
        df = self.snapshot.collect_snapshot(
            subreddits=["Prozac", "depression"],
            time_window="year",
            post_limit=1
        )
        
        # Verify results
        self.assertEqual(len(df), 2)  # Should get posts from both subreddits
        self.assertEqual(set(df['subreddit']), {"Prozac", "depression"})
        self.assertEqual(set(df['id']), {"1", "2"})

if __name__ == '__main__':
    unittest.main() 