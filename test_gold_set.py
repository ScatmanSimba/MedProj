import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from src.data.gold_set import GoldSetCreator

def test_gold_set_creation():
    # Create a small test DataFrame
    test_data = pd.DataFrame({
        'post_text': [
            "Lexapro made me numb. Wellbutrin gives me energy.",
            "Prozac helps with my anxiety but makes me sleepy.",
            "Zoloft has been great for my depression."
        ],
        'medications': [
            ["Lexapro", "Wellbutrin"],
            ["Prozac"],
            ["Zoloft"]
        ]
    })
    
    # Create gold set
    creator = GoldSetCreator()
    gold_set = creator.create_gold_set(test_data, sample_size=3)
    
    # Verify columns
    required_columns = ['post_text', 'medications', 'response_dimension_scores', 
                       'response_dimension_confidence', 'med_to_sentences']
    missing_columns = [col for col in required_columns if col not in gold_set.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    # Verify data types
    if not isinstance(gold_set['response_dimension_scores'].iloc[0], dict):
        print("response_dimension_scores should be a dictionary")
        return False
    
    if not isinstance(gold_set['response_dimension_confidence'].iloc[0], dict):
        print("response_dimension_confidence should be a dictionary")
        return False
    
    if not isinstance(gold_set['med_to_sentences'].iloc[0], dict):
        print("med_to_sentences should be a dictionary")
        return False
    
    print("Gold set creation test passed!")
    return True

if __name__ == "__main__":
    test_gold_set_creation() 