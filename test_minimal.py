import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import json
from src.features.response_attribution import ResponseAttributor

def test_minimal():
    # Test attribution
    print("\nTesting Response Attribution:")
    text = "Lexapro made me numb. Wellbutrin gives me energy."
    attrib = ResponseAttributor()
    res = attrib.attribute_responses(text, ["Lexapro", "Wellbutrin"])
    print(json.dumps(res['dimension_scores'], indent=2))
    
    # Test gold set creation
    print("\nTesting Gold Set Creation:")
    test_data = pd.DataFrame({
        'post_text': [text],
        'medications': [["Lexapro", "Wellbutrin"]]
    })
    
    from src.data.gold_set import GoldSetCreator
    creator = GoldSetCreator()
    gold_set = creator.create_gold_set(test_data, sample_size=1)
    
    print("\nGold Set Columns:")
    print(gold_set.columns.tolist())
    
    print("\nFirst Row Data:")
    print(json.dumps({
        'post_text': gold_set['post_text'].iloc[0],
        'medications': gold_set['medications'].iloc[0],
        'response_dimension_scores': gold_set['response_dimension_scores'].iloc[0],
        'med_to_sentences': gold_set['med_to_sentences'].iloc[0]
    }, indent=2))

if __name__ == "__main__":
    test_minimal() 