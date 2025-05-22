import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import json
from src.data.gold_set import GoldSetCreator

def test_interface_data():
    # Create test data
    test_data = pd.DataFrame({
        'post_text': [
            "Lexapro made me numb. Wellbutrin gives me energy.",
            "Prozac helps with my anxiety but makes me sleepy."
        ],
        'medications': [
            ["Lexapro", "Wellbutrin"],
            ["Prozac"]
        ]
    })
    
    # Create gold set
    creator = GoldSetCreator()
    gold_set = creator.create_gold_set(test_data, sample_size=2)
    
    # Save a test annotation
    test_annotation = {
        'post_id': gold_set.index[0],
        'response_dimension_scores': {
            'Lexapro': {
                'activation': 0.3,
                'emotional': 0.2,
                'metabolic': 0.5
            },
            'Wellbutrin': {
                'activation': 0.8,
                'emotional': 0.6,
                'metabolic': 0.4
            }
        },
        'confidence': 0.8,
        'notes': 'Test annotation',
        'annotator_id': 'test_annotator'
    }
    
    creator.save_annotation(gold_set.index[0], test_annotation)
    
    # Verify annotation was saved
    saved_annotation = creator.get_annotation(gold_set.index[0])
    print("\nSaved Annotation:")
    print(json.dumps(saved_annotation, indent=2))
    
    # Get unannotated posts
    unannotated = creator.get_unannotated_posts()
    print("\nUnannotated Posts:")
    print(f"Count: {len(unannotated)}")
    print("Columns:", unannotated.columns.tolist())
    
    # Verify data structure
    print("\nVerifying Data Structure:")
    print("1. Gold set has required columns:", all(col in gold_set.columns for col in [
        'post_text', 'medications', 'response_dimension_scores', 
        'response_dimension_confidence', 'med_to_sentences'
    ]))
    
    print("2. Annotations are saved correctly:", saved_annotation is not None)
    print("3. Unannotated posts are filtered correctly:", len(unannotated) == 1)

if __name__ == "__main__":
    test_interface_data() 