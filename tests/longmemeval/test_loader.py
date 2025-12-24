import os
import sys
from tests.longmemeval.loader import LongMemEvalLoader

def test_loading():
    base_dir = "tests/longmemeval/data/custom_history_data"
    
    # Define paths
    questions_path = os.path.join(base_dir, "2_questions/0822_all_500_questions_final_v2.json")
    bg_path = os.path.join(base_dir, "1_attr_bg/data_1_attr_bg.json")
    
    sessions_path_1 = os.path.join(base_dir, "6_session_cache/data_6_session_cache.json")
    # sessions_path_2 = os.path.join(base_dir, "5_filler_sess/data_5_filler_sess.json")
    
    loader = LongMemEvalLoader(
        questions_path=questions_path,
        backgrounds_path=bg_path,
        sessions_paths=[sessions_path_1]
    )
    
    print("Loading data...")
    # Load 5 items to test
    items = loader.load(limit=5)
    
    print(f"Successfully loaded {len(items)} items.")
    for i, item in enumerate(items):
        print(f"\nItem {i+1}:")
        print(f"  ID: {item.id}")
        print(f"  Type: {item.question_type}")
        print(f"  Question: {item.question}")
        print(f"  Answer: {item.answer}")
        print(f"  Background Length: {len(item.background_text)}")
        print(f"  Conversation Turns: {len(item.conversations)}")
        if item.conversations:
            print(f"First Turn: {item.conversations[0].content[:50]}...")

if __name__ == "__main__":
    test_loading()

