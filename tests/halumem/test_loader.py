from tests.halumem.loader import HaluMemLoader
import os

def test_loader():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'HaluMem-Medium.jsonl')
    
    loader = HaluMemLoader(data_path)
    items = loader.load(limit=1)
    
    print(f"Loaded {len(items)} items")
    if items:
        item = items[0]
        print(f"UUID: {item.uuid}")
        print(f"Number of sessions: {len(item.sessions)}")
        if item.sessions:
            session = item.sessions[0]
            print(f"Session 1 dialogue length: {len(session.dialogue)}")
            print(f"Session 1 questions: {len(session.questions)}")
            if session.questions:
                print(f"First question: {session.questions[0].question}")

if __name__ == "__main__":
    test_loader()

