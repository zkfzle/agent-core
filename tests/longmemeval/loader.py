from typing import List, Optional, Dict, Any
import json
from dataclasses import dataclass, field

@dataclass
class ConversationTurn:
    role: str
    content: str
    has_answer: Optional[bool] = None

@dataclass
class QuestionContent:
    question: str
    answer: str
    evidence: Optional[List[str]] = None
    decomp_facts: Optional[Dict[str, str]] = None

@dataclass
class LongMemEvalItem:
    id: str
    background_id: str
    background_text: str
    question_type: str
    question: str
    answer: str
    evidence: List[str] = field(default_factory=list)
    conversations: List[ConversationTurn] = field(default_factory=list)

class LongMemEvalLoader:
    def __init__(self, 
                 questions_path: str, 
                 backgrounds_path: str, 
                 sessions_paths: List[str]):
        """
        Initialize the loader with file paths.
        
        Args:
            questions_path: Path to the questions JSON file (e.g. data_2_questions.json)
            backgrounds_path: Path to the backgrounds JSON file (e.g. data_1_attr_bg.json)
            sessions_paths: List of paths to session/history JSON files (e.g. data_6_session_cache.json)
        """
        self.questions_path = questions_path
        self.backgrounds_path = backgrounds_path
        self.sessions_paths = sessions_paths
        
        # Caches
        self._backgrounds_map: Dict[str, str] = {}
        self._sessions_map: Dict[str, List[ConversationTurn]] = {}

    def _load_backgrounds(self):
        """Load and index backgrounds by background_id."""
        with open(self.backgrounds_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            # Each item has a list of backgrounds
            for bg in item.get('backgrounds', []):
                bg_id = bg.get('background_id')
                bg_text = bg.get('background_text')
                if bg_id and bg_text:
                    self._backgrounds_map[bg_id] = bg_text

    def _load_sessions(self):
        """Load and index sessions from all session files."""
        for path in self.sessions_paths:
            with open(path, 'r') as f:
                data = json.load(f)
                
            for item in data:
                session_id = item.get('session_id')
                if not session_id:
                    continue
                
                turns_data = []
                
                # Case 1: 'sessions' key which is a list of lists of turns
                if 'sessions' in item and isinstance(item['sessions'], list):
                    for sub_sess in item['sessions']:
                        if isinstance(sub_sess, list):
                            turns_data.extend(sub_sess)
                
                # Case 2: 'session' key which is a list of turns
                elif 'session' in item and isinstance(item['session'], list):
                    turns_data.extend(item['session'])
                
                # Case 3: 'session_X' keys (e.g. session_1, session_2)
                else:
                    # Collect all keys that look like session_N
                    # Sort them to maintain order if possible, though dict order is usually insertion order
                    found_numbered = False
                    for i in range(1, 20):
                        key = f"session_{i}"
                        if key in item and isinstance(item[key], list):
                            turns_data.extend(item[key])
                            found_numbered = True
                    
                    if not found_numbered:
                        # Fallback: check raw keys just in case
                         for k, v in item.items():
                            if k.startswith('session_') and isinstance(v, list) and k not in ['session_id']:
                                 pass

                if turns_data:
                    turns = [
                        ConversationTurn(
                            role=t.get('role', ''),
                            content=t.get('content', ''),
                            has_answer=t.get('has_answer')
                        ) for t in turns_data
                    ]
                    self._sessions_map[session_id] = turns

    def load(self, limit: Optional[int] = None) -> List[LongMemEvalItem]:
        """
        Load and assemble the dataset.
        
        Args:
            limit: Optional limit on the number of items to load (useful for testing).
        """
        # Load dependencies if not already loaded
        if not self._backgrounds_map:
            self._load_backgrounds()
        if not self._sessions_map:
            self._load_sessions()
            
        with open(self.questions_path, 'r') as f:
            questions_data = json.load(f)
            
        results = []
        count = 0
        
        for q_item in questions_data:
            if limit and count >= limit:
                break
                
            q_content = q_item.get('question_content', {})
            bg_id = q_item.get('background_id')
            
            # Retrieve Background
            bg_text = self._backgrounds_map.get(bg_id, "")
            
            # Retrieve and Concatenate Conversations
            conversations = []
            session_refs = q_item.get('sessions', [])
            for sess_ref in session_refs:
                sid = sess_ref.get('session_id')
                if sid in self._sessions_map:
                    conversations.extend(self._sessions_map[sid])
            
            # Extract Evidence (handle different formats if necessary)
            evidence = []
            if 'evidence' in q_content:
                evidence = q_content['evidence']
            elif 'decomp_facts' in q_content:
                # Convert decomp_facts dict to list of strings
                facts = q_content['decomp_facts']
                if isinstance(facts, dict):
                    evidence = list(facts.values())
            
            item = LongMemEvalItem(
                id=q_item.get('question_id', ''),
                background_id=bg_id if bg_id else '',
                background_text=bg_text,
                question_type=q_item.get('question_type', 'unknown'),
                question=q_content.get('question', ''),
                answer=q_content.get('answer', ''),
                evidence=evidence,
                conversations=conversations
            )
            
            results.append(item)
            count += 1
            
        return results
