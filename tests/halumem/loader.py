from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass, field

@dataclass
class HaluMemMemory:
    index: int
    memory_content: str
    memory_type: str
    timestamp: str
    importance: float
    is_update: str
    original_memories: List[str]
    memory_source: str
    event_source: Optional[int] = None
    reason: Optional[str] = None
    source_memory_index: Optional[int] = None
    reference_memory_content: Optional[str] = None
    reference_memory_index: Optional[int] = None
    reference_memory_type: Optional[str] = None

@dataclass
class HaluMemTurn:
    role: str
    content: str
    timestamp: str
    dialogue_turn: int

@dataclass
class HaluMemQuestion:
    question: str
    answer: str
    evidence: List[Dict[str, Any]]
    difficulty: str
    question_type: str

@dataclass
class HaluMemSession:
    start_time: str
    end_time: str
    memory_points: List[HaluMemMemory]
    dialogue: List[HaluMemTurn]
    questions: List[HaluMemQuestion]
    dialogue_token_length: int
    question_count: int
    memory_points_count: int

@dataclass
class HaluMemItem:
    uuid: str
    persona_info: str
    sessions: List[HaluMemSession]
    total_dialogue_token_length: int
    total_question_count: int
    token_cost: Dict[str, Any]

class HaluMemLoader:
    def __init__(self, data_path: str):
        """
        Initialize the loader with the path to the dataset file.
        
        Args:
            data_path: Path to the JSONL file (e.g. HaluMem-Medium.jsonl)
        """
        self.data_path = data_path

    def load(self, limit: Optional[int] = None) -> List[HaluMemItem]:
        """
        Load the dataset.
        
        Args:
            limit: Optional limit on the number of items to load.
        
        Returns:
            List of HaluMemItem objects.
        """
        items = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if limit and count >= limit:
                    break
                if not line.strip():
                    continue
                
                data = json.loads(line)
                
                sessions = []
                for sess_data in data.get('sessions', []):
                    # Parse Memory Points
                    memories = []
                    for mem in sess_data.get('memory_points', []):
                        # Handle optional fields with get
                        memory_obj = HaluMemMemory(
                            index=mem.get('index', -1),
                            memory_content=mem.get('memory_content', ''),
                            memory_type=mem.get('memory_type', ''),
                            timestamp=mem.get('timestamp', ''),
                            importance=mem.get('importance', 0.0),
                            is_update=str(mem.get('is_update', 'False')),
                            original_memories=mem.get('original_memories', []),
                            memory_source=mem.get('memory_source', ''),
                            event_source=mem.get('event_source'),
                            reason=mem.get('reason'),
                            source_memory_index=mem.get('source_memory_index'),
                            reference_memory_content=mem.get('reference_memory_content'),
                            reference_memory_index=mem.get('reference_memory_index'),
                            reference_memory_type=mem.get('reference_memory_type')
                        )
                        memories.append(memory_obj)
                    
                    # Parse Dialogue
                    dialogue = []
                    for turn in sess_data.get('dialogue', []):
                        turn_obj = HaluMemTurn(
                            role=turn.get('role', ''),
                            content=turn.get('content', ''),
                            timestamp=turn.get('timestamp', ''),
                            dialogue_turn=turn.get('dialogue_turn', -1)
                        )
                        dialogue.append(turn_obj)
                    
                    # Parse Questions
                    questions = []
                    for q in sess_data.get('questions', []):
                        q_obj = HaluMemQuestion(
                            question=q.get('question', ''),
                            answer=q.get('answer', ''),
                            evidence=q.get('evidence', []),
                            difficulty=q.get('difficulty', ''),
                            question_type=q.get('question_type', '')
                        )
                        questions.append(q_obj)
                    
                    session_obj = HaluMemSession(
                        start_time=sess_data.get('start_time', ''),
                        end_time=sess_data.get('end_time', ''),
                        memory_points=memories,
                        dialogue=dialogue,
                        questions=questions,
                        dialogue_token_length=sess_data.get('dialogue_token_length', 0),
                        question_count=sess_data.get('question_count', 0),
                        memory_points_count=sess_data.get('memory_points_count', 0)
                    )
                    sessions.append(session_obj)
                
                item = HaluMemItem(
                    uuid=data.get('uuid', ''),
                    persona_info=data.get('persona_info', ''),
                    sessions=sessions,
                    total_dialogue_token_length=data.get('total_dialogue_token_length', 0),
                    total_question_count=data.get('total_question_count', 0),
                    token_cost=data.get('token_cost', {})
                )
                items.append(item)
                count += 1
                
        return items

