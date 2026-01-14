#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
LongMemEval Graph Memory Evaluation Script

Used to evaluate the performance of Graph Memory on the LongMemEval dataset.
Graph Memory stores conversation history and retrieves relevant memories to answer questions.
"""

import os
import sys
import json
import argparse
import re
import concurrent.futures
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import openai
from openai import OpenAI

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Load .env file from: {env_path}")
else:
    print(f"Did not find .env file: {env_path}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.longmemeval.data_loader import LongMemEvalLoader, LongMemEvalItem

# Import official evaluation tool evaluate_qa module
EVAL_DIR = os.path.join(os.path.dirname(__file__), 'src', 'evaluation')
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
import evaluate_qa

from openjiuwen.core.memory.store.graph_store.base import GraphMemory
from openjiuwen.core.memory.config.graph.config import GraphConfig, LLMConfig, EpisodeType
from openjiuwen.core.memory.config.graph.extraction_strategies import DEFAULT_STRATEGY
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.llm.messages import HumanMessage, SystemMessage
from openjiuwen.core.memory.store.graph_store.api_services.llm_reranker import GraphLLMClient


def parse_question_date(date_str: str) -> Optional[datetime]:
    """
    Parse the question date string, supports multiple date formats.
    
    Args:
        date_str: Date string, possible formats include:
            - ISO format: '2023-05-30T23:40:00Z' or '2023-05-30T23:40:00+00:00'
            - Format with weekday: '2023/05/30 (Tue) 23:40'
            - Other common formats
    
    Returns:
        datetime object, or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip() 

    try:
        # Matching format: YYYY/MM/DD (Day) HH:MM or YYYY/MM/DD HH:MM
        pattern = r'(\d{4})/(\d{2})/(\d{2})\s*(?:\([^)]+\))?\s*(\d{2}):(\d{2})'
        match = re.match(pattern, date_str)
        if match:
            year, month, day, hour, minute = map(int, match.groups())
            return datetime(year, month, day, hour, minute)
    except (ValueError, AttributeError):
        pass
        return None
    


def load_data(data_path: str, limit: Optional[int] = None) -> List[LongMemEvalItem]:
    """
    Load the LongMemEval dataset.
    
    Args:
        data_path: Data file path (new format, single JSON file)
        limit: Optional limit on the number of items to load (for fast testing)
        
    Returns:
        List[LongMemEvalItem]: List of loaded data items
    """
    print(f"Load Data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    try:
        loader = LongMemEvalLoader(data_path=data_path)
        items = loader.load(limit=limit)
    except Exception as e:
        print(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"Successfully loaded {len(items)} test cases")
    
    # Print data statistics
    if items:
        type_counts = {}
        total_conversations = 0
        total_sessions = 0
        items_with_empty_conversations = 0
        
        for item in items:
            type_counts[item.question_type] = type_counts.get(item.question_type, 0) + 1
            total_conversations += len(item.conversations)
            if item.haystack_sessions:
                total_sessions += len(item.haystack_sessions)
            if not item.conversations:
                items_with_empty_conversations += 1
        
        print("\nData statistics:")
        for q_type, count in sorted(type_counts.items()):
            print(f"  - {q_type}: {count} items")
        
        print(f"\nConversation statistics:")
        print(f"  - Total conversation turns: {total_conversations}")
        print(f"  - Average turns per item: {total_conversations // len(items) if items else 0}")
        if total_sessions > 0:
            print(f"  - Total sessions: {total_sessions}")
            print(f"  - Average sessions per item: {total_sessions // len(items)}")
        if items_with_empty_conversations > 0:
            print(f"  - Warning: {items_with_empty_conversations} items have no conversation history")
        
        # Print information of the first sample as an example
        sample = items[0]
        print(f"\nFirst sample data:")
        print(f"  - ID: {sample.id}")
        print(f"  - Question Type: {sample.question_type}")
        print(f"  - Question: {sample.question[:80]}...")
        print(f"  - Conversation Turns: {len(sample.conversations)}")
        if sample.haystack_sessions:
            print(f"  - Number of Sessions: {len(sample.haystack_sessions)}")
        if sample.question_date:
            print(f"  - Question Date: {sample.question_date}")
    
    return items


def init_evaluator_client() -> Tuple[OpenAI, str]:
    """
    initialize evaluator client
        
    Returns:
        Tuple[OpenAI, str]: (evaluator client, model name)
    """
    EVALUATOR_MODEL = "gpt-4o"
    EVALUATOR_API_BASE = os.getenv("EVALUATOR_API_BASE")
    EVALUATOR_API_KEY = os.getenv("EVALUATOR_API_KEY")
    
    metric_model, _ = evaluate_qa.model_zoo[EVALUATOR_MODEL]
    
    metric_client = OpenAI(
        api_key=EVALUATOR_API_KEY,
        base_url=EVALUATOR_API_BASE,
    )
    
    return metric_client, metric_model


def init_graph_memory(
    db_name: str = "longmemeval_evaluation",
    language: str = "en",
    llm_config: Optional[LLMConfig] = None
) -> GraphMemory:
    """
    Initialize Graph Memory instance.
    """
    print(f"\nProcess info: Initializing graph memory (Collection: {db_name})...")
    
    # Configure graph memory database
    local_endpoint = os.getenv("LOCAL_ENDPOINT")
    
    db_uri = f"http://{local_endpoint}"
    print(f"Milvus endpoint: {db_uri}")
    
    db_config = GraphConfig(
        uri=db_uri,
        name=db_name,
        backend="milvus",
        wipe_at_startup=True,
        timeout=30.0,
    )
    
    # Configure LLM
    if llm_config is None:
        try:
            llm_config = LLMConfig.default_config()
            # Explicitly increase timeout to 180 seconds to handle complex graph extraction tasks
            llm_config.timeout = 180.0
            print("get llm config from default config")
        except (ValueError, KeyError) as e:
            print(f"Failed to get LLM config from default config: {e}")
            print("Please set JIUWEN_GRAPH_MEM_LLM_URL, JIUWEN_GRAPH_MEM_LLM_KEY, JIUWEN_GRAPH_MEM_LLM_MODEL")
            raise
    
    # Create Graph Memory instance
    graph_memory = GraphMemory(
        db_config=db_config,
        llm_config=llm_config,
        language=language,
    )
    
    print("Process info: Successfully initialized graph memory")
    return graph_memory


def prepare_chunks(item: LongMemEvalItem, chunk_size: int, chunk_overlap: int) -> List[Tuple[List[dict], datetime]]:
    """
    Global Chunking: Flatten all sessions, inject time metadata, and merge chunks by a fixed size.
    """
    all_messages = [] # List[Dict[role, content, timestamp]]
    base_reference_time = parse_question_date(item.question_date) if item.question_date else None
    
    # 1. Extract and flatten all messages
    sessions = item.haystack_sessions if hasattr(item, 'haystack_sessions') and item.haystack_sessions else []
    if not sessions and item.conversations:
        # If no session structure, fallback to conversations
        raw_conv = [{"role": getattr(turn, 'role', 'user'), "content": getattr(turn, 'content', '')} for turn in item.conversations]
        sessions = [raw_conv]

    for s_idx, session in enumerate(sessions):
        # Get the actual date of the session
        session_date = None
        if hasattr(item, 'haystack_dates') and item.haystack_dates and s_idx < len(item.haystack_dates):
            session_date = parse_question_date(item.haystack_dates[s_idx])
        
        if session_date is None and base_reference_time:
            session_date = base_reference_time - timedelta(hours=1) + timedelta(minutes=s_idx * 10)

        # Inject a timestamp hint at the start of the session (enhance LLM time awareness)
        date_prefix = f"[System: The following messages occurred on {session_date.strftime('%Y-%m-%d %H:%M:%S')}]\n" if session_date else ""
        
        for t_idx, turn in enumerate(session):
            role = turn.get('role') if isinstance(turn, dict) else getattr(turn, 'role', None)
            content = turn.get('content') if isinstance(turn, dict) else getattr(turn, 'content', None)
            
            if role in ['user', 'assistant'] and content:
                # Add date prefix only before the first message of the session
                processed_content = date_prefix + str(content) if t_idx == 0 else str(content)
                all_messages.append({
                    "role": role,
                    "content": processed_content,
                    "timestamp": session_date + timedelta(seconds=t_idx) if session_date else None
                })

    if not all_messages:
        return []

    # 2. Sliding window chunking on messages
    final_chunks = []
    total_len = len(all_messages)
    step = max(1, chunk_size - chunk_overlap)
    
    for i in range(0, total_len, step):
        chunk_data = all_messages[i:i + chunk_size]
        # Extract start time of current chunk
        chunk_ref_time = chunk_data[0]["timestamp"]
        # Remove timestamp field, keep only role and content for graph_memory
        clean_messages = [{"role": m["role"], "content": m["content"]} for m in chunk_data]
        final_chunks.append((clean_messages, chunk_ref_time))
        
        if i + chunk_size >= total_len:
            break
            
    return final_chunks


def store_conversations_to_memory(
    graph_memory: GraphMemory,
    item: LongMemEvalItem,
    user_id: str,
    use_conversation: bool = True,
    chunk_size: Optional[int] = 20,
    chunk_overlap: int = 2,
    chunk_strategy: str = "session"
) -> bool:
    """
    Storage Phase: Receive preprocessed chunks and store them.
    """
    if not use_conversation:
        return False

    # 1. [Chunking Phase] Execute independently
    processed_chunks = prepare_chunks(item, chunk_size, chunk_overlap)
    
    if not processed_chunks:
        return False

    # 2. [Storage Phase] Traverse and store
    total = len(processed_chunks)
    print(f"\n    [Ingestion] Background: {user_id[:8]}... | Chunks Prepared: {total}")
    
    try:
        for idx, (chunk_messages, ref_time) in enumerate(processed_chunks):
            start_t = datetime.now()
            print(f"      > [{start_t.strftime('%H:%M:%S')}] Chunk {idx+1}/{total} Ingesting...", end='', flush=True)
            
            graph_memory.add_memory(
                src_type=EpisodeType.conversation,
                user_id=user_id,
                content=chunk_messages,
                reference_time=ref_time,
            )
            
            dur = (datetime.now() - start_t).total_seconds()
            print(f" Done! ({dur:.1f}s)")
        
        graph_memory.db_backend.refresh()
        return True
    except Exception as e:
        print(f"\n    Background {user_id[:8]} ingestion failed: {e}")
        return False


def format_retrieved_memories(search_results: Dict[str, List[Tuple[float, Any]]]) -> str:
    """
    Format retrieved memories into a string.
    
    Args:
        search_results: Search results, format: {collection_name: [(score, object), ...]}
        
    Returns:
        str: Formatted memory text
    """
    if not search_results:
        return ""
    
    formatted_parts = []
    
    # Format entities
    if "entity" in search_results and search_results["entity"]:
        formatted_parts.append("Relevant Entities:")
        for score, entity in search_results["entity"]:
            entity_info = f"- {entity.name}"
            if hasattr(entity, 'content') and entity.content:
                entity_info += f": {entity.content}"
            if hasattr(entity, 'attributes') and entity.attributes:
                attrs = ", ".join([f"{k}={v}" for k, v in entity.attributes.items() if v])
                if attrs:
                    entity_info += f" ({attrs})"
            formatted_parts.append(entity_info)
    
    # Format relations
    if "relation" in search_results and search_results["relation"]:
        formatted_parts.append("\nRelevant Relations:")
        for score, relation in search_results["relation"]:
            relation_info = f"- {relation.name}"
            if hasattr(relation, 'content') and relation.content:
                relation_info += f": {relation.content}"
            formatted_parts.append(relation_info)
    
    # Format episodes
    if "episode" in search_results and search_results["episode"]:
        formatted_parts.append("\nRelevant Episodes:")
        for score, episode in search_results["episode"]:
            episode_info = f"- {episode.content}"
            formatted_parts.append(episode_info)
    
    return "\n".join(formatted_parts) if formatted_parts else ""


def search_memories(
    graph_memory: GraphMemory,
    query: str,
    user_id: str,
    search_top_k: int = 5
) -> Dict[str, List[Tuple[float, Any]]]:
    """
    Retrieve relevant memories from Graph Memory.
    
    Args:
        graph_memory: Graph Memory instance
        query: Query question
        user_id: User ID
        search_top_k: Number of memories to retrieve (top_k)
        
    Returns:
        Dict[str, List[Tuple[float, Any]]]: Search results
    """
    try:
        search_results = graph_memory.search(
            query=query,
            user_id=user_id,
            search_strategy="default",
            entity=True,
            relation=True,
            episode=True,
        )
        
        # Limit return quantity for each type
        for key in search_results:
            if search_results[key]:
                search_results[key] = search_results[key][:search_top_k]
        
        return search_results
    except Exception as e:
        print(f"Failed to retrieve memories: {e}")
        return {}


def generate_answer(
    graph_memory: GraphMemory,
    question: str,
    retrieved_memories: str,
    conversation_history: Optional[str] = None,
    max_history_length: Optional[int] = 5000
) -> str:
    """
    Use LLM to generate an answer based on retrieved memories.
    
    Args:
        graph_memory: Graph Memory instance (used to get LLM client)
        question: Question
        retrieved_memories: Formatted retrieved memory text
        conversation_history: Conversation history (optional)
        max_history_length: Maximum character length of conversation history (to avoid context overflow)
        
    Returns:
        str: Generated answer
    """
    # Build prompt
    prompt_parts = []
    
    if retrieved_memories:
        prompt_parts.append("Here are the retrieved relevant memories:")
        prompt_parts.append(retrieved_memories)
        prompt_parts.append("")
    
    # Limit conversation history length to avoid context overflow
    if conversation_history:
        if max_history_length and len(conversation_history) > max_history_length:
            # Truncate the last part (usually most recent conversation)
            conversation_history = conversation_history[-max_history_length:]
            prompt_parts.append("Conversation history (recent):")
        else:
            prompt_parts.append("Conversation history:")
        prompt_parts.append(conversation_history)
        prompt_parts.append("")
    
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("\nPlease answer the question based on the information above. If you cannot find the answer from the memories, please respond with 'I don't know'.")
    
    prompt = "\n".join(prompt_parts)
    
    try:
        # Use LLM client from Graph Memory to generate answer
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on retrieved memories. Be concise and accurate."},
            {"role": "user", "content": prompt}
        ]
        
        response = graph_memory.llm_client.invoke(
            messages=messages,
            enable_thinking=False,
        )
        
        return response.content.strip() if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"Failed to generate answer: {e}")
        return f"Error generating answer: {str(e)}"



def evaluate_result(
    output_graphmemory: List[Dict[str, Any]],
    reference_items: List[LongMemEvalItem],
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate the accuracy of generated answers using officially defined evaluation standards.
    
    Args:
        output_graphmemory: List of generated answers, each containing:
            - question_id: ID of the question
            - hypothesis: Generated answer
        reference_items: Reference data list (LongMemEvalItem)
        output_file: Path to output result file
        verbose: Whether to print detailed information
        
    Returns:
        Dict[str, Any]: Evaluation results, including:
            - accuracy: Overall accuracy
            - by_type: Accuracy grouped by question type
            - logs: Detailed evaluation logs
    """
    print("\n" + "=" * 80)
    print("Process info: Evaluating results using official standards")
    print("=" * 80)
    
    # Initialize evaluator client
    metric_client, metric_model = init_evaluator_client()
    
    # Map question_id to reference data
    qid2qdata = {item.id: item for item in reference_items}
    qid2qtype = {item.id: item.question_type for item in reference_items}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}
    
    # Evaluate each generated answer
    logs = []
    for entry in tqdm(output_graphmemory, desc="Evaluating"):
        question_id = entry.get('question_id')
        if not question_id:
            print(f'Warning: skipping entry without question_id: {entry}')
            continue
        
        if question_id not in qid2qtype:
            print(f'Warning: skipping {question_id} as it is not in reference data.')
            continue
        
        qtype = qid2qtype[question_id]
        q = qid2qdata[question_id].question
        ans = qid2qdata[question_id].answer
        hyp = entry.get('hypothesis', '')
        
        is_abstention = '_abs' in question_id
        
        prompt = evaluate_qa.get_anscheck_prompt(qtype, q, ans, hyp, abstention=is_abstention)
        
        # Call evaluation model
        kwargs = {
            'model': metric_model,
            'messages': [
                {"role": "user", "content": prompt}
            ],
            'n': 1,
            'temperature': 0,
            'max_tokens': 10
        }
        
        try:
            completion = evaluate_qa.chat_completions_with_backoff(metric_client, **kwargs)
            eval_response = completion.choices[0].message.content.strip()
            label = 'yes' in eval_response.lower()
        except Exception as e:
            print(f'Error evaluating {question_id}: {e}')
            label = False
            eval_response = f"Error: {str(e)}"
        
        # Record evaluation result
        entry['autoeval_label'] = {
            'model': metric_model,
            'label': label
        }
        logs.append(entry)
        
        # Record accuracy by question type
        qtype2acc[qtype].append(1 if label else 0)
        
        # Print details
        if verbose:
            print(json.dumps({
                'question_id': question_id,
                'question': q,
                'answer': ans,
                'hypothesis': hyp,
                'autoeval_label': label
            }, indent=4, ensure_ascii=False), flush=True)
    
    # Calculate overall accuracy
    overall_accuracy = round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4)
    
    # Calculate accuracy by type
    by_type_accuracy = {}
    for k, v in qtype2acc.items():
        if len(v) > 0:
            by_type_accuracy[k] = {
                'accuracy': round(np.mean(v), 4),
                'count': len(v)
            }
    
    # Print evaluation results
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f'Overall Accuracy: {overall_accuracy}')
    print('\nAccuracy by question type:')
    for k, v in by_type_accuracy.items():
        print(f'\t{k}: {v["accuracy"]} ({v["count"]})')
    
    # Save results to file
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for entry in logs:
                print(json.dumps(entry, ensure_ascii=False), file=out_f)
        print(f'\nResults saved to: {output_file}')
    
    # Return evaluation results
    return {
        'accuracy': overall_accuracy,
        'by_type': by_type_accuracy,
        'logs': logs,
        'total_count': len(logs)
    }


def process_single_item(
    item: LongMemEvalItem,
    graph_memory: GraphMemory,
    use_conversation: bool,
    search_top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str
) -> Dict[str, Any]:
    """
    Processing flow for a single test item: Ingestion -> Retrieval -> Generation
    """
    # 1. Store conversation history for the question (item.id as Key)
    if use_conversation:
        store_conversations_to_memory(
            graph_memory=graph_memory,
            item=item,
            user_id=item.id,
            use_conversation=use_conversation,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy
        )
    
    # 2. Retrieve for the question
    search_results = search_memories(
        graph_memory=graph_memory,
        query=item.question,
        user_id=item.id,
        search_top_k=search_top_k
    )
    retrieved_memories = format_retrieved_memories(search_results)
    
    # 3. Format context conversation
    conversation_history = None
    if use_conversation and item.conversations:
        recent_turns = item.conversations[-10:] if len(item.conversations) > 10 else item.conversations
        conversation_history = "\n".join([
            f"{getattr(turn, 'role', 'user')}: {getattr(turn, 'content', '')}" for turn in recent_turns
        ])
    
    # 4. Generate answer
    answer = generate_answer(
        graph_memory=graph_memory,
        question=item.question,
        retrieved_memories=retrieved_memories,
        conversation_history=conversation_history,
        max_history_length=3000
    )
    
    return {
        'question_id': item.id,
        'question': item.question,
        'answer': item.answer,
        'question_type': item.question_type,
        'hypothesis': answer,
        'retrieved_memories': retrieved_memories,
    }


def main():
    """
    Main function: Parse arguments, load data
    """
    parser = argparse.ArgumentParser(description="LongMemEval Graph Memory Evaluation Script (New Data Format)")
    
    # Data related parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Data file path (new format, single JSON file, e.g., longmemeval_oracle.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of test cases (for fast testing)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for parallel processing (default: 4)")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Number of conversation turns per chunk (default: 8)")
    parser.add_argument("--chunk_overlap", type=int, default=2,
                        help="Number of overlapping turns between chunks (default: 2)")
    parser.add_argument("--chunk_strategy", type=str, default="session",
                        choices=["fixed", "session"],
                        help="Chunking strategy: fixed or session (default: session)")
    parser.add_argument("--db_name", type=str, default="longmemeval_evaluation",
                        help="Milvus collection name (default: longmemeval_evaluation)")
    
    args = parser.parse_args()
    OUTPUT_DIR = "./eval_results"
    
    # Storage path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    STORAGE_PATH = os.path.join(script_dir, "result")
    # Ensure directory exists
    os.makedirs(STORAGE_PATH, exist_ok=True)
    
    LANGUAGE = "en"
    
    # Graph Memory LLM configuration: read from .env file
    GRAPH_LLM_API_BASE = os.getenv("JIUWEN_GRAPH_MEM_LLM_URL")
    GRAPH_LLM_API_KEY = os.getenv("JIUWEN_GRAPH_MEM_LLM_KEY")
    GRAPH_LLM_MODEL_NAME = os.getenv("JIUWEN_GRAPH_MEM_LLM_MODEL")
    
    # Milvus configuration: read from .env file
    LOCAL_ENDPOINT = os.getenv("LOCAL_ENDPOINT")
    
    # Evaluator configuration: fixed to official standard
    EVALUATOR_MODEL = "gpt-4o"
    EVALUATOR_API_BASE = os.getenv("EVALUATOR_API_BASE")
    EVALUATOR_API_KEY = os.getenv("EVALUATOR_API_KEY")
    
    # Other configurations
    USE_CONVERSATION = True  # Use conversation history
    SEARCH_TOP_K = 5  # Retrieval top_k
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    CHUNK_STRATEGY = args.chunk_strategy
    NUM_WORKERS = args.num_workers
    
    # Set environment variables
    os.environ.setdefault("LLM_SSL_VERIFY", "false")
    
    # Print configuration information
    print("=" * 80)
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Storage Path: {STORAGE_PATH}")
    print(f"  Graph Memory Language: {LANGUAGE}")
    print(f"  Evaluator Model: {EVALUATOR_MODEL} (Official Standard)")
    print(f"  Use Conversation History: {USE_CONVERSATION}")
    print(f"  Retrieval Top K: {SEARCH_TOP_K}")
    print(f"  Conversation Chunk Strategy: {CHUNK_STRATEGY}")
    print(f"  Conversation Chunk Size: {CHUNK_SIZE}")
    print(f"  Conversation Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"  Number of Workers: {NUM_WORKERS}")
    print("=" * 80)
    
    # ==================== Step 1: Load Data ====================
    print("\n" + "=" * 80)
    print("Process info: Loading data")
    print("=" * 80)
    
    try:
        items = load_data(args.data_path, limit=args.limit)
    except Exception as e:
        print(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Data loaded successfully!")
    print("=" * 80)
    print(f"\n Prepare to evaluate {len(items)} test cases")
    print(f" Results will be saved to {OUTPUT_DIR}")
    
    # ==================== Step 2: Initialize Graph Memory ====================
    print("\n" + "=" * 80)
    print("Process info: Initializing graph memory")
    print("=" * 80)
    
    try:
        graph_memory = init_graph_memory(db_name=args.db_name, language=LANGUAGE)
    except Exception as e:
        print(f"Failed to initialize graph memory: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== Step 3 & 4: Process items parallelly (Store -> Retrieve -> Generate) ====================
    print("\n" + "=" * 80)
    print(f"Process info: Processing Items Parallelly (Workers: {NUM_WORKERS})")
    print("=" * 80)
    
    output_results = []
    
    if NUM_WORKERS > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_single_item,
                    item,
                    graph_memory,
                    USE_CONVERSATION,
                    SEARCH_TOP_K,
                    CHUNK_SIZE,
                    CHUNK_OVERLAP,
                    CHUNK_STRATEGY
                ) for item in items
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating Items"):
                try:
                    result = future.result()
                    output_results.append(result)
                except Exception as e:
                    print(f"\nError processing item: {e}")
    else:
        for item in tqdm(items, desc="Evaluating Items"):
            result = process_single_item(
                item,
                graph_memory,
                USE_CONVERSATION,
                SEARCH_TOP_K,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
                CHUNK_STRATEGY
            )
            output_results.append(result)
    
    print(f"\nSuccessfully generated answers for {len(output_results)} questions")
    
    # ==================== Step 5: Evaluate Results ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f"results_{timestamp}.json")
    metrics_file = os.path.join(OUTPUT_DIR, f"metrics_{timestamp}.json")
    
    eval_results = evaluate_result(
        output_graphmemory=output_results,
        reference_items=items,
        output_file=results_file,
        verbose=False
    )
    
    # Save evaluation metrics
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_accuracy': eval_results['accuracy'],
            'by_type': eval_results['by_type'],
            'total_count': eval_results['total_count'],
            'config': {
                'use_conversation': USE_CONVERSATION,
                'search_top_k': SEARCH_TOP_K,
                'language': LANGUAGE,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP,
                'chunk_strategy': CHUNK_STRATEGY,
                'num_workers': NUM_WORKERS,
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n Evaluation completed!")
    print(f"   Results saved to: {results_file}")
    print(f"   Metrics saved to: {metrics_file}")



if __name__ == "__main__":
    main()

