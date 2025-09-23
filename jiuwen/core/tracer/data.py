from enum import Enum
class InvokeType(Enum):
    """
    Agent Invoke Type
    """
    PROMPT = "prompt"
    LLM = "llm"
    PLUGIN = "plugin"
    WORKFLOW = "workflow"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    EVALUATOR = "evalutor"
    
class NodeStatus(Enum):
    """
    Workflow Node Status For Message
    """
    START = "start"
    FINISH = "finish"
    RUNNING = "running"
    ERROR = "error"