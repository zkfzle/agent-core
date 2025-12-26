def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"