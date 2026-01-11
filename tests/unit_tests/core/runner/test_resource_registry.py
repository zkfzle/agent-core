from concurrent.futures import ThreadPoolExecutor, as_completed
from openjiuwen.core.runner.resources_manager.resource_registry import ResourceRegistry


def access_all_resources():
    registry = ResourceRegistry()
    return [
        registry.workflow(),
        registry.tool(),
        registry.prompt(),
        registry.model(),
        registry.agent(),
        registry.agent_group(),
    ]


def test_resource_registry_concurrent():

    num_threads = 10

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(access_all_resources) for _ in range(num_threads)]

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception occurred: {e}")
                assert False, f"Exception in concurrent access: {e}"

    for i in range(1, len(results)):
        for j in range(len(results[i])):
            assert results[i][j] is results[0][j], f"Resource at index {j} is not singleton across threads"

    print("All resources are thread-safe and singleton across threads.")
