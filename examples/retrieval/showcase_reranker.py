"""Example script demonstrating reranker usage with Standard reranker"""

from openjiuwen.core.retrieval import RerankerConfig, StandardReranker

# Query and documents for reranking (feel free to edit)
QUERY = "Hello"
DOCUMENTS = ["Hi", "ALoha", "bonjour"]

# Standard reranker config (provide your own)
STANDARD_RERANKER_CONFIG = RerankerConfig(
    model="reranker-model", api_base="http://your-embedding-service/v1", api_key="your-api-key"
)


def main():
    """Main example demonstrating reranker usage"""
    standard_reranker = StandardReranker(STANDARD_RERANKER_CONFIG, verify=False)
    standard_result = standard_reranker.rerank_sync(query=QUERY, doc=DOCUMENTS)
    print(f"Query: {QUERY}")
    print(f"Documents: {DOCUMENTS}")
    print(f"Reranked result: {standard_result}")


if __name__ == "__main__":
    main()
