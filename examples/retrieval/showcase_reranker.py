"""Example script demonstrating reranker usage with Standard reranker"""

import asyncio

from openjiuwen.core.retrieval import RerankerConfig, StandardReranker

# Query and documents for reranking (feel free to edit)
QUERY = "Hello"
DOCUMENTS = ["Hi", "Aloha", "bonjour"]
INSTRUCTION = "greeting in french"

# Standard reranker config (provide your own)
STANDARD_RERANKER_CONFIG = RerankerConfig(
    model="reranker-model", api_base="http://your-embedding-service/v1", api_key="your-api-key"
)


async def main():
    """Main example demonstrating reranker usage"""
    standard_reranker = StandardReranker(STANDARD_RERANKER_CONFIG, verify=False)
    rerank_req = []
    for instruction in [False, INSTRUCTION]:
        rerank_req.append(standard_reranker.rerank(query=QUERY, doc=DOCUMENTS, instruct=instruction))
    print(f"Query: {QUERY}")
    print(f"Documents: {DOCUMENTS}")
    no_instruct, instruct = await asyncio.gather(*rerank_req)
    print("Reranked result without instruction:")
    for doc, prob in no_instruct.items():
        print(f"  {doc:7s}: {prob}")
    print(f"Reranked result with {instruction=}:")
    for doc, prob in instruct.items():
        print(f"  {doc:7s}: {prob}")


if __name__ == "__main__":
    asyncio.run(main())
