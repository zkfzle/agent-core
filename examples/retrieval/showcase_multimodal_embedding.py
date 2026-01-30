"""Test to verify that images significantly affect embedded vectors"""

import asyncio
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from openjiuwen.core.retrieval.common.config import EmbeddingConfig
from openjiuwen.core.retrieval.common.document import MultimodalDocument
from openjiuwen.core.retrieval.embedding.vllm_embedding import VLLMEmbedding

# Text section of documents (feel free to edit)
REFERENCE_TEXT = "A photograph of a person"
DIFFERENT_TEXT = "Picture of an octopus in ocean"

# Image file paths (supply your own images)
REF_IMAGE = Path("reference.jpg")
SAME_IMAGE_DIFFERENT_CONTENT = Path("reference.ppm")
DIFFERENT_IMAGE = Path("different.ppm")

# Embedding model config (provide your own)
EMBEDDING_CONFIG = EmbeddingConfig(
    model_name="multimodal-embedding-model", base_url="http://your-embedding-service/v1", api_key="your-key"
)
EMBEDDING_DIM = 128  # Set to None to use default dimension


def cosine_similarity(vec1: ArrayLike, vec2: ArrayLike) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot_product / (norm1 * norm2))


def euclidean_distance(vec1: ArrayLike, vec2: ArrayLike) -> float:
    """Calculate euclidean distance between two vectors"""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    return float(np.linalg.norm(vec1 - vec2))


async def main():
    """Main test"""
    # Create documents with different images
    # doc1 and doc2 have the same image but different formats
    # doc3 has a different image
    # doc4 has alternative text
    docs = [MultimodalDocument() for _ in range(4)]
    docs[0].add_field(
        "text",
        REFERENCE_TEXT,
    ).add_field(
        "image",
        file_path=REF_IMAGE,
    )
    docs[1].add_field(
        "text",
        REFERENCE_TEXT,
    ).add_field(
        "image",
        file_path=SAME_IMAGE_DIFFERENT_CONTENT,
    )
    docs[2].add_field(
        "text",
        REFERENCE_TEXT,
    ).add_field(
        "image",
        file_path=DIFFERENT_IMAGE,
    )
    docs[3].add_field(
        "text",
        DIFFERENT_TEXT,
    ).add_field(
        "image",
        file_path=REF_IMAGE,
    )

    # Initialize embedding model
    model = VLLMEmbedding(EMBEDDING_CONFIG, dimension=EMBEDDING_DIM, timeout=10)

    # Generate embeddings
    print("Generating embeddings...")
    emb1, emb2, emb3, emb4 = await asyncio.gather(*(model.embed_multimodal(doc) for doc in docs))

    print(f"Embedding dimensions: {len(emb1)}")

    # Compare embeddings
    # doc1 vs doc2: Same image, different format - should be similar
    sim_1_2 = cosine_similarity(emb1, emb2)
    dist_1_2 = euclidean_distance(emb1, emb2)
    print("\ndoc1 (sophie.jpg) vs doc2 (sophie.ppm):")
    print(f"  Cosine similarity: {sim_1_2:.4f}")
    print(f"  Euclidean distance: {dist_1_2:.4f}")

    # doc1 vs doc3: Different images - should be different
    sim_1_3 = cosine_similarity(emb1, emb3)
    dist_1_3 = euclidean_distance(emb1, emb3)
    print("\ndoc1 (sophie.jpg) vs doc3 (BillGates.ppm):")
    print(f"  Cosine similarity: {sim_1_3:.4f}")
    print(f"  Euclidean distance: {dist_1_3:.4f}")

    # doc2 vs doc3: Different images - should be different
    sim_2_3 = cosine_similarity(emb2, emb3)
    dist_2_3 = euclidean_distance(emb2, emb3)
    print("\ndoc2 (sophie.ppm) vs doc3 (BillGates.ppm):")
    print(f"  Cosine similarity: {sim_2_3:.4f}")
    print(f"  Euclidean distance: {dist_2_3:.4f}")

    # doc1 vs doc4: Same image, different text - tests text influence
    sim_1_4 = cosine_similarity(emb1, emb4)
    dist_1_4 = euclidean_distance(emb1, emb4)
    print("\ndoc1 (sophie.jpg + person text) vs doc4 (sophie.jpg + octopus text):")
    print(f"  Cosine similarity: {sim_1_4:.4f}")
    print(f"  Euclidean distance: {dist_1_4:.4f}")

    # doc4 vs doc2: Same image (different format), different text
    sim_4_2 = cosine_similarity(emb4, emb2)
    dist_4_2 = euclidean_distance(emb4, emb2)
    print("\ndoc4 (sophie.jpg + octopus text) vs doc2 (sophie.ppm + person text):")
    print(f"  Cosine similarity: {sim_4_2:.4f}")
    print(f"  Euclidean distance: {dist_4_2:.4f}")

    # doc4 vs doc3: Different image, different text
    sim_4_3 = cosine_similarity(emb4, emb3)
    dist_4_3 = euclidean_distance(emb4, emb3)
    print("\ndoc4 (sophie.jpg + octopus text) vs doc3 (BillGates.ppm + person text):")
    print(f"  Cosine similarity: {sim_4_3:.4f}")
    print(f"  Euclidean distance: {dist_4_3:.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)

    # Same image (different format) should be more similar than different images
    if sim_1_2 > sim_1_3 and sim_1_2 > sim_2_3:
        print("✓ PASS: Same image (different format) produces more similar embeddings")
        print(f"  Similarity between same image: {sim_1_2:.4f}")
        print(f"  Similarity between different images: {max(sim_1_3, sim_2_3):.4f}")
    else:
        print("✗ FAIL: Same image embeddings are not more similar than different images")
        print(f"  Similarity between same image: {sim_1_2:.4f}")
        print(f"  Similarity between different images: {max(sim_1_3, sim_2_3):.4f}")

    # Different images should have low similarity (typically < 0.9 for different images)
    if sim_1_3 < 0.9 and sim_2_3 < 0.9:
        print("✓ PASS: Different images produce significantly different embeddings")
        print(f"  Similarity between different images: {max(sim_1_3, sim_2_3):.4f} < 0.9")
    else:
        print("✗ WARNING: Different images may not be sufficiently different")
        print(f"  Similarity between different images: {max(sim_1_3, sim_2_3):.4f}")

    # Euclidean distance: same image should have smaller distance
    if dist_1_2 < dist_1_3 and dist_1_2 < dist_2_3:
        print("✓ PASS: Same image (different format) has smaller euclidean distance")
        print(f"  Distance between same image: {dist_1_2:.4f}")
        print(f"  Distance between different images: {min(dist_1_3, dist_2_3):.4f}")
    else:
        print("✗ FAIL: Same image does not have smaller euclidean distance")

    # Text influence analysis: same image with different text
    print("\n" + "-" * 60)
    print("Text Influence Analysis (same image, different text):")
    print("-" * 60)
    if sim_1_4 > sim_1_3 and sim_1_4 > sim_2_3:
        print("✓ PASS: Same image with different text is more similar than different images")
        print(f"  Similarity (same image, different text): {sim_1_4:.4f}")
        print(f"  Similarity (different images): {max(sim_1_3, sim_2_3):.4f}")
    else:
        print("✗ WARNING: Text may have strong influence, or image similarity is low")
        print(f"  Similarity (same image, different text): {sim_1_4:.4f}")
        print(f"  Similarity (different images): {max(sim_1_3, sim_2_3):.4f}")

    # Compare same image with same text vs different text
    if sim_1_2 > sim_1_4:
        print("✓ PASS: Same image with same text is more similar than same image with different text")
        print(f"  Similarity (same image, same text): {sim_1_2:.4f}")
        print(f"  Similarity (same image, different text): {sim_1_4:.4f}")
        print(f"  Text influence ratio: {sim_1_4 / sim_1_2:.4f}")
    else:
        print("✗ WARNING: Text may have minimal influence on embeddings")
        print(f"  Similarity (same image, same text): {sim_1_2:.4f}")
        print(f"  Similarity (same image, different text): {sim_1_4:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Images significantly affect embeddings: {sim_1_2 > sim_1_3 and sim_1_2 > sim_2_3}")
    print(f"Same image, same text similarity: {sim_1_2:.4f}")
    print(f"Same image, different text similarity: {sim_1_4:.4f}")
    print(f"Different images similarity: {max(sim_1_3, sim_2_3):.4f}")
    print(f"Image difference ratio: {max(sim_1_3, sim_2_3) / sim_1_2:.4f}")
    if sim_1_2 > sim_1_4:
        print(f"Text influence ratio: {sim_1_4 / sim_1_2:.4f} (lower = more text influence)")


if __name__ == "__main__":
    asyncio.run(main())
