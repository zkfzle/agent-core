# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Triple data model test cases
"""
import pytest
from pydantic import ValidationError

from openjiuwen.core.retrieval.common.triple import Triple


class TestTriple:
    """Triple data model tests"""

    def test_create_triple(self):
        """Test creating triple"""
        triple = Triple(
            subject="Alice",
            predicate="knows",
            object="Bob",
        )
        assert triple.subject == "Alice"
        assert triple.predicate == "knows"
        assert triple.object == "Bob"
        assert triple.confidence is None
        assert triple.metadata == {}

    def test_create_triple_with_confidence(self):
        """Test creating triple with confidence"""
        triple = Triple(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.95,
        )
        assert triple.confidence == 0.95

    def test_create_triple_with_metadata(self):
        """Test creating triple with metadata"""
        metadata = {"source": "test", "doc_id": "doc_1"}
        triple = Triple(
            subject="Alice",
            predicate="knows",
            object="Bob",
            metadata=metadata,
        )
        assert triple.metadata == metadata

    def test_create_triple_with_all_fields(self):
        """Test creating triple with all fields"""
        metadata = {"source": "test"}
        triple = Triple(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.95,
            metadata=metadata,
        )
        assert triple.subject == "Alice"
        assert triple.predicate == "knows"
        assert triple.object == "Bob"
        assert triple.confidence == 0.95
        assert triple.metadata == metadata

    def test_missing_required_fields(self):
        """Test missing required fields"""
        with pytest.raises(ValidationError):
            Triple()
        
        with pytest.raises(ValidationError):
            Triple(subject="Alice")
        
        with pytest.raises(ValidationError):
            Triple(subject="Alice", predicate="knows")

