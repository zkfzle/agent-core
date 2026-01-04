#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Backward Compatibility Tests

Ensure old interfaces continue working and issue proper deprecation warnings
"""
import warnings
import pytest


class TestLegacyImports:
    """Test old import paths"""
    
    def test_all_old_imports_work(self):
        """All old imports still work"""
        try:
            from openjiuwen.core.single_agent import (
                AgentConfig,
                ControllerAgent,
                AgentSession,
                WorkflowFactory,
                workflow_provider,
                create_react_agent_config,
                LLMCallConfig,
                ConstrainConfig,
            )
        except ImportError as e:
            pytest.fail(f"Old imports failed: {e}")
    
    def test_new_imports_work(self):
        """New imports also work"""
        try:
            from openjiuwen.core.single_agent import (
                AgentCard,
                BaseAgent,
                ReActAgent,
                ReActAgentConfig,
            )
        except ImportError as e:
            pytest.fail(f"New imports failed: {e}")
    
    def test_legacy_mixin_import(self):
        """LegacyMethodsMixin can be imported"""
        try:
            from openjiuwen.core.single_agent import LegacyMethodsMixin
            assert LegacyMethodsMixin is not None
        except ImportError as e:
            pytest.fail(f"LegacyMethodsMixin import failed: {e}")


class TestLegacyConstructor:
    """Test old constructor methods"""
    
    def test_react_agent_old_style_construction(self):
        """ReActAgent old construction style still works"""
        from openjiuwen.core.single_agent import ReActAgent, ReActAgentConfig
        from openjiuwen.core.foundation.llm import ModelConfig, BaseModelInfo
        
        # Create model config
        model_info = BaseModelInfo(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        model_config = ModelConfig(
            model_provider="openai",
            model_info=model_info
        )
        
        # Create agent config
        config = ReActAgentConfig(
            id="test_agent",
            version="1.0",
            description="Test Agent",
            model=model_config
        )
        
        # Create agent - should work without errors
        agent = ReActAgent(agent_config=config)
        
        # Verify agent created successfully
        assert agent.agent_config.id == "test_agent"
        assert agent.agent_config.version == "1.0"
    
    def test_react_agent_with_tools_parameter(self):
        """Support old tools parameter"""
        from openjiuwen.core.single_agent import ReActAgent, ReActAgentConfig
        from openjiuwen.core.foundation.llm import ModelConfig, BaseModelInfo
        
        # Create model config
        model_info = BaseModelInfo(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        model_config = ModelConfig(
            model_provider="openai",
            model_info=model_info
        )
        
        # Create agent config
        config = ReActAgentConfig(
            id="test_agent",
            version="1.0",
            model=model_config
        )
        
        # Create agent with tools parameter (should not raise error)
        agent = ReActAgent(agent_config=config, tools=[])
        assert agent is not None


class TestLegacyMethods:
    """Test legacy methods"""
    
    def test_add_tools_method_issues_warning(self):
        """add_tools() method still works but issues warning"""
        from openjiuwen.core.single_agent import ReActAgent, ReActAgentConfig
        from openjiuwen.core.foundation.llm import ModelConfig, BaseModelInfo
        
        # Create agent
        model_info = BaseModelInfo(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        model_config = ModelConfig(
            model_provider="openai",
            model_info=model_info
        )
        config = ReActAgentConfig(
            id="test_agent",
            version="1.0",
            model=model_config
        )
        agent = ReActAgent(agent_config=config)
        
        # Call add_tools and check for deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = agent.add_tools([])
            
            # Verify deprecation warning was issued
            assert len(w) > 0
            assert any("deprecated" in str(x.message).lower() for x in w)
            assert any("add_ability" in str(x.message) for x in w)
            # Verify returns self for chaining
            assert result is agent
    
    def test_add_workflows_method_issues_warning(self):
        """add_workflows() method still works but issues warning"""
        from openjiuwen.core.single_agent import ReActAgent, ReActAgentConfig
        from openjiuwen.core.foundation.llm import ModelConfig, BaseModelInfo
        
        # Create agent
        model_info = BaseModelInfo(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        model_config = ModelConfig(
            model_provider="openai",
            model_info=model_info
        )
        config = ReActAgentConfig(
            id="test_agent",
            version="1.0",
            model=model_config
        )
        agent = ReActAgent(agent_config=config)
        
        # Call add_workflows and check for deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = agent.add_workflows([])
            
            assert len(w) > 0
            assert any("deprecated" in str(x.message).lower() for x in w)
            assert any("add_ability" in str(x.message) for x in w)
            assert result is agent


class TestWarningMessages:
    """Test warning messages correctness"""
    
    def test_deprecation_warning_contains_migration_info(self):
        """Deprecation warnings contain migration information"""
        from openjiuwen.core.single_agent import ReActAgent, ReActAgentConfig
        from openjiuwen.core.foundation.llm import ModelConfig, BaseModelInfo
        
        # Create agent
        model_info = BaseModelInfo(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        model_config = ModelConfig(
            model_provider="openai",
            model_info=model_info
        )
        config = ReActAgentConfig(
            id="test_agent",
            version="1.0",
            model=model_config
        )
        agent = ReActAgent(agent_config=config)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent.add_tools([])
            
            assert len(w) > 0
            warning_msg = str(w[0].message)
            # Verify contains version number
            assert "1.0.0" in warning_msg or "v1.0" in warning_msg
            # Verify warning message is issued
            assert "deprecated" in warning_msg.lower()


class TestCreateReactAgentConfig:
    """Test create_react_agent_config factory function"""
    
    def test_create_react_agent_config_issues_warning(self):
        """create_react_agent_config() issues deprecation warning"""
        from openjiuwen.core.single_agent import create_react_agent_config
        from openjiuwen.core.foundation.llm import ModelConfig, BaseModelInfo
        
        model_info = BaseModelInfo(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        model_config = ModelConfig(
            model_provider="openai",
            model_info=model_info
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = create_react_agent_config(
                agent_id="test",
                agent_version="1.0",
                description="test",
                model=model_config,
                prompt_template=[]
            )
            
            # Verify deprecation warning
            assert len(w) > 0
            assert any("deprecated" in str(x.message).lower() for x in w)
            assert any("ReActAgentConfig" in str(x.message) for x in w)
            # Verify config created successfully
            assert config.id == "test"
            assert config.version == "1.0"


class TestLegacyCompatibilityIntegration:
    """Integration tests for legacy compatibility"""
    
    def test_old_and_new_apis_coexist(self):
        """Old and new APIs can coexist"""
        from openjiuwen.core.single_agent import (
            ReActAgent,
            ReActAgentConfig,
            AgentCard,
            BaseAgent
        )
        from openjiuwen.core.foundation.llm import ModelConfig, BaseModelInfo
        
        # Create model config
        model_info = BaseModelInfo(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        model_config = ModelConfig(
            model_provider="openai",
            model_info=model_info
        )
        
        # Old style agent
        old_config = ReActAgentConfig(
            id="old_agent",
            version="1.0",
            model=model_config
        )
        old_agent = ReActAgent(agent_config=old_config)
        
        # Verify old agent is created successfully
        assert old_agent is not None
        # Old agent uses legacy BaseAgent, not the new one
        from openjiuwen.core.single_agent.legacy import LegacyBaseAgent
        assert isinstance(old_agent, LegacyBaseAgent)

