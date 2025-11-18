#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
import json
from typing import Optional, List, Dict

from .models import Workflow, Position, NodeType
from .converter_utils import ConverterUtils
from .converters import StartConverter, EndConverter, LLMConverter, IntentDetectionConverter, QuestionerConverter, \
                        CodeConverter, PluginConverter, OutputConverter, BranchConverter
from .simpleir_to_mermaid import SimpleIrToMermaid


class DLTransformer:
    _dsl_converter_registry = {
        NodeType.Start.dl_type: StartConverter,
        NodeType.End.dl_type: EndConverter,
        NodeType.LLM.dl_type: LLMConverter,
        NodeType.IntentDetection.dl_type: IntentDetectionConverter,
        NodeType.Questioner.dl_type: QuestionerConverter,
        NodeType.Code.dl_type: CodeConverter,
        NodeType.Plugin.dl_type: PluginConverter,
        NodeType.Output.dl_type: OutputConverter,
        NodeType.Branch.dl_type: BranchConverter,
    }

    def transform_to_mermaid(self, dl_content: List[Dict]) -> str:
        mermaid_result = SimpleIrToMermaid.transform_to_mermaid(dl_content)
        return mermaid_result

    def transform_to_dsl(self, dl_content: str, resource: Optional[dict] = None) -> str:
        nodes = json.loads(dl_content)
        nodes_dict = {node["id"]: node for node in nodes}
        workflow = Workflow()
        x, y = 0, 0
        for node in nodes:
            converter_class = self._dsl_converter_registry.get(node["type"])
            if not converter_class:
                pass # raise JiuwenBaseException not support node_type

            if node["type"] in [NodeType.Plugin.dl_type]:
                node_converter = converter_class(node, nodes_dict, resource=resource, position=Position(x, y))
            else:
                node_converter = converter_class(node, nodes_dict, position=Position(x, y))
            node_converter.convert()
            workflow.nodes.append(node_converter.node)
            workflow.edges.extend(node_converter.edges)
            x += 20
            y += 20
        return json.dumps(ConverterUtils.convert_to_dict(workflow), ensure_ascii=False)
