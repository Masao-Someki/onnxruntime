# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modification by Masao Someki
# Copyright (c) 2022 Masao Someki

import sys
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union

import numpy as np
from onnx import NodeProto, helper
from onnxruntime.transformers.fusion_base import Fusion
from onnxruntime.transformers.onnx_model import OnnxModel

from fusion_attention import AttentionMask

logger = getLogger(__name__)


class FusionRelativeShift(Fusion):
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, "RelativeShift", ["LayerNormalization"])

    def create_relshift_node(
        self,
        is_legacy: bool,
        input: str,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an CrossAttention node.
        Args:
            input (str): input name
            output (str): output name
        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        relshift_node_name = self.model.create_node_name("RelativeShift")
        relshift_node = helper.make_node(
            "RelativeShift",
            inputs=input,
            outputs=[output],
            name=relshift_node_name,
        )
        relshift_node.domain = "espnet_onnx"
        if is_legacy:
            relshift_node.attribute.extend([helper.make_attribute("legacy", 1)])

        return relshift_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        ## Search Conformer
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
                start_node,
                ["Mul", "Add", "MatMul", "Reshape", "Transpose", "MatMul"], # Conformer has ff_scale
                [1, None, None, None, 0, 0],
            )
        if qkv_nodes is not None:
            (_, _, matmul_qkv, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            return

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_cross_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        # root input
        # root_nodes = self.model.match_parent_path(matmul_v, ["LayerNormalization"], [None])
        # if root_nodes is None:
        #     return
        # root_input = root_nodes[0].output[0]

        qk_nodes = self.model.match_parent_path(matmul_qkv,
            ["Softmax","Add", "Div", "Add", "MatMul"], [0, 0, None, 0, 0])
        if qk_nodes is None:
            return
        (_, add_qk, _, add_bias_uv, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Add", "Reshape", "Add", "MatMul"],
            [0, 0, 0, None, None]
        ) # bias_u
        if q_nodes is None:
            return

        (_, bias_u, reshape_q, add_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if k_nodes is None:
            return
        (_, _, add_k, matmul_k) = k_nodes

        # find bias_v from
        # latest version
        is_legacy = False

        # torch 1.11
        pos_v_matmul_nodes = self.model.match_parent_path(
            add_bias_uv,
            ["Slice", "Unsqueeze", "Add", "Div", "Gather", "Shape", "MatMul"],
            [1, 2, 0, None, None, None, None])

        # torch 1.12
        if pos_v_matmul_nodes is None:
            pos_v_matmul_nodes = self.model.match_parent_path(
                add_bias_uv,
                ["Slice", "Unsqueeze", "Add", "Div", "Squeeze", "Slice", "Shape", "MatMul"],
                [1, 2, 0, None, None, None, None, None])

        if pos_v_matmul_nodes is None:
            # check if model is legacy version
            pos_v_matmul_nodes = self.model.match_parent_path(
                add_bias_uv,
                ["Slice", "Reshape", "Concat", "MatMul"],
                [1, 0, 0, 1])
            if pos_v_matmul_nodes is not None:
                is_legacy = True

        if pos_v_matmul_nodes is None:
            return

        pos_v_matmul = pos_v_matmul_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        if matmul_v.input[0] == matmul_k.input[0]:
            rel_shift_last_node = add_bias_uv

            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            new_node = self.create_relshift_node(
                is_legacy,
                [matmul_qk.output[0], pos_v_matmul.output[0]],
                rel_shift_last_node.output[0],
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([rel_shift_last_node])
            self.nodes_to_remove.extend(pos_v_matmul_nodes[:-1])

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            # self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True
