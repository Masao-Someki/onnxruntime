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
from fusion_attention import AttentionMask
from onnx import NodeProto, TensorProto, helper, numpy_helper

from onnxruntime.transformers.fusion_base import Fusion
from onnxruntime.transformers.fusion_options import AttentionMaskFormat
from onnxruntime.transformers.fusion_utils import FusionUtils, NumpyHelper
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime.transformers.shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto

logger = getLogger(__name__)


class FusionRelPosAttention(Fusion):
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(model, "RelPosAttention", ["LayerNormalization", "SkipLayerNormalization"])
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_mask = attention_mask

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.
        Args:
            reshape_q (NodeProto): reshape node for Q
        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape = self.model.get_initializer(reshape_q.input[1])
        if q_shape is None:
            logger.debug(f"{reshape_q.input[1]} is not initializer.")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        q_shape_value = NumpyHelper.to_array(q_shape)
        if len(q_shape_value) != 4 or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0):
            logger.debug(f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, head_size].")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def get_add_qk_str(self, add_qk: NodeProto):
        shape_infer = self.model.infer_runtime_shape(update=True)
        if shape_infer is None:
            return

        input_0_shape = shape_infer.get_edge_shape(add_qk.input[0])
        input_1_shape = shape_infer.get_edge_shape(add_qk.input[1])

        if input_0_shape is None or input_1_shape is None:
            logger.debug(f"one of the inputs of {add_qk} is None")
            return None

        if input_0_shape != input_1_shape:
            logger.debug(f"the shape of two inputs of {add_qk} is not same")
            return None

        return add_qk.input[1]

    def create_attention_node(
        self,
        mask_index: str,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        pos_matmul: NodeProto,
        q_add: NodeProto,
        k_add: NodeProto,
        v_add: NodeProto,
        pos_u_add: NodeProto,
        pos_v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        is_legacy: bool,
        input: str,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an CrossAttention node.
        Args:
            mask_index (str): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for  K
            v_matmul (NodeProto): MatMul node in fully connection for  V
            q_add (NodeProto): Add bias node in fully connection for Q
            k_add (NodeProto): Add bias node in fully connection for K
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name
        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        _q_weight = self.model.get_initializer(q_matmul.input[1])
        _k_weight = self.model.get_initializer(k_matmul.input[1])
        _v_weight = self.model.get_initializer(v_matmul.input[1])
        _p_weight = self.model.get_initializer(pos_matmul.input[1])
        _q_bias = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
        _k_bias = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
        _v_bias = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])
        _pos_u_bias = self.model.get_initializer(pos_u_add.input[1]) or self.model.get_initializer(pos_u_add.input[0])
        _pos_v_bias = self.model.get_initializer(pos_v_add.input[1]) or self.model.get_initializer(pos_v_add.input[0])

        if _q_weight is None:
            print(f"{q_matmul.input[1]} is not initializer. Please set do_constant_folding=True in torch.onnx.export")
            return None
        if not (_k_weight and _v_weight and _q_bias and _k_bias):
            return None

        qw = NumpyHelper.to_array(_q_weight)
        kw = NumpyHelper.to_array(_k_weight)
        vw = NumpyHelper.to_array(_v_weight)
        pw = NumpyHelper.to_array(_p_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]
        pw_in_size = pw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size == pw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size {hidden_size} is not same as weight matrix dimension of q,k,v paths {qw_in_size}, provide correct input hidden size or pass 0"
            )

        # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
        # For 2d weights, the shapes would be [in_size, out_size].
        # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
        qw_out_size = np.prod(qw.shape[1:])
        kw_out_size = np.prod(kw.shape[1:])
        vw_out_size = np.prod(vw.shape[1:])
        pw_out_size = np.prod(pw.shape[1:])

        qkvw = np.concatenate((qw, kw, vw), axis=1)
        qkv_weight_dim = qw_out_size + kw_out_size + vw_out_size

        qb = NumpyHelper.to_array(_q_bias)
        kb = NumpyHelper.to_array(_k_bias)
        vb = NumpyHelper.to_array(_v_bias)
        pub = NumpyHelper.to_array(_pos_u_bias)
        pvb = NumpyHelper.to_array(_pos_v_bias)

        q_bias_shape = np.prod(qb.shape)
        k_bias_shape = np.prod(kb.shape)
        v_bias_shape = np.prod(vb.shape)

        assert q_bias_shape == k_bias_shape == qw_out_size == pw_out_size
        assert v_bias_shape == vw_out_size

        qkvb = np.concatenate((qb, kb, vb), axis=0)
        qkv_bias_dim = q_bias_shape + k_bias_shape + v_bias_shape

        attention_node_name = self.model.create_node_name("RelPosAttention")

        weight = helper.make_tensor(
            name=attention_node_name + "_weight",
            data_type=TensorProto.FLOAT,
            dims=[qw_in_size, qkv_weight_dim],
            vals=qkvw.flatten().tolist(),
        )
        pos_weight = helper.make_tensor(
            name=attention_node_name + "_pos_weight",
            data_type=TensorProto.FLOAT,
            dims=[pw_in_size, pw_out_size],
            vals=pw.flatten().tolist(),
        )

        # Sometimes weights and bias are stored in fp16
        if weight.data_type == 10:
            weight.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(weight).astype(np.float16), weight.name))
            pos_weight.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(pos_weight).astype(np.float16), pos_weight.name))
        self.model.add_initializer(weight, self.this_graph_name)
        self.model.add_initializer(pos_weight, self.this_graph_name)

        bias = helper.make_tensor(
            name=attention_node_name + "_bias",
            data_type=TensorProto.FLOAT,
            dims=[qkv_bias_dim],
            vals=qkvb.flatten().tolist(),
        )
        head_size = hidden_size // num_heads
        pos_bias_u = helper.make_tensor(
            name=attention_node_name + "_pu_bias",
            data_type=TensorProto.FLOAT,
            dims=[num_heads, head_size],
            vals=pub.flatten().tolist(),
        )
        pos_bias_v = helper.make_tensor(
            name=attention_node_name + "_pv_bias",
            data_type=TensorProto.FLOAT,
            dims=[num_heads, head_size],
            vals=pvb.flatten().tolist(),
        )
        if bias.data_type == 10:
            bias.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(bias).astype(np.float16), bias.name))
            pos_bias_u.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(pos_bias_u).astype(np.float16), pos_bias_u.name))
            pos_bias_v.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(pos_bias_v).astype(np.float16), pos_bias_v.name))
        self.model.add_initializer(bias, self.this_graph_name)
        self.model.add_initializer(pos_bias_u, self.this_graph_name)
        self.model.add_initializer(pos_bias_v, self.this_graph_name)

        attention_inputs = [input[0]] + [
            attention_node_name + "_weight"] + [input[1]] + [
            attention_node_name + "_pos_weight"] + [
            attention_node_name + "_bias",
            attention_node_name + "_pu_bias",
            attention_node_name + "_pv_bias",
        ]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        attention_node = helper.make_node(
            "RelPosAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "espnet_onnx"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        if is_legacy:
            attention_node.attribute.extend([helper.make_attribute("legacy", 1)])

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        import sys
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

        if qkv_nodes is None:
            qkv_nodes = self.model.match_parent_path(
                    start_node,
                    ["Add", "MatMul", "Reshape", "Transpose", "MatMul"], # Conformer has ff_scale
                    [1, 1, 0, 0, 0],
                )
            if qkv_nodes is not None:
                (_, matmul_qkv, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        if qkv_nodes is None:
            return

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_cross_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        # root input
        root_node = self.model.match_parent(matmul_v, "LayerNormalization", 0)
        if root_node is None:
            root_node = self.model.match_parent(matmul_v, "SkipLayerNormalization", 0)

        if root_node is None:
            return
        root_input = root_node.output[0]

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
        pos_v_matmul_nodes = self.model.match_parent_path(
            add_bias_uv,
            ["Slice", "Unsqueeze", "Add", "Div", "Gather", "Shape", "MatMul"],
            [1, 2, 0, None, None, None, None])

        # Latest relative shift for torch 2.0.1
        if pos_v_matmul_nodes is None:
            pos_v_matmul_nodes = self.model.match_parent_path(
                add_bias_uv,
                ["Slice", "Unsqueeze", "Add", "Div", "Squeeze", "Slice", "Shape", "MatMul"],
                [1, 2, 0, 0, 0, 0, 0, 0])

        if pos_v_matmul_nodes is None:
            # check if model is legacy version
            pos_v_matmul_nodes = self.model.match_parent_path(
                add_bias_uv,
                ["Slice", "Reshape", "Concat", "MatMul"],
                [1, 0, 0, 1])
            if pos_v_matmul_nodes is not None:
                is_legacy = True

        # torch 2.0.1
        # It seems that torch.2.0.1 cannot export Slice -> Add properly,
        # we need to trace the model from q_reshape.
        if pos_v_matmul_nodes is None:
            q_reshape = self.model.match_parent_path(
                matmul_qk, ["Transpose", "Add", "Reshape"], [0, 0, 0])[-1]
            qv_add = input_name_to_nodes[q_reshape.output[0]][0]
            relshift_subgraph = self.model.get_children_subgraph_nodes(
                qv_add, [add_bias_uv]
            )
            slice_node = [r for r in relshift_subgraph if r.op_type == "Slice"][0]
            pos_v_matmul_nodes = self.model.match_parent_path(
                slice_node,
                ["Reshape", "Concat", "MatMul"],
                [0, 0, 1])
            pos_v_matmul_nodes = [slice_node] + pos_v_matmul_nodes
            is_legacy = True

        if pos_v_matmul_nodes is None:
            return

        pos_v_matmul = pos_v_matmul_nodes[-1]

        bias_v_nodes = self.model.match_parent_path(
            pos_v_matmul,
            ["Transpose", "Add"],
            [0, None])
        if bias_v_nodes is None:
            return
        bias_v = bias_v_nodes[-1]

        pos_emb_nodes = self.model.match_parent_path(
            pos_v_matmul,
            ["Transpose", "Reshape", "MatMul", "Slice"],
            [1, None, 0, None])

        if pos_emb_nodes is None:
            return
        (_, _, matmul_pos, pos_emb) = pos_emb_nodes

        mask_index = None
        _, *mask_nodes, _ = self.model.match_parent_path(
            add_qk, ['Mul', 'Sub', 'Unsqueeze', 'Unsqueeze'],
            [None, 0, 1, 0])
        if mask_nodes is None:
            return
        mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        if matmul_v.input[0] == matmul_k.input[0]:
            attention_last_node = reshape_qkv

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            new_node = self.create_attention_node(
                mask_index,
                matmul_q,
                matmul_k,
                matmul_v,
                matmul_pos,
                add_q,
                add_k,
                add_v,
                bias_u,
                bias_v,
                q_num_heads,
                q_hidden_size,
                is_legacy,
                [root_input, pos_emb.output[0]],
                attention_last_node.output[0],
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)
            self.nodes_to_remove.extend(pos_v_matmul_nodes)
            self.nodes_to_remove.extend(bias_v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            # self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True
