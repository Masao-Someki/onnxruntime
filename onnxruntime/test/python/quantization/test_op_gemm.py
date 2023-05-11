#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestOpGemm(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_gemm(self, output_model_path):
        #      (input)
        #         |
        #        Gemm
        #         |
        #        Clip
        #         |
        #        Gemm
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_gemm(input_name, weight_shape, weight_name, bias_shape, bias_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

            return onnx.helper.make_node(
                "Gemm",
                [input_name, weight_name, bias_name],
                [output_name],
                alpha=1.0,
                beta=1.0,
                transB=1,
            )

        # make gemm1 node
        gemm1_output_name = "gemm1_output"
        gemm1_node = make_gemm(
            input_name,
            [100, 10],
            "linear1.weight",
            [100],
            "linear1.bias",
            gemm1_output_name,
        )

        # make Clip
        clip_min_name = "clip_min"
        clip_max_name = "clip_max"
        clip_output_name = "clip_output"
        clip_inputs = [gemm1_output_name, clip_min_name, clip_max_name]
        clip_outputs = [clip_output_name]
        initializers.append(onnx.numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name=clip_min_name))
        initializers.append(onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=clip_max_name))
        clip_node = onnx.helper.make_node("Clip", clip_inputs, clip_outputs)

        # make gemm2 node
        gemm2_node = make_gemm(
            clip_output_name,
            [10, 100],
            "linear2.weight",
            [10],
            "linear2.bias",
            output_name,
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, 10])
        graph_name = "gemm_test"
        graph = helper.make_graph(
            [gemm1_node, clip_node, gemm2_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def construct_model_cross_attention(self, output_model_path):
        #      (query)  (key)
        #         |       |
        #         ----+----
        #             |
        #       CrossAttention
        #             |
        #           MatMul
        #             |
        #          (output)
        query_name = 'query'
        key_name = 'key'
        output_name = 'output'
        initializers = []

        def make_cross_attention_node(q_name, k_name, q_weight_shape, k_weight_shape, q_weight_name, k_weight_name,
                q_bias_shape, k_bias_shape, q_bias_name, k_bias_name, output_name):
            q_weight_data = np.random.normal(0, 0.1, q_weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(q_weight_data, name=q_weight_name))
            k_weight_data = np.random.normal(0, 0.1, k_weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(k_weight_data, name=k_weight_name))

            q_bias_data = np.random.normal(0, 0.1, q_bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(q_bias_data, name=q_bias_name))
            k_bias_data = np.random.normal(0, 0.1, k_bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(k_bias_data, name=k_bias_name))

            return onnx.helper.make_node('CrossAttention',
                [q_name, k_name, q_weight_name, k_weight_name, q_bias_name, k_bias_name], [output_name])

        def make_matmul_node(input_name, weight_shape, weight_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            return onnx.helper.make_node('MatMul', [input_name, weight_name], [output_name])

        # make cross attention node
        attention_output_name = "attention_output"
        cross_attention_node = make_cross_attention_node(
            query_name, key_name, [10, 10], [10, 20], 'q.weight', 'kv.weight',
            [10], [20], 'q.bias', 'kv.bias', attention_output_name)
        cross_attention_node.domain = "espnet_onnx"
        cross_attention_node.attribute.extend([helper.make_attribute("num_heads", 5)])

        # make matmul node
        matmul_node = make_matmul_node(attention_output_name, [10, 10], 'matmul.weight', output_name)

        # make graph
        query_tensor = helper.make_tensor_value_info(query_name, TensorProto.FLOAT, [1, -1, 10])
        key_tensor = helper.make_tensor_value_info(key_name, TensorProto.FLOAT, [1, -1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, -1, 10])

        graph_name = 'cross_attention_test'
        graph = helper.make_graph([cross_attention_node, matmul_node], graph_name,
                                  [query_tensor, key_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, producer_name='espnet_onnx', opset_imports=[
            helper.make_opsetid('espnet_onnx', 1), helper.make_opsetid('', 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)
    
        def construct_model_relpos_attention(self, output_model_path):
            #      (query) (pos_emb)
            #         |        |
            #         ----+----
            #             |
            #       RelPosAttention
            #             |
            #           MatMul
            #             |
            #          (output)
            input_name = 'input'
            pos_name = 'pos_emb'
            output_name = 'output'
            initializers = []

        def make_relpos_attention_node(i_name, p_name, i_weight_shape, p_weight_shape, i_weight_name, p_weight_name,
                i_bias_shape, i_bias_name, u_bias_shape, u_bias_name, v_bias_shape, v_bias_name, output_name):
            i_weight_data = np.random.normal(0, 0.1, i_weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(i_weight_data, name=i_weight_name))
            p_weight_data = np.random.normal(0, 0.1, p_weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(p_weight_data, name=p_weight_name))

            bias_data = np.random.normal(0, 0.1, i_bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=i_bias_name))
            u_bias_data = np.random.normal(0, 0.1, u_bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(u_bias_data, name=u_bias_name))
            v_bias_data = np.random.normal(0, 0.1, v_bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(v_bias_data, name=v_bias_name))

            return onnx.helper.make_node('RelPosAttention',
                [i_name, i_weight_name, p_name, p_weight_name, i_bias_name, u_bias_name, v_bias_name], [output_name])

        def make_matmul_node(input_name, weight_shape, weight_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            return onnx.helper.make_node('MatMul', [input_name, weight_name], [output_name])

        # make cross attention node
        attention_output_name = "attention_output"
        relpos_attention_node = make_relpos_attention_node(
            input_name, pos_name, [10, 30], [10, 10], 'input.weight', 'pos.weight',
            [30], 'input.bias', [2,5], 'pos_u.bias', [2,5], 'pos_v.bias', attention_output_name)
        relpos_attention_node.domain = "espnet_onnx"
        relpos_attention_node.attribute.extend([helper.make_attribute("num_heads", 2)])

        # make matmul node
        matmul_node = make_matmul_node(attention_output_name, [10, 10], 'matmul.weight', output_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, -1, 10])
        pos_tensor = helper.make_tensor_value_info(pos_name, TensorProto.FLOAT, [1, -1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, -1, 10])

        graph_name = 'relpos_attention_test'
        graph = helper.make_graph([relpos_attention_node, matmul_node], graph_name,
                                  [input_tensor, pos_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, producer_name='espnet_onnx', opset_imports=[
            helper.make_opsetid('espnet_onnx', 1), helper.make_opsetid('', 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def static_quant_test(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},
    ):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = "gemm_fp32.quant_{}{}.onnx".format(activation_type_str, weight_type_str)

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        qdq_count = 1 if activation_type == QuantType.QUInt8 else 2
        clip_count = 0 if activation_type == QuantType.QUInt8 else 1
        quant_nodes = {"QGemm": 2, "QuantizeLinear": qdq_count, "DequantizeLinear": qdq_count, "Clip": clip_count}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())

    def static_quant_test_qdq(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},
    ):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = "gemm_fp32.quant_dqd_{}{}.onnx".format(activation_type_str, weight_type_str)

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        clip_count = 0 if activation_type == QuantType.QUInt8 else 1
        q_count = 3 if activation_type == QuantType.QUInt8 else 4
        dq_count = 7 if activation_type == QuantType.QUInt8 else 8
        quant_nodes = {"Gemm": 2, "QuantizeLinear": q_count, "DequantizeLinear": dq_count, "Clip": clip_count}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())

    def dynamic_quant_test(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},
    ):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = "gemm_fp32.quant_dynamic_{}{}.onnx".format(activation_type_str, weight_type_str)

        quantize_dynamic(
            model_fp32_path,
            model_int8_path,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        quant_nodes = {"MatMulInteger": 2}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {"MatMulInteger": [["i", 2, activation_proto_qtype]]}
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(
            self,
            model_fp32_path,
            model_int8_path,
            {"input": np.random.rand(5, 10).astype(np.float32)},
        )
    
    def dynamic_cross_attention_quant_test(self, model_fp32_path, model_int8_path, per_channel, reduce_range):
        quantize_dynamic(model_fp32_path, model_int8_path, per_channel=per_channel, reduce_range=reduce_range)
        quant_nodes = {'QCrossAttention': 1, 'MatMulInteger': 1}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path,
                {'query': np.random.rand(1, 5, 10).astype(np.float32), 'key': np.random.rand(1, 5, 10).astype(np.float32)})

    def dynamic_relpos_attention_quant_test(self, model_fp32_path, model_int8_path, per_channel, reduce_range):
        quantize_dynamic(model_fp32_path, model_int8_path, per_channel=per_channel, reduce_range=reduce_range)
        quant_nodes = {'QRelPosAttention': 1, 'MatMulInteger': 1}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path,
                {'input': np.random.rand(1, 5, 10).astype(np.float32), 'pos_emb': np.random.rand(1, 9, 10).astype(np.float32)})

    def test_quantize_gemm(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )
        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )
        self.dynamic_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )

    def test_quantize_gemm_s8s8(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )
        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

        # dynamic quantization doesn't support activation:int8
        # self.dynamic_quant_test(model_fp32_path, data_reader, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        #                        extra_options={'ActivationSymmetric': True})

    def test_quantize_cross_attention(self):
        np.random.seed(1)
        model_fp32_path = 'cross_attention_fp32.onnx'
        model_int8_path = 'cross_attention_fp32.quant.onnx'
        self.construct_model_cross_attention(model_fp32_path)

        self.dynamic_cross_attention_quant_test(model_fp32_path, model_int8_path, True, True)
        self.dynamic_cross_attention_quant_test(model_fp32_path, model_int8_path, True, False)
        self.dynamic_cross_attention_quant_test(model_fp32_path, model_int8_path, False, True)
        self.dynamic_cross_attention_quant_test(model_fp32_path, model_int8_path, False, False)

    def test_quantize_relpos_attention(self):
        np.random.seed(1)
        model_fp32_path = 'relpos_attention_fp32.onnx'
        model_int8_path = 'relpos_attention_fp32.quant.onnx'
        self.construct_model_relpos_attention(model_fp32_path)

        self.dynamic_relpos_attention_quant_test(model_fp32_path, model_int8_path, True, True)
        self.dynamic_relpos_attention_quant_test(model_fp32_path, model_int8_path, True, False)
        self.dynamic_relpos_attention_quant_test(model_fp32_path, model_int8_path, False, True)
        self.dynamic_relpos_attention_quant_test(model_fp32_path, model_int8_path, False, False)


if __name__ == "__main__":
    unittest.main()
