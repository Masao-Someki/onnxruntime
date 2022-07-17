import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, espnet_onnx_domain
from onnx import onnx_pb as onnx_proto
'''
    Quantize RelPos Attention
'''


class RelPosAttentionQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        '''
            parameter node: CrossAttention node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized CrossAttention node.
        '''
        node = self.node
        assert (node.op_type == "RelPosAttention")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0, 1, 3, 4] ,reduce_range=True, op_level_per_channel=True)
        if quantized_input_names is None:
            return super().quantize()

        qattention_name = "" if node.name == "" else node.name + "_quant"

        inputs = []
        inputs.extend(quantized_input_names) # input, weights, pos_emb, pos_weights
        inputs.extend([node.input[4], node.input[5], node.input[6]]) # bias, bias_u, bias_v
        inputs.extend(scale_names) # input_scale, weight_scale, pos_scale, pos_weight_scale
        inputs.extend(zero_point_names) # zero points

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = espnet_onnx_domain
        qattention_node = onnx.helper.make_node("QRelPosAttention", inputs, node.output, qattention_name, **kwargs)
        nodes.append(qattention_node)

        self.quantizer.new_nodes += nodes
