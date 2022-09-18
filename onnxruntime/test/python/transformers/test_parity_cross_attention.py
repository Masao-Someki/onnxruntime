# --------------------------------------------------------------------------
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
# Modification by Masao Someki
# Copyright (c) 2022 Masao Someki

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import time

torch.manual_seed(0)

class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0
    num_heads = 0
    head_size = 0
    embed_dim = 0
    
    def __init__(self, b, s, s2, n, h):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.num_heads = n
        self.head_size = h
        self.embed_dim = self.num_heads * self.head_size


class AttentionProjection(nn.Module):
    def __init__(self, num_heads, head_dim, embed_dim, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def shape_state(self, state, batch_size):
        return state.view(batch_size * self.num_heads, -1, self.head_dim)
    
    def shape_proj(self, proj, batch_size):
        return proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(
        self,
        query,
        key,
    ):
        bsz = torch._shape_as_tensor(query)[0]
        k = self.k_proj(key)
        v = self.v_proj(key)
        k = self.shape_proj(k, bsz)
        v = self.shape_proj(v, bsz)
        return k, v


class AttentionForONNX(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        device='cpu'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.encoder_decoder_attention = True
        self.k_v_proj = AttentionProjection(num_heads, self.head_dim, embed_dim, bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"
        self.device = device
        
    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(
        self,
        query,
        key: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Batch x Time(SeqLen) x Channel"""
        bsz, tgt_len, embed_dim = query.size()
        # get here for encoder decoder cause of static_kv
        k, v = self.k_v_proj(query, key)
        q = self.q_proj(query)
        q = self._shape(q, tgt_len, bsz)
        src_len = key.size(1)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        assert attn_weights.size() == (bsz, self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        assert v is not None
        attn_output = torch.matmul(attn_weights, v)
        return attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
    
    def ORT_forward(
        self,
        query,
        key: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Batch x Time(SeqLen) x Channel"""
        q_weight = self.q_proj.weight.T
        kv_weight = torch.cat(
            (
                self.k_v_proj.k_proj.weight.T,
                self.k_v_proj.v_proj.weight.T,
            ),
            dim=-1,
        )
        q_bias = self.q_proj.bias
        kv_bias = torch.stack(
            (self.k_v_proj.k_proj.bias, self.k_v_proj.v_proj.bias),
            dim=0,
        )
        kv_bias = kv_bias.reshape(2 * self.embed_dim)
        onnx_model_str = create_cross_attention_graph(
            query,
            key,
            q_weight,
            kv_weight,
            q_bias,
            kv_bias,
            self.num_heads,
        )
        ort_inputs = {
            "query": numpy.ascontiguousarray(query.cpu().numpy()),
            "key": numpy.ascontiguousarray(key.cpu().numpy()),
        }
        from onnxruntime import InferenceSession, SessionOptions
        sess_options = SessionOptions()
        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider']
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=providers)
        ort_output = ort_session.run(None, ort_inputs)[0]
        ort_output = ort_session.run(None, ort_inputs)[0]
        start_time = time.time()
        ort_output = ort_session.run(None, ort_inputs)[0]
        end_time = time.time()
        return ort_output, end_time - start_time


def create_cross_attention_graph(
    query,
    key,
    q_weight,
    kv_weight,
    q_bias,
    kv_bias,
    num_heads_,
):
    from onnx import TensorProto, helper
    B, S, NH = query.size()
    S2 = key.size()[1]
    N = num_heads_
    nodes = [
        helper.make_node(
            "CrossAttention",
            [
                "query",
                "key",
                "q_weight",
                "kv_weight",
                "q_bias",
                "kv_bias"
            ],
            ["output"],
            "CrossAttention_0",
            num_heads=num_heads_,
            domain="espnet_onnx",
        ),
    ]
    initializers = [
        helper.make_tensor("q_weight", TensorProto.FLOAT, [NH, NH], q_weight.flatten().tolist()),
        helper.make_tensor("kv_weight", TensorProto.FLOAT, [NH, 2 * NH], kv_weight.flatten().tolist()),
        helper.make_tensor("q_bias", TensorProto.FLOAT, [NH], q_bias.flatten().tolist()),
        helper.make_tensor("kv_bias", TensorProto.FLOAT, [2 * NH], kv_bias.flatten().tolist()),
    ]
    graph = helper.make_graph(
        nodes,
        "CrossAttention_Graph",
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, [B, S, NH]),
            helper.make_tensor_value_info("key", TensorProto.FLOAT, [B, S2, NH]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [B, S, NH]),
        ],
        initializers,
    )
    model = helper.make_model(graph)
    return model.SerializeToString()


def create_inputs(
    config: Config, device: str,
):
    query = torch.normal(
        mean=0.0,
        std=0.1,
        size=(config.batch_size, config.sequence_length, config.embed_dim),
    ).type(torch.float32).to(device)
    key = torch.normal(
        mean=0.0,
        std=0.1,
        size=(config.batch_size, config.kv_sequence_length, config.embed_dim),
    ).type(torch.float32).to(device)

    return query, key


def parity_check(
    config,
    device='cpu',
    rtol=1e-4,
    atol=5e-4,
):
    query, key = create_inputs(config, device)
    attn = AttentionForONNX(config.embed_dim, config.num_heads, device)
    attn.to(device)
    attn_output = attn.forward(
        query,
        key,
    )
    start_time = time.time()
    attn_output = attn.forward(
        query,
        key,
    )
    torch_time = time.time() - start_time
    attn_output_ort, onnx_time = attn.ORT_forward(
        query,
        key,
    )
    attn_output_ort_1, _ = attn.ORT_forward(
        query,
        key,
    )
    print(
        " B:",
        config.batch_size,
        " S:",
        config.sequence_length,
        " S*:",
        config.kv_sequence_length,
        " h:",
        config.embed_dim,
        "[attn_output, randomness] parity:",
        numpy.allclose(
            attn_output.cpu().detach().numpy(),
            attn_output_ort,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
        numpy.allclose(
            attn_output_ort_1,
            attn_output_ort,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
        mse(attn_output.cpu().detach().numpy(), attn_output_ort),
        mse(attn_output_ort_1, attn_output_ort),
        "speed up:%.5f" % (torch_time / onnx_time),
        "device:", d
    )

def mse(a, b):
    return ((a - b) ** 2).mean()

if __name__ == "__main__":
    # for d in ['cpu', 'cuda']:
    for d in ['cuda']:
        for b in [1]:
            for s in [32, 256]:
                for s2 in [256]:
                    for n in [4, 8]:
                        for h in [64,128]:
                            config = Config(b, s, s2, n, h)
                            parity_check(
                                config,
                                device=d
                            )
