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
import time
import os
from typing import Dict, List, Optional, Tuple

import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional as F

torch.manual_seed(0)

class Config:
    batch_size = 0
    sequence_length = 0
    legacy = 0
    num_heads = 0
    head_size = 0
    embed_dim = 0
    
    def __init__(self, b, s, leg, n, h):
        self.batch_size = b
        self.sequence_length = s
        self.legacy = leg
        self.num_heads = n
        self.head_size = h
        self.embed_dim = self.num_heads * self.head_size


class AttentionForONNX(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        device=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.device = device
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim**-0.5
        self.encoder_decoder_attention = True
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.p_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"
        self.pos_bias_u = torch.randn(self.num_heads, self.head_dim).to(device)
        self.pos_bias_v = torch.randn(self.num_heads, self.head_dim).to(device)
        
    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1)).to(self.device)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]
        return x
    
    def legacy_rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1)).to(self.device)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        return x
    
    def forward(
        self,
        input,
        pos_emb: Tensor,
        legacy: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Batch x Time(SeqLen) x Channel"""
        bsz, tgt_len, embed_dim = input.size()
        # get here for encoder decoder cause of static_kv
        q = self.q_proj(input).reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(input).reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(input).reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        p = self.p_proj(pos_emb).reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = q.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        
        if legacy:
            matrix_bd = self.legacy_rel_shift(matrix_bd)
        else:
            matrix_bd = self.rel_shift(matrix_bd)
            
        score = (matrix_ac + matrix_bd) / math.sqrt(self.head_dim)
        attn = torch.softmax(score, dim=-1)
        attn_output = torch.matmul(attn, v)
        
        return attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
    
    def ORT_forward(
        self,
        input,
        pos_emb: Tensor,
        legacy: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Batch x Time(SeqLen) x Channel"""
        weight = torch.cat(
            (
                self.q_proj.weight.T,
                self.k_proj.weight.T,
                self.v_proj.weight.T,
            ),
            dim=-1,
        )
        pos_weights = self.p_proj.weight.T
        
        bias = torch.stack(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias),
            dim=0,
        )
        bias = bias.reshape(3 * self.embed_dim)
        
        onnx_model_str = create_relpos_attention_graph(
            input,
            pos_emb,
            weight,
            pos_weights,
            bias,
            self.pos_bias_u,
            self.pos_bias_v,
            self.num_heads,
            legacy,
        )
        ort_inputs = {
            "input": numpy.ascontiguousarray(input.cpu().numpy()),
            "pos_emb": numpy.ascontiguousarray(pos_emb.cpu().numpy()),
        }
        from onnxruntime import InferenceSession, SessionOptions
        sess_options = SessionOptions()
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CUDAExecutionProvider"])
        ort_output = ort_session.run(None, ort_inputs)[0]
        start_time = time.time()
        ort_output = ort_session.run(None, ort_inputs)[0]
        end_time = time.time() - start_time
        return ort_output, end_time


def create_relpos_attention_graph(
    input,
    pos_emb,
    weights,
    pos_weight,
    bias,
    pos_bias_u,
    pos_bias_v,
    num_heads_,
    is_legacy,
):
    from onnx import TensorProto, helper
    B, S, NH = input.size()
    S2 = pos_emb.size()[1]
    N = num_heads_
    nodes = [
        helper.make_node(
            "RelPosAttention",
            [
                "input",
                "weights",
                "pos_emb",
                "pos_weight",
                "bias",
                "pos_bias_u",
                "pos_bias_v"
            ],
            ["output"],
            "RelPosAttention_0",
            num_heads=num_heads_,
            legacy=1 if is_legacy else 0,
            domain="espnet_onnx",
        ),
    ]
    initializers = [
        helper.make_tensor("weights", TensorProto.FLOAT, [NH, 3 * NH], weights.flatten().tolist()),
        helper.make_tensor("pos_weight", TensorProto.FLOAT, [NH, NH], pos_weight.flatten().tolist()),
        helper.make_tensor("bias", TensorProto.FLOAT, [3 * NH], bias.flatten().tolist()),
        helper.make_tensor("pos_bias_u", TensorProto.FLOAT, [N, NH // N], pos_bias_u.flatten().tolist()),
        helper.make_tensor("pos_bias_v", TensorProto.FLOAT, [N, NH // N], pos_bias_v.flatten().tolist()),
    ]
    graph = helper.make_graph(
        nodes,
        "RelPosAttention_Graph",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [B, S, NH]),
            helper.make_tensor_value_info("pos_emb", TensorProto.FLOAT, [B, S2, NH]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [B, S, NH]),
        ],
        initializers,
    )
    model = helper.make_model(graph)
    return model.SerializeToString()


def create_inputs(
    config: Config,
    device,
):
    input = torch.normal(
        mean=0.0,
        std=0.1,
        size=(config.batch_size, config.sequence_length, config.embed_dim),
    ).to(device)
    if config.legacy:
        pos_emb = torch.normal(
            mean=0.0,
            std=0.1,
            size=(config.batch_size, config.sequence_length, config.embed_dim),
        ).to(device)
    else:
        pos_emb = torch.normal(
            mean=0.0,
            std=0.1,
            size=(config.batch_size, 2 * config.sequence_length - 1, config.embed_dim),
        ).to(device)

    return input, pos_emb


def parity_check(
    config,
    rtol=1e-4,
    atol=5e-4,
    device=None,
):
    query, key = create_inputs(config, device)
    attn = AttentionForONNX(config.embed_dim, config.num_heads, device=device)
    attn.to(device)
    attn_output = attn.forward(
        query,
        key,
        config.legacy,
    )
    start_time = time.time()
    attn_output = attn.forward(
        query,
        key,
        config.legacy,
    )
    torch_time = time.time() - start_time
    attn_output_ort, onnx_time = attn.ORT_forward(
        query,
        key,
        config.legacy,
    )
    attn_output_ort_1, _ = attn.ORT_forward(
        query,
        key,
        config.legacy,
    )
    print(
        " B:",
        config.batch_size,
        " S:",
        config.sequence_length,
        " legacy:",
        config.legacy,
        " h:",
        config.embed_dim,
        # "[attn_output, randomness] parity:",
        # numpy.allclose(
        #     attn_output.cpu().detach().numpy(),
        #     attn_output_ort,
        #     rtol=rtol,
        #     atol=atol,
        #     equal_nan=True,
        # ),
        # numpy.allclose(
        #     attn_output_ort_1,
        #     attn_output_ort,
        #     rtol=rtol,
        #     atol=atol,
        #     equal_nan=True,
        # ),
        mse(attn_output.cpu().detach().numpy(), attn_output_ort),
        mse(attn_output_ort_1, attn_output_ort),
        "speed up:%.5f" % (torch_time / onnx_time)
    )

def mse(a, b):
    return ((a - b) ** 2).mean()

if __name__ == "__main__":
    for b in [1]:
        for s in [1, 32, 256]:
            for leg in [0, 1]:
                for n in [4, 8]:
                    for h in [32, 64]:
                        config = Config(b, s, leg, n, h)
                        parity_check(
                            config,
                            device='cuda'
                        )
