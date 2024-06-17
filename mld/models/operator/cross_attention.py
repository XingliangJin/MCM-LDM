# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional
from numpy import block

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1
        #_get_clones 会按照num_block克隆
        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src   ####(2, bz, 256)

        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)     ###(2,bs,512)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x


# 825_mclip_xyz_noloss_fea_concat        
class SkipTransformerEncoder_concat(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1
        #_get_clones 会按照num_block克隆
        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, style,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src   ####(2, bz, 256)
        style = style

        xs = []
        for module in self.input_blocks:
            x = module(x, style, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, style, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)     ###(2,bs,512)
            x = linear(x)
            x = module(x, style, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x

#添加上style的skiptransformer 
class SkipTransformerEncoder_style(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(3*self.d_model, self.d_model), num_block)
        self.linear_blocks_in = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)
        #self.linear_blocks_mid = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    #加上style条件
    def forward(self, src, sty,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src
        y = sty.repeat(x.shape[0],1,1)

        xs = []
        for (module, linear) in zip(self.input_blocks, self.linear_blocks_in):
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)
            x = torch.cat([x, y], dim=-1)
            x = linear(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop(), y], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x


# 903_mclipxyz_noloss_noroot_feaconcat_last       
class SkipTransformerEncoder_concat_last(nn.Module):
    def __init__(self, encoder_layer, encoder_layer1, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1
        #_get_clones 会按照num_block克隆
        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer1, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, style,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src   ####(2, bz, 256)
        style = style

        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)     ###(2,bs,512)
            x = linear(x)
            x = module(x, style, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x

#添加上style的skiptransformer 
class SkipTransformerEncoder_style(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(3*self.d_model, self.d_model), num_block)
        self.linear_blocks_in = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)
        #self.linear_blocks_mid = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    #加上style条件
    def forward(self, src, sty,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src
        y = sty.repeat(x.shape[0],1,1)

        xs = []
        for (module, linear) in zip(self.input_blocks, self.linear_blocks_in):
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)
            x = torch.cat([x, y], dim=-1)
            x = linear(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop(), y], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x
    



#7-9  two_diff_c2s 

#for content gate
class ContentGate(nn.Module):
    def __init__(self, input_dim, style_dim, text_encoded_dim: int = 768):
        super(ContentGate, self).__init__()
        self.latent_dim = 256
        self.linear = nn.Linear(style_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, style):
        gate = self.sigmoid(self.linear(style))
        return x + gate * style

class SkipTransformerEncoder_c2s_content(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1
        #_get_clones 会按照num_block克隆
        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src   ####(2, bz, 256)

        xs = []
        x_out = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)
            x_out.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        x_out.append(x)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)     ###(2,bs,512)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            x_out.append(x)
            

        if self.norm is not None:
            x = self.norm(x)
        return x, x_out


class SkipTransformerEncoder_c2s_style(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        self.style_residual = ContentGate(256,256)

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.num = num_block
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.input_style = _get_clones(self.style_residual, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.middle_style = _get_clone(self.style_residual)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.output_style = _get_clones(self.style_residual, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)
        #self.linear_blocks_mid = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    #加上content交互过来的信息
    def forward(self, src, con,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src
        y = con

        xs = []
        for (module, residual) in zip(self.input_blocks, self.input_style):
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            x = residual(x,y[0])
            y = y[1:]
            xs.append(x)
            
            

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        x = self.middle_style(x,y[0])
        y = y[1:]

        for (module, linear, residual, i) in zip(self.output_blocks, self.linear_blocks, self.output_style, range(self.num_layers)):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if i != self.num-1:
                x = residual(x,y[0])
                y = y[1:]

        if self.norm is not None:
            x = self.norm(x)
        return x








#for gate
class StyleGate(nn.Module):
    def __init__(self, input_dim, style_dim, text_encoded_dim: int = 768):
        super(StyleGate, self).__init__()
        self.latent_dim = 256
        self.linear = nn.Linear(style_dim, input_dim)
        self.emb_proj_st = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, style):
        style = self.emb_proj_st(style)
        gate = self.sigmoid(self.linear(style))
        return x + gate * style


class SkipTransformerEncoder_style_gate(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        self.style_residual = StyleGate(256,256)

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.input_style = _get_clones(self.style_residual, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.output_style = _get_clones(self.style_residual, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)
        #self.linear_blocks_mid = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    #加上style条件
    def forward(self, src, sty,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src
        y = sty.repeat(x.shape[0],1,1)

        xs = []
        for (module, residual) in zip(self.input_blocks, self.input_style):
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)
            x = residual(x,y)
            

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear, residual) in zip(self.output_blocks, self.linear_blocks, self.output_style):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = residual(x,y)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x


class SkipTransformerEncoder_style1(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.style_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(3*self.d_model, self.d_model), num_block)
        self.linear_blocks_in = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)
        #self.linear_blocks_mid = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    #加上style条件
    def forward(self, src, sty,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src
        y = sty.repeat(x.shape[0],1,1)
        y = self.style_block(y)

        xs = []
        for (module, linear) in zip(self.input_blocks, self.linear_blocks_in):
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)
            x = torch.cat([x, y], dim=-1)
            x = linear(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop(), y], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x

class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = decoder_layer.d_model
        
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        x = tgt

        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



# 添加adain的transformer层
# 添加adain的transformer层
# 添加adain的transformer层
# 添加adain的transformer层


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)  # batch size & channels
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class TransformerEncoderLayer_adain(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adain = AdaptiveInstanceNorm1d(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
# 走的流程是这个

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # self attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # add + norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # adain, 只对sample作用
        # src_sample = src[0,...].unsqueeze(0)
        # src_sample = self.adain(src_sample.permute(1,2,0)).permute(2,0,1)
        # src = torch.cat([src_sample, src[1:,...]], dim=0)

        # adain
        src = self.adain(src.permute(1,2,0)).permute(2,0,1)

        # FFN
        src3 = self.linear4(self.dropout(self.activation(self.linear3(src))))

        # add
        src = src + self.dropout3(src3)


        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# 824_mclip_xyz_noloss_add_adain
class TransformerEncoderLayer_add_adain(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adain = AdaptiveInstanceNorm1d(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
# 走的流程是这个

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # self attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # add + norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # adain
        src3 = self.adain(src.permute(1,2,0)).permute(2,0,1)

        # FFN
        src = self.linear4(self.dropout(self.activation(self.linear3(src3))))

        # add
        src = src + self.dropout3(src3)


        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)




# 825_mclip_xyz_noloss_fea_concat
class TransformerEncoderLayer_concat(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.linear5 = nn.Linear(2*d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adain = AdaptiveInstanceNorm1d(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
# 走的流程是这个

    def forward_post(self,
                     src,
                     style,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # self attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # add + norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # print("style")
        # print(style.shape)
        # concat
        style = style.repeat(src.shape[0], 1, 1)
        # print("style1")
        # print(style.shape)
        src = torch.cat((src, style), dim=-1)
        # print("src")
        # print(src.shape)
        src = self.linear5(src)
        # print("src1")
        # print(src.shape)

        # FFN
        src3 = self.linear4(self.dropout(self.activation(self.linear3(src))))

        # add
        src = src + self.dropout3(src3)


        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,style,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src,style, src_mask, src_key_padding_mask, pos)


# 927 feaconcat_add
class TransformerEncoderLayer_concat_add(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.linear5 = nn.Linear(2*d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adain = AdaptiveInstanceNorm1d(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
# 走的流程是这个

    def forward_post(self,
                     src,
                     style,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # self attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # add + norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # print("style")
        # print(style.shape)
        # concat
        style = style.repeat(src.shape[0], 1, 1)
        fus = torch.cat((src, style), dim=-1)
        fus = self.linear5(fus)
        # print(fus.shape)
        # add
        scr = src + torch.sigmoid(fus)
        # print(src.shape)


        # FFN
        src3 = self.linear4(self.dropout(self.activation(self.linear3(src))))

        # add
        src = src + self.dropout3(src3)


        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,style,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src,style, src_mask, src_key_padding_mask, pos)



# 830_mlcipxyz_noloss_noroot_croatt

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, content, style):
        # MultiheadAttention需要输入的形状为（sequence_length, batch_size, feature_dim）
        # content作为query，style作为key和value
        output, _ = self.multihead_attention(content, style, style)
        return output

class TransformerEncoderLayer_croatt(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.self_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        # self.linear5 = nn.Linear(2*d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.adain = AdaptiveInstanceNorm1d(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
# 走的流程是这个

    def forward_post(self,
                     src,
                     style,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # self attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # add + norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # cross attention
        style = style.repeat(src.shape[0], 1, 1)

        src3= self.self_cross_attn(src, style, style)[0]
        # add
        src = src + self.dropout3(src3)
        src = self.norm3(src)


        # FFN
        src4 = self.linear4(self.dropout5(self.activation(self.linear3(src))))

        # add
        src = src + self.dropout4(src4)


        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,style,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src,style, src_mask, src_key_padding_mask, pos)

# 903_mlcipxyz_noloss_noroot_atn

# Attention module from motionpuzzle
class adaptive_style(nn.Module):
    def __init__(self, in_ch):
        super(adaptive_style, self).__init__()
        self.in_ch = in_ch
        self.f = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.g = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.h = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.k = nn.Conv2d(in_ch, in_ch, (1, 1))
    
    def forward(self, x, s_sty, return_nl_map=False):
        r"""
            x: (n, c, t1, v)
            s_sty: (n, c, t2, v)
        """
        b = s_sty.shape[0]

        F = self.f(nn.functional.instance_norm(x)) 
        G = self.g(nn.functional.instance_norm(s_sty))
        H = self.h(s_sty)

        F = F.view(b, self.in_ch, -1).permute(0, 2, 1)
        G = G.view(b, self.in_ch, -1)
        S = torch.bmm(F, G)

        S = self.sm(S)
        H = H.view(b, self.in_ch, -1)
        O = torch.bmm(H, S.permute(0, 2, 1))

        O = O.view(x.size())

        O = self.k(O)
        O += x
        
        if return_nl_map:
            return O, S
        return O
# Attention module ours
class atn_style(nn.Module):
    def __init__(self, in_ch):
        super(atn_style, self).__init__()
        self.in_ch = in_ch
        self.f = nn.Conv1d(in_ch, in_ch, 1)
        self.g = nn.Conv1d(in_ch, in_ch, 1)
        self.h = nn.Conv1d(in_ch, in_ch, 1)
        self.sm = nn.Softmax(dim=-1)
        self.k = nn.Conv1d(in_ch, in_ch, 1)
    
    def forward(self, x, s_sty, return_nl_map=False):
        r"""
            x: (seq, bs, fea)
            s_sty: (seq, bs, fea)

            (bs, fea, )
        """
        b = s_sty.shape[1]
        # 第一步：permute一下符合网络输入形状（n, c, l）即batchsize, 维度, 长度(多少个数据)
        F = x.permute(1, 2, 0)
        G = s_sty.permute(1, 2, 0)
        s_sty = s_sty.permute(1, 2, 0)

        # 第二步，使用IN和conv1d编码
        F = nn.functional.instance_norm(F)
        G = nn.functional.instance_norm(G)
        F = self.f(F) 
        G = self.g(G)
        H = self.h(s_sty)

        # 第三步，style和src相乘获得attention map
        F = F.permute(0, 2, 1)
        S = torch.bmm(F, G)
        S = self.sm(S)

        # 第四步，style和attention map相乘
        O = torch.bmm(H, S.permute(0, 2, 1))

        # 第五步，加上src的残差
        O = O.permute(0, 2, 1)
        O = self.k(O).permute(2, 0, 1)
        O += x
        
        if return_nl_map:
            return O, S
        return O

class TransformerEncoderLayer_atn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_cross_attn = atn_style(d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        # self.linear5 = nn.Linear(2*d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.adain = AdaptiveInstanceNorm1d(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
# 走的流程是这个

    def forward_post(self,
                     src,
                     style,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # self attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # add + norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # atn style(3, bs, 256) src(3, bs, 256)
        style = style.repeat(src.shape[0], 1, 1)

        src3= self.self_cross_attn(src, style)
        # add
        src = src + self.dropout3(src3)
        src = self.norm3(src)


        # FFN
        src4 = self.linear4(self.dropout5(self.activation(self.linear3(src))))

        # add
        src = src + self.dropout4(src4)


        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,style,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src,style, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
                     
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clone(module):
    return copy.deepcopy(module)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")