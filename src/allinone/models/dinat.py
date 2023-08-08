"""This is a modification of:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinat/modeling_dinat.py
  so that it can provide both 1D and 2D attention.
"""

import math
import torch
from abc import ABC,  abstractmethod
from typing import Optional, Tuple, Callable
from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb
from ..config import Config
from .utils import *


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
  """
  Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

  Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
  however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
  See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
  layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
  argument.
  """
  if drop_prob == 0.0 or not training:
    return input
  keep_prob = 1 - drop_prob
  shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
  random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
  random_tensor.floor_()  # binarize
  output = input.div(keep_prob) * random_tensor
  return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Dinat
class DinatDropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
  
  def __init__(self, drop_prob: Optional[float] = None) -> None:
    super().__init__()
    self.drop_prob = drop_prob
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return drop_path(hidden_states, self.drop_prob, self.training)
  
  def extra_repr(self) -> str:
    return "p={}".format(self.drop_prob)


class _NeighborhoodAttentionNd(ABC, nn.Module):
  # rpb is learnable relative positional biases; same concept is used Swin.
  rpb: nn.Parameter
  nattendqkrpb: Callable
  nattendav: Callable
  
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__()
    if dim % num_heads != 0:
      raise ValueError(
        f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
      )
    
    self.num_attention_heads = num_heads
    self.attention_head_size = int(dim / num_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    self.kernel_size = kernel_size
    self.dilation = dilation
    
    self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    
    self.dropout = nn.Dropout(cfg.drop_attention)
  
  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor]:
    query_layer = self.transpose_for_scores(self.query(hidden_states))
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    
    # Apply the scale factor before computing attention weights. It's usually more efficient because
    # attention weights are typically a bigger tensor compared to query.
    # It gives identical results because scalars are commutable in matrix multiplication.
    query_layer = query_layer / math.sqrt(self.attention_head_size)
    
    # Compute NA between "query" and "key" to get the raw attention scores, and add relative positional biases.
    # attention_scores = natten2dqkrpb(query_layer, key_layer, self.rpb, self.dilation)
    attention_scores = self.nattendqkrpb(query_layer, key_layer, self.rpb, self.kernel_size, self.dilation)
    
    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)
    
    # context_layer = natten2dav(attention_probs, value_layer, self.dilation)
    context_layer = self.nattendav(attention_probs, value_layer, self.kernel_size, self.dilation)
    if len(context_layer.shape) > 4:  # 2D
      context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
    else:  # 1D
      context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)
    
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    
    return outputs
  
  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_x_shape)
    if len(x.shape) > 4:  # 2D
      return x.permute(0, 3, 1, 2, 4)
    else:  # 1D
      return x.permute(0, 2, 1, 3)


class NeighborhoodAttention1d(_NeighborhoodAttentionNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    self.rpb = nn.Parameter(
      torch.zeros(num_heads, (2 * self.kernel_size - 1)),
      requires_grad=True,
    )
    self.nattendqkrpb = natten1dqkrpb
    self.nattendav = natten1dav


class NeighborhoodAttention2d(_NeighborhoodAttentionNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    self.rpb = nn.Parameter(
      torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)),
      requires_grad=True,
    )
    self.nattendqkrpb = natten2dqkrpb
    self.nattendav = natten2dav


# Copied from transformers.models.nat.modeling_nat.NeighborhoodAttentionOutput
class NeighborhoodAttentionOutput(nn.Module):
  def __init__(self, config: Config, dim: int):
    super().__init__()
    self.dense = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(config.drop_attention)
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    
    return hidden_states


class _NeighborhoodAttentionModuleNd(ABC, nn.Module):
  self: _NeighborhoodAttentionNd
  
  def __init__(self, cfg: Config, dim: int):
    super().__init__()
    # self.self = _NeighborhoodAttentionNd(config, dim, num_heads, kernel_size, dilation)
    self.output = NeighborhoodAttentionOutput(cfg, dim)
  
  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor]:
    self_outputs = self.self(hidden_states, output_attentions)
    attention_output = self.output(self_outputs[0])
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


class NeighborhoodAttentionModule1d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg: Config, dim: int, num_heads: int, kernel_size: int, dilation: int):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention1d(cfg, dim, num_heads, kernel_size, dilation)


class NeighborhoodAttentionModule2d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg: Config, dim: int, num_heads: int, kernel_size: int, dilation: int):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention2d(cfg, dim, num_heads, kernel_size, dilation)


# Copied from transformers.models.nat.modeling_nat.NatIntermediate with Nat->Dinat
class DinatIntermediate(nn.Module):
  def __init__(self, config: Config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    if isinstance(config.act_transformer, str):
      self.intermediate_act_fn = get_activation_function(config.act_transformer)
    else:
      self.intermediate_act_fn = config.act_transformer
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


# Copied from transformers.models.nat.modeling_nat.NatOutput with Nat->Dinat
class DinatOutput(nn.Module):
  def __init__(self, config: Config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    self.dropout = nn.Dropout(config.drop_hidden)
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class _DinatLayerNd(ABC, nn.Module):
  attention: _NeighborhoodAttentionModuleNd
  attention2: Optional[_NeighborhoodAttentionModuleNd]
  
  def __init__(
    self,
    cfg: Config,
    dim: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__()
    self.double_attention = double_attention
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.window_size = self.kernel_size * self.dilation
    if double_attention:
      self.window_size *= 2
    self.layernorm_before = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
    self.drop_path = DinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    dim_after = dim * 2 if double_attention else dim
    self.layernorm_after = nn.LayerNorm(dim_after, eps=cfg.layer_norm_eps)
    self.intermediate = DinatIntermediate(cfg, dim_after, int(dim_after * cfg.mlp_ratio))
    self.output = DinatOutput(cfg, int(dim_after * cfg.mlp_ratio), dim)
  
  @abstractmethod
  def maybe_pad(self, *args, **kwargs):
    raise NotImplementedError
  
  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(hidden_states.shape) > 3:
      is_2d = True
      N, K, T, C = hidden_states.size()
    else:
      is_2d = False
      N, T, C = hidden_states.shape
    shortcut = hidden_states
    
    hidden_states = self.layernorm_before(hidden_states)
    # pad hidden_states if they are smaller than kernel size x dilation
    if is_2d:
      hidden_states, pad_values = self.maybe_pad(hidden_states, K, T)
      _, height_pad, width_pad, _ = hidden_states.shape
    else:
      hidden_states, pad_values = self.maybe_pad(hidden_states, T)
    
    attention_inputs = hidden_states
    hidden_states_list = []
    for attention in [self.attention, self.attention2]:
      if attention is None:
        continue
      
      attention_output = attention(attention_inputs, output_attentions=output_attentions)
      attention_output = attention_output[0]
      
      if is_2d:
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
          attention_output = attention_output[:, :K, :T, :].contiguous()
      else:
        was_padded = pad_values[3] > 0
        if was_padded:
          attention_output = attention_output[:, :T, :].contiguous()
      
      hidden_states = shortcut + self.drop_path(attention_output)
      hidden_states_list.append(hidden_states)
    
    if self.double_attention:
      hidden_states = torch.cat(hidden_states_list, dim=-1)
      shortcut = torch.stack(hidden_states_list).sum(dim=0) / 2.
    else:
      shortcut = hidden_states
    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.output(self.intermediate(layer_output))
    
    layer_output = shortcut + self.drop_path(layer_output)
    
    # layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
    layer_outputs = (layer_output,)
    return layer_outputs


class DinatLayer1d(_DinatLayerNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention)
    self.attention = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation)
    if double_attention:
      self.attention2 = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation * 2)
    else:
      self.attention2 = None
  
  def maybe_pad(self, hidden_states, frames):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0)
    if frames < window_size:
      pad_l = 0
      pad_r = max(0, window_size - frames)
      pad_values = (0, 0, pad_l, pad_r)
      hidden_states = nn.functional.pad(hidden_states, pad_values)
    return hidden_states, pad_values


class DinatLayer2d(_DinatLayerNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention=False)
    self.attention = NeighborhoodAttentionModule2d(cfg, dim, num_heads, kernel_size, dilation)
    self.attention2 = None
  
  def maybe_pad(self, hidden_states, height, width):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0, 0, 0)
    if height < window_size or width < window_size:
      pad_l = pad_t = 0
      pad_r = max(0, window_size - width)
      pad_b = max(0, window_size - height)
      pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
      hidden_states = nn.functional.pad(hidden_states, pad_values)
    return hidden_states, pad_values
