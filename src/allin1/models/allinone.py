import torch
import torch.nn as nn

from typing import Optional
from .dinat import DinatLayer1d, DinatLayer2d
from .utils import get_activation_function
from ..config import Config
from ..typings import AllInOneOutput


class AllInOne(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()

    self.cfg = cfg
    self.num_levels = cfg.depth
    self.num_features = int(cfg.dim_embed * 2 ** (self.num_levels - 1))

    self.embeddings = AllInOneEmbeddings(cfg)

    self.encoder = AllInOneEncoder(
      cfg,
      depth=cfg.depth,
    )

    self.norm = nn.LayerNorm(cfg.dim_embed, eps=cfg.layer_norm_eps)

    self.beat_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.05)
    self.downbeat_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.0125)
    self.section_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.001)
    self.function_classifier = Head(num_classes=cfg.data.num_labels, cfg=cfg)

    self.dropout = nn.Dropout(cfg.drop_last)

  def forward(
    self,
    inputs: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    # N: batch size
    # K: instrument
    # C: channel
    # T: time
    # F: frequency
    # x has shape of: N, K, T, F
    N, K, T, F = inputs.shape

    inputs = inputs.reshape(-1, 1, T, F)  # N x K, C=1, T, F=81
    frame_embed = self.embeddings(inputs)  # NK, T, C=16

    encoder_outputs = self.encoder(
      frame_embed,
      output_attentions=output_attentions,
    )
    hidden_state_levels = encoder_outputs[0]

    hidden_states = hidden_state_levels[-1].reshape(N, K, T, -1)  # N, K, T, C=16
    hidden_states = self.norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    logits_beat = self.beat_classifier(hidden_states)
    logits_downbeat = self.downbeat_classifier(hidden_states)
    logits_section = self.section_classifier(hidden_states)
    logits_function = self.function_classifier(hidden_states)

    return AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_section=logits_section,
      logits_function=logits_function,
      embeddings=hidden_states,
    )


class AllInOneEncoder(nn.Module):
  def __init__(self, cfg: Config, depth: int):
    super().__init__()
    self.cfg = cfg

    drop_path_rates = [x.item() for x in torch.linspace(0, cfg.drop_path, depth)]
    dilations = [
      min(cfg.dilation_factor ** i, cfg.dilation_max)
      for i in range(depth)
    ]
    self.layers = nn.ModuleList(
      [
        AllInOneBlock(
          cfg=cfg,
          dilation=dilations[i],
          drop_path_rate=drop_path_rates[i],
        )
        for i in range(depth)
      ]
    )

  def forward(
    self,
    frame_embed: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    # N: batch size
    # K: instrument
    # T: time
    # C: channel
    # x has shape of: NK, T, C=16

    hidden_state_levels = []
    hidden_states = frame_embed
    for i, layer in enumerate(self.layers):
      layer_outputs = layer(hidden_states, output_attentions)
      hidden_states = layer_outputs[0]
      hidden_state_levels.append(hidden_states)

    outputs = (hidden_state_levels,)
    if output_attentions:
      outputs += layer_outputs[1:]
    return outputs


class AllInOneBlock(nn.Module):
  def __init__(self, cfg: Config, dilation: int, drop_path_rate: float):
    super().__init__()

    self.cfg = cfg
    self.dilation = dilation

    self.timelayer = DinatLayer1d(
      cfg=cfg,
      dim=cfg.dim_embed,
      num_heads=cfg.num_heads,
      kernel_size=cfg.kernel_size,
      dilation=dilation,
      drop_path_rate=drop_path_rate,
      double_attention=cfg.double_attention,
    )

    if cfg.instrument_attention:
      self.instlayer = DinatLayer2d(
        cfg=cfg,
        dim=cfg.dim_embed,
        num_heads=cfg.num_heads,
        kernel_size=5,
        dilation=1,
        drop_path_rate=drop_path_rate,
      )
    else:
      self.instlayer = DinatLayer1d(
        cfg=cfg,
        dim=cfg.dim_embed,
        num_heads=cfg.num_heads,
        kernel_size=5,
        dilation=1,
        drop_path_rate=drop_path_rate,
        double_attention=False,
      )

  def forward(
    self,
    hidden_states: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    # N: batch size
    # K: instrument
    # T: time
    # C: channel
    # x has shape of: NK, T, C=16
    NK, T, C = hidden_states.shape
    N, K = NK // self.cfg.data.num_instruments, self.cfg.data.num_instruments

    timelayer_outputs = self.timelayer(hidden_states, output_attentions)
    hidden_states = timelayer_outputs[0]
    if self.cfg.instrument_attention:
      hidden_states = hidden_states.reshape(N, K, T, C)
      instlayer_outputs = self.instlayer(hidden_states, output_attentions)
      hidden_states = instlayer_outputs[0]
      hidden_states = hidden_states.reshape(NK, T, C)
    else:
      instlayer_outputs = self.instlayer(hidden_states, output_attentions)
      hidden_states = instlayer_outputs[0]

    outputs = (hidden_states,)
    if output_attentions:
      outputs += timelayer_outputs[1:]
      if self.instlayer is not None:
        outputs += instlayer_outputs[1:]
    return outputs


class AllInOneEmbeddings(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    dim_input, hidden_size = cfg.dim_input, cfg.dim_embed
    self.dim_input = dim_input
    self.hidden_size = hidden_size

    self.act_fn = get_activation_function(cfg.act_conv)
    first_conv_filters = hidden_size if cfg.model == 'tcn' else hidden_size // 2

    self.conv0 = nn.Conv2d(1, first_conv_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
    self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
    self.drop0 = nn.Dropout(cfg.drop_conv)

    self.conv1 = nn.Conv2d(first_conv_filters, hidden_size, kernel_size=(1, 12), stride=(1, 1), padding=(0, 0))
    self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
    self.drop1 = nn.Dropout(cfg.drop_conv)

    self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
    self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))

    self.norm = nn.LayerNorm(cfg.dim_embed)
    self.dropout = nn.Dropout(cfg.drop_conv)

  def forward(self, x: torch.FloatTensor):
    # NK: batch x inst
    # C: channel
    # T: time
    # F: frequency
    # x has shape of: NK, C=1, T, F
    # x = x.unsqueeze(1)  # NK, C=1, T, F=81
    x = self.conv0(x)  # NK, C=16, T, F=79
    x = self.pool0(x)  # NK, C=16, T, F=26
    x = self.act_fn(x)
    x = self.drop0(x)

    x = self.conv1(x)  # NK, C=16, T, F=15
    x = self.pool1(x)  # NK, C=16, T, F=5
    x = self.act_fn(x)
    x = self.drop1(x)

    x = self.conv2(x)  # NK, C=16, T, F=3
    x = self.pool2(x)  # NK, C=16, T, F=1
    x = self.act_fn(x)

    embeddings = x.squeeze(-1)  # NK, C=16, T
    embeddings = embeddings.permute(0, 2, 1)  # NK, T, C=16
    embeddings = self.norm(embeddings)
    embeddings = self.dropout(embeddings)

    return embeddings


class Head(nn.Module):
  def __init__(self, num_classes: int, cfg: Config, init_confidence: float = None):
    super().__init__()
    self.classifier = nn.Linear(cfg.data.num_instruments * cfg.dim_embed, num_classes)

    if init_confidence is not None:
      self.reset_parameters(init_confidence)

  def reset_parameters(self, confidence) -> None:
    """
    Initialization following:
    "Focal loss for dense object detection." ICCV. 2017.
    """
    self.classifier.bias.data.fill_(-torch.log(torch.tensor(1 / confidence - 1)))

  def forward(self, x: torch.FloatTensor):
    # x shape: N, K, T, C=24
    batch, inst, frame, embed = x.shape
    x = x.permute(0, 2, 1, 3)  # batch, frame, inst, embed
    x = x.reshape(batch, frame, inst * embed)  # batch, frame, inst x embed
    logits = self.classifier(x)  # batch, frame, class
    logits = logits.permute(0, 2, 1)  # batch, class, frame
    if logits.shape[1] == 1:
      logits = logits.squeeze(1)
    return logits
