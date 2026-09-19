"""Exact codec optimization adapted from kadirnar/fast-dacvae.

Derived from optimize.py at 406f2e5c803927ef18cc9bbe38d715e5417459b9.
See third_party/fast-dacvae/LICENSE and docs/fast-codec-parallel.md.
The upstream replay/approximation/noise/watermark shortcuts are not used.
Copyright (c) 2026 Kadir Nar. MIT license; see the attributed license file.
"""

from collections import OrderedDict

import torch
from torch import nn

SOURCE_REVISION = "406f2e5c803927ef18cc9bbe38d715e5417459b9"


class ExactSnake2d(nn.Module):
    def __init__(self, original):
        super().__init__()
        alpha = original.alpha.detach().unsqueeze(-1)
        self.register_buffer("alpha", alpha)
        self.register_buffer("inverse_alpha", (alpha + 1e-9).reciprocal())

    def forward(self, x):
        return x + self.inverse_alpha * torch.sin(self.alpha * x).square()


class Residual2d(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.block = convert(original.block)
        self.true_skip = original.true_skip

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if not self.true_skip and pad > 0:
            x = x[..., pad:-pad]
        return x + y


def convert(module):
    """Preserve weights and convolution geometry; fail on unsupported padding."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        if getattr(module, "pad_mode", "none") != "none" or module.padding_mode != "zeros":
            raise ValueError("Fast codec conversion requires static zero-padding convolutions")
        if hasattr(module, "weight_g") or hasattr(module, "weight_v"):
            raise ValueError("Fold weight normalization before fast codec conversion")
        kwargs = dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=(1, module.kernel_size[0]),
            stride=(1, module.stride[0]),
            padding=(0, module.padding[0]),
            dilation=(1, module.dilation[0]),
            groups=module.groups,
            bias=module.bias is not None,
            # Meta initialization avoids allocating throwaway weights or consuming RNG.
            device="meta",
            dtype=module.weight.dtype,
        )
        if isinstance(module, nn.ConvTranspose1d):
            result = nn.ConvTranspose2d(**kwargs, output_padding=(0, module.output_padding[0]))
        else:
            result = nn.Conv2d(**kwargs)
        result.weight = nn.Parameter(
            module.weight.detach().unsqueeze(2).to(memory_format=torch.channels_last), requires_grad=False
        )
        if module.bias is not None:
            result.bias = nn.Parameter(module.bias.detach(), requires_grad=False)
        return result
    name = type(module).__name__
    if name == "Snake1d":
        return ExactSnake2d(module)
    if name == "ResidualUnit":
        return Residual2d(module)
    if name in {"Encoder", "EncoderBlock"}:
        return convert(module.block)
    if isinstance(module, nn.Sequential):
        return nn.Sequential(*(convert(child) for child in module))
    if isinstance(module, (nn.Identity, nn.Tanh, nn.ELU)):
        return module
    raise ValueError(f"Unsupported fast codec layer: {name}")


def decoder_groups(block):
    size = block._chunk_size
    chunks = [list(block.block[i : i + size]) for i in range(0, len(block.block), size)]
    forward = [layer for i, chunk in enumerate(chunks) if i % size == 0 for layer in chunk]
    other = [layer for i, chunk in enumerate(chunks) if i % size != 0 for layer in chunk]
    return (
        nn.Sequential(*forward),
        nn.Sequential(*other[len(other) // 2 :]),
        nn.Sequential(*other[: len(other) // 2]),
    )


class FastPosterior(nn.Module):
    def __init__(self, model, layout):
        super().__init__()
        self.channels_last = layout == "channels_last"
        self.encoder = convert(model.encoder) if self.channels_last else model.encoder
        self.projection = convert(model.quantizer.in_proj) if self.channels_last else model.quantizer.in_proj

    def forward(self, audio):
        x = audio.unsqueeze(2).to(memory_format=torch.channels_last) if self.channels_last else audio
        mean, _ = self.projection(self.encoder(x)).chunk(2, dim=1)
        return mean.squeeze(2) if self.channels_last else mean


class FastDecoderTrunk(nn.Module):
    def __init__(self, model, layout):
        super().__init__()
        self.channels_last = layout == "channels_last"
        self.projection = (
            convert(model.quantizer.out_proj) if self.channels_last else model.quantizer.out_proj
        )
        layers = []
        for layer in model.decoder.model:
            if type(layer).__name__ == "DecoderBlock":
                forward, up, down = decoder_groups(layer)
                layers.append(convert(forward) if self.channels_last else forward)
                # Retain the original 3D watermark network, with cached groups.
                layer.upsample_group = lambda group=up: group
                layer.downsample_group = lambda group=down: group
            else:
                layers.append(convert(layer) if self.channels_last else layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, latent):
        x = latent.unsqueeze(2).to(memory_format=torch.channels_last) if self.channels_last else latent
        output = self.layers(self.projection(x))
        return output.squeeze(2) if self.channels_last else output


class CudaGraphPool:
    """Bounded repeated-shape graphs; always copy fresh input and return owned output.

    Graphs are never captured on CPU or for every one-off duration. Once the pool
    is full, other shapes use eager/compiled execution without additional capture.
    Each pool belongs to one codec process and is not shared between threads.
    """

    def __init__(self, function, max_shapes=4, warmup_calls=3):
        if max_shapes < 1 or warmup_calls < 1:
            raise ValueError("Graph limits must be positive")
        self.function, self.max_shapes, self.warmup_calls = function, max_shapes, warmup_calls
        self.graphs, self.visits = {}, OrderedDict()
        self.hits, self.fallbacks = 0, 0

    def __call__(self, x):
        if x.device.type != "cuda":
            raise ValueError("CUDA graphs require a CUDA device")
        key = (
            tuple(x.shape),
            x.dtype,
            x.device,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )
        if key not in self.graphs:
            self.visits[key] = self.visits.get(key, 0) + 1
            self.visits.move_to_end(key)
            if len(self.visits) > 128:
                self.visits.popitem(last=False)
            if len(self.graphs) >= self.max_shapes or self.visits[key] < self.warmup_calls:
                self.fallbacks += 1
                return self.function(x)
            with torch.cuda.device(x.device):
                static_input = x.clone()
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.function(static_input)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    static_output = self.function(static_input)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                self.graphs[key] = (graph, static_input, static_output)
        graph, static_input, static_output = self.graphs[key]
        static_input.copy_(x)
        graph.replay()
        self.hits += 1
        return static_output.clone()


class FastCodec:
    def __init__(
        self,
        model,
        encoder_only=False,
        compile_model=False,
        cuda_graphs=False,
        graph_max_shapes=4,
        graph_warmup=3,
        layout="native",
    ):
        if layout not in {"native", "channels_last"}:
            raise ValueError("Fast codec layout must be native or channels_last")
        self.encoder = FastPosterior(model, layout).eval().requires_grad_(False)
        self.decoder = None if encoder_only else FastDecoderTrunk(model, layout).eval().requires_grad_(False)
        if compile_model:
            self.encoder = torch.compile(self.encoder, dynamic=True, fullgraph=True)
            if self.decoder is not None:
                self.decoder = torch.compile(self.decoder, dynamic=True, fullgraph=True)
        self.encoder_graphs = (
            CudaGraphPool(self.encoder, graph_max_shapes, graph_warmup) if cuda_graphs else None
        )
        self.decoder_graphs = (
            CudaGraphPool(self.decoder, graph_max_shapes, graph_warmup)
            if cuda_graphs and self.decoder is not None
            else None
        )

    def encode(self, audio):
        return (self.encoder_graphs or self.encoder)(audio)

    def decode(self, latent, model):
        trunk = (self.decoder_graphs or self.decoder)(latent)
        # Keep fresh per-request watermark messages and the complete pretrained path.
        return model.decoder.watermark(trunk)

    def graph_statistics(self):
        return {
            name: {
                "captured_shapes": len(pool.graphs),
                "replays": pool.hits,
                "uncaptured_calls": pool.fallbacks,
            }
            for name, pool in (("encoder", self.encoder_graphs), ("decoder", self.decoder_graphs))
            if pool is not None
        }
