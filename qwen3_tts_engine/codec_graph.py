"""Bounded streaming codec windows, with compute graphs per input shape."""
import torch


class CodecGraph:
    def __init__(self, decoder):
        self.decoder = decoder
        self.captures = {}

    @torch.inference_mode()
    def decode(self, codes):
        key = (tuple(codes.shape), codes.dtype, codes.device)
        if key not in self.captures:
            static = codes.clone()
            stream = torch.npu.Stream(device=codes.device)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    self.decoder(static)
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                output = self.decoder(static)
            self.captures[key] = (graph, static, output)
        graph, static, output = self.captures[key]
        static.copy_(codes)
        graph.replay()
        return output.clone()

    def close(self):
        torch.npu.synchronize()
        for graph, *_ in self.captures.values():
            graph.reset()
        self.captures.clear()
