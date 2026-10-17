"""Capture semantic sampling and seen-token updates without padding request rows."""
import torch


class SamplingGraph:
    def __init__(self, sample, batch, vocabulary, device):
        self.sample = sample
        self.logits = torch.zeros(batch, vocabulary, device=device, dtype=torch.float32)
        self.uniforms = torch.full((batch,), 0.5, device=device)
        self.steps = torch.zeros(batch, device=device, dtype=torch.long)
        self.seen = torch.zeros(batch, vocabulary, device=device, dtype=torch.bool)
        self.graph = None

    def compute(self):
        tokens = self.sample(self.logits, self.uniforms, self.seen, self.steps)
        return tokens, self.seen.scatter(1, tokens, True)

    @torch.inference_mode()
    def capture(self):
        stream = torch.npu.Stream(device=self.logits.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.tokens, self.updated_seen = self.compute()

    @torch.inference_mode()
    def run(self, logits, uniforms, seen, steps):
        self.logits.copy_(logits)
        self.uniforms.copy_(uniforms)
        self.seen.copy_(seen)
        self.steps.copy_(steps)
        self.graph.replay()
        return self.tokens.clone(), self.updated_seen.clone()

    def close(self):
        torch.npu.synchronize()
        if self.graph is not None:
            self.graph.reset()
