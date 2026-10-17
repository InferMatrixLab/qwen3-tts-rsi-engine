"""Reference model execution on a single owner NPU thread."""

import numpy as np
import torch
import torch_npu
from qwen_tts import Qwen3TTSModel

from qwen3_tts_engine.predictor import PredictorGraph


class CustomVoiceModel:
    def __init__(self, model_path, device, predictor_graph=False, talker_graph=False):
        torch.npu.set_device(device)
        torch.set_num_threads(4)
        self.device = device
        self.wrapper = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        )
        self.wrapper.model.eval()
        self.predictor_graph = predictor_graph
        if predictor_graph:
            graph_type = PredictorGraph
            if predictor_graph == "whole":
                from qwen3_tts_engine.whole_predictor import WholePredictorGraph
                graph_type = WholePredictorGraph
            self.wrapper.model.talker.code_predictor = graph_type(self.wrapper.model.talker.code_predictor)
        self.talker_graph = talker_graph
        if talker_graph:
            from qwen3_tts_engine.talker_graph import TalkerGraph
            self.wrapper.model.talker.model = TalkerGraph(self.wrapper.model.talker.model)
        self.speakers = self.wrapper.get_supported_speakers()
        self.languages = self.wrapper.get_supported_languages()

    @torch.inference_mode()
    def generate(self, payloads):
        torch.npu.set_device(self.device)
        seed = payloads[0]["seed"]
        torch.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
        limit = payloads[0]["max_new_tokens"]
        codec_frames = [0] * len(payloads)
        ended = [False] * len(payloads)
        eos = self.wrapper.model.config.talker_config.codec_eos_token_id

        def observe(module, arguments, output):
            codes = output.hidden_states[-1]
            if codes is not None:
                for index, token in enumerate(codes[:, 0].tolist()):
                    if token == eos:
                        ended[index] = True
                    elif not ended[index]:
                        codec_frames[index] += 1

        hook = self.wrapper.model.talker.register_forward_hook(observe)
        try:
            waveforms, sample_rate = self.wrapper.generate_custom_voice(
                text=[p["input"] for p in payloads],
                speaker=[p["voice"] for p in payloads],
                language=[p["language"] for p in payloads],
                instruct=[p["instructions"] for p in payloads],
                max_new_tokens=limit,
                non_streaming_mode=True,
            )
            torch.npu.synchronize()
        finally:
            hook.remove()
        results = []
        for index, waveform in enumerate(waveforms):
            if not np.isfinite(waveform).all() or len(waveform) == 0:
                raise ValueError("Model emitted empty or nonfinite audio")
            pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
            results.append({"pcm": pcm, "sample_rate": sample_rate,
                            "codec_frames": codec_frames[index],
                            "finish_reason": "length" if codec_frames[index] >= limit - 1 else "stop"})
        return results

    def close(self):
        torch.npu.set_device(self.device)
        torch.npu.synchronize()
        if self.predictor_graph:
            self.wrapper.model.talker.code_predictor.close()
        if self.talker_graph:
            self.wrapper.model.talker.model.close()
        del self.wrapper
        torch.npu.empty_cache()
