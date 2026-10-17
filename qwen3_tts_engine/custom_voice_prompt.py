# Adapted from Qwen3-TTS, Copyright 2026 The Qwen team, Alibaba Group
# and the HuggingFace Inc. team. Licensed under Apache-2.0.
"""Experimental single-request CustomVoice non-streaming prompt preparation.

Caches only model-constant special-token and speaker/language embeddings.
Request text, instructions and their projections are never cached.
Service opt-in: --direct-prompt. Model weights, dtype and device must remain
fixed for the lifetime of this builder.
"""
import torch


class CustomVoicePrompt:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.model = wrapper.model
        self.talker = self.model.talker
        if self.model.tts_model_type != 'custom_voice' or self.model.tts_model_size != '1b7':
            raise ValueError('1.7B CustomVoice model required')
        self.special = None
        self.prefixes = {}

    @torch.inference_mode()
    def build(self, payload):
        wrapper, talker, config = self.wrapper, self.talker, self.model.config
        language, speaker = payload['language'], payload['voice']
        wrapper._validate_languages([language])
        wrapper._validate_speakers([speaker])
        ids = wrapper._tokenize_texts([wrapper._build_assistant_text(payload['input'])])[0]
        token_config = config.talker_config
        language_id = None if language.lower() == 'auto' else token_config.codec_language_id[language.lower()]
        dialect = token_config.spk_is_dialect[speaker.lower()]
        if language.lower() in ('chinese', 'auto') and dialect is not False:
            language_id = token_config.codec_language_id[dialect]
        if self.special is None:
            special_ids = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id,
                                         config.tts_pad_token_id]], device=ids.device, dtype=ids.dtype)
            self.special = talker.text_projection(talker.get_text_embeddings()(special_ids)).chunk(3, dim=1)
        bos, eos, pad = self.special
        key = (speaker.lower(), language_id)
        if key not in self.prefixes:
            tags = ([token_config.codec_nothink_id, token_config.codec_think_bos_id,
                     token_config.codec_think_eos_id] if language_id is None else
                    [token_config.codec_think_id, token_config.codec_think_bos_id,
                     language_id, token_config.codec_think_eos_id])
            embedding = talker.get_input_embeddings()
            tag_embed = embedding(torch.tensor([tags], device=ids.device, dtype=ids.dtype))
            speaker_embed = embedding(torch.tensor(token_config.spk_id[speaker.lower()],
                                                  device=ids.device, dtype=ids.dtype)).view(1, 1, -1)
            tail = embedding(torch.tensor([[token_config.codec_pad_id, token_config.codec_bos_id]],
                                          device=ids.device, dtype=ids.dtype))
            codec = torch.cat((tag_embed, speaker_embed, tail), dim=1)
            prefix = torch.cat((pad.expand(-1, codec.shape[1] - 2, -1), bos), dim=1) + codec[:, :-1]
            self.prefixes[key] = (prefix, tail[:, :1], pad + tail[:, 1:])
        prefix, codec_pad, final_bos = self.prefixes[key]
        parts = []
        instruction = payload.get('instructions', '')
        if instruction:
            instruction_ids = wrapper._tokenize_texts([wrapper._build_instruct_text(instruction)])[0]
            parts.append(talker.text_projection(talker.get_text_embeddings()(instruction_ids)))
        parts.append(talker.text_projection(talker.get_text_embeddings()(ids[:, :3])))
        parts.append(prefix)
        text = talker.text_projection(talker.get_text_embeddings()(ids[:, 3:-5]))
        parts.append(torch.cat((text, eos), dim=1) + codec_pad)
        parts.append(final_bos)
        inputs = torch.cat(parts, dim=1)
        result = wrapper._merge_generate_kwargs(max_new_tokens=payload['max_new_tokens'])
        result.update(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=ids.device, dtype=torch.long),
                      trailing_text_hidden=pad.clone(), tts_pad_embed=pad.clone())
        return result
