"""AffectAgent rollout orchestration and paper-defined counterfactuals."""

import json
import h5py
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as affectgpt_config

from .schemas import (
    EmotionGeneratorOutput,
    EvidenceFilterOutput,
    EvidenceItem,
    FusionDiagnostics,
    QueryPlannerOutput,
    RolloutResult,
    RolloutSample,
)
from .prompts import (
    build_emotion_generator_messages,
    build_evidence_filter_messages,
    build_query_planner_messages,
    parse_emotion_generator_output,
    parse_evidence_filter_output,
    parse_query_planner_output,
)


class AffectAgentPipeline:
    """
    Official AffectAgent pipeline. All three agents share one multimodal LLM.
    """

    def __init__(
        self,
        chat,
        retriever,
        raaf=None,
        mb_moe=None,
        token_store_path: Optional[str] = None,
        id_to_row_path: Optional[str] = None,
        max_new_tokens_q: int = 256,
        max_new_tokens_s: int = 128,
        max_new_tokens_g: int = 256,
        retrieval_top_k: int = 3,
        channel_b_top_k: int = 1,
        compute_counterfactual_rewards: bool = True,
        enable_raaf: bool = True,
        enable_mb_moe: bool = True,
        support_fusion=None,
        modality_moe=None,
    ):
        """
        Args:
            chat:           AffectGPT 的 Chat 实例 (包含 model + tokenizer + 多模态处理)
            retriever: frozen dual-channel Retriever
            raaf: Retrieval-Augmented Adaptive Fusion
            mb_moe: Modality-Balancing MoE
            token_store_path: Channel B token_store 的 HDF5 路径
            id_to_row_path:   Channel B global_id → row 映射的 JSON 路径
            max_new_tokens_q/s/g: 各 Agent 的最大生成长度
            retrieval_top_k: Channel A 每组召回数量
            channel_b_top_k: Channel B 召回数量
        """
        self.chat = chat
        self.model = chat.model
        self.tokenizer = chat.tokenizer
        self.retriever = retriever
        self.raaf = raaf if raaf is not None else support_fusion
        self.mb_moe = mb_moe if mb_moe is not None else modality_moe
        # Backward-compatible attributes used by pre-release integrations.
        self.support_fusion = self.raaf
        self.modality_moe = self.mb_moe
        self.max_new_tokens_q = max_new_tokens_q
        self.max_new_tokens_s = max_new_tokens_s
        self.max_new_tokens_g = max_new_tokens_g
        self.retrieval_top_k = retrieval_top_k
        self.channel_b_top_k = channel_b_top_k
        self.compute_counterfactual_rewards = compute_counterfactual_rewards
        self.enable_raaf = enable_raaf
        self.enable_mb_moe = enable_mb_moe
        self.device = chat.device

        # Channel B token store (预计算的多模态 tokens)
        self.token_store_path = token_store_path
        self.id_to_row = {}
        if id_to_row_path and os.path.isfile(id_to_row_path):
            with open(id_to_row_path, "r") as f:
                raw = json.load(f)
                self.id_to_row = {int(k): v for k, v in raw.items()}

        # 特殊 token IDs
        self.FRAME_PATCH_TOKEN_ID = self.tokenizer.get_vocab().get(affectgpt_config.DEFAULT_FRAME_PATCH_TOKEN, -1)
        self.FACE_PATCH_TOKEN_ID = self.tokenizer.get_vocab().get(affectgpt_config.DEFAULT_FACE_PATCH_TOKEN, -1)
        self.AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab().get(affectgpt_config.DEFAULT_AUDIO_PATCH_TOKEN, -1)
        self.MULTI_PATCH_TOKEN_ID = self.tokenizer.get_vocab().get(affectgpt_config.DEFAULT_MULTI_PATCH_TOKEN, -1)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab().get(affectgpt_config.DEFAULT_IMAGE_PATCH_TOKEN, -1)

    # ══════════════════════════════════════════════════════
    # 底层：AffectGPT 原生多模态生成
    # ══════════════════════════════════════════════════════

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """将 chat messages 转为 AffectGPT 的 prompt 格式。"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"###Human: {content}")
        parts.append("###Assistant:")
        return "\n".join(parts)

    def _extract_generated_ids(
        self,
        query_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
    ) -> torch.Tensor:
        if sequence_ids.dim() > 1:
            sequence_ids = sequence_ids.squeeze(0)
        if query_ids is not None and sequence_ids.size(0) >= query_ids.size(0):
            prefix = query_ids.to(sequence_ids.device)
            if torch.equal(sequence_ids[:prefix.size(0)], prefix):
                return sequence_ids[prefix.size(0):].cpu()
        return sequence_ids.cpu()

    def _build_text_prompt_inputs(
        self,
        messages: List[Dict],
        llama_model=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self._messages_to_prompt(messages)
        input_ids = self.chat.to_token_ids(prompt, max_length=2000).to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)
        embed_tokens = (llama_model or self.model.llama_model).model.model.embed_tokens
        input_embeds = embed_tokens(input_ids)
        return input_ids, attention_mask, input_embeds

    def _build_multimodal_prompt_inputs(
        self,
        messages: List[Dict],
        img_list: Dict,
        llama_model=None,
        balanced_video=None,
        balanced_audio=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self._messages_to_prompt(messages)
        prompt = self.chat.replace_token_for_multimodal(prompt)

        input_ids = self.chat.to_token_ids(prompt, max_length=2000).to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)
        temp_input_ids = input_ids.clone()
        for patch_id in [self.FRAME_PATCH_TOKEN_ID, self.FACE_PATCH_TOKEN_ID,
                         self.AUDIO_PATCH_TOKEN_ID, self.MULTI_PATCH_TOKEN_ID,
                         self.IMAGE_PATCH_TOKEN_ID]:
            if patch_id >= 0:
                temp_input_ids[temp_input_ids == patch_id] = 0

        embed_tokens = (llama_model or self.model.llama_model).model.model.embed_tokens
        cur_input_embeds = embed_tokens(temp_input_ids)

        audio_embeds = balanced_audio if balanced_audio is not None else img_list.get('audio')
        replacements = [
            (self.FRAME_PATCH_TOKEN_ID, self.chat.num_video_query_token, img_list.get('frame')),
            (self.FACE_PATCH_TOKEN_ID, self.chat.num_video_query_token, img_list.get('face')),
            (self.AUDIO_PATCH_TOKEN_ID, self.chat.num_audio_query_token, audio_embeds),
            (self.MULTI_PATCH_TOKEN_ID, self.chat.num_multi_query_token, img_list.get('multi')),
            (self.IMAGE_PATCH_TOKEN_ID, self.chat.num_image_query_token, img_list.get('image')),
        ]

        if balanced_video is not None:
            replacements = [
                (self.FRAME_PATCH_TOKEN_ID, self.chat.num_video_query_token, balanced_video),
                (self.FACE_PATCH_TOKEN_ID, self.chat.num_video_query_token, balanced_video),
                (self.AUDIO_PATCH_TOKEN_ID, self.chat.num_audio_query_token, audio_embeds),
                (self.MULTI_PATCH_TOKEN_ID, self.chat.num_multi_query_token, img_list.get('multi')),
                (self.IMAGE_PATCH_TOKEN_ID, self.chat.num_image_query_token, img_list.get('image')),
            ]

        offset = 0
        for patch_token_id, query_token_number, embeds in replacements:
            if patch_token_id < 0:
                continue
            if (input_ids == patch_token_id).sum() == 0:
                continue
            if embeds is None:
                continue
            cur_features = embeds[0] if embeds.dim() == 3 else embeds
            cur_features = cur_features.to(self.device)
            # 只 detach 不需要梯度的特征（如 img_list 中的原始 Q-Former 输出）
            # 保留需要梯度的特征（如 fusion 模块的 balanced tokens），以便 PPO 梯度回传
            if not cur_features.requires_grad:
                cur_features = cur_features.detach()
            masked_indices = torch.where(input_ids == patch_token_id)[0]
            if len(masked_indices) == 0:
                continue
            mask_index_start = int(masked_indices[0].item()) + offset
            cur_input_embeds = torch.cat((
                cur_input_embeds[:mask_index_start],
                cur_features,
                cur_input_embeds[mask_index_start + query_token_number:],
            ), dim=0)
            offset += cur_features.size(0) - query_token_number

        # 多模态替换后 cur_input_embeds 长度可能已变化，重建 attention_mask 以对齐
        attention_mask = torch.ones(cur_input_embeds.size(0), dtype=torch.long, device=self.device)
        return input_ids, attention_mask, cur_input_embeds

    def _compute_response_stats_from_prompt(
        self,
        llama_model,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if response_ids is None:
            return torch.empty(0, device=self.device), None

        response_ids = response_ids.to(self.device)
        if response_ids.dim() > 1:
            response_ids = response_ids.squeeze(0)
        if response_ids.numel() == 0:
            return torch.empty(0, device=self.device), None

        response_embeds = llama_model.model.model.embed_tokens(response_ids)
        full_embeds = torch.cat([prompt_embeds, response_embeds], dim=0).unsqueeze(0)
        # 对齐 dtype（fusion 模块输出 fp32，LLM 权重可能为 fp16）
        model_dtype = next(llama_model.parameters()).dtype
        full_embeds = full_embeds.to(dtype=model_dtype)
        full_attention_mask = torch.cat([
            attention_mask,
            torch.ones(response_ids.size(0), dtype=attention_mask.dtype, device=self.device),
        ], dim=0).unsqueeze(0)

        outputs = llama_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        hidden_states = outputs.hidden_states[-1]
        prompt_len = prompt_embeds.size(0)
        response_logits = logits[0, prompt_len - 1: -1, :]
        log_probs = torch.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

        response_hidden = hidden_states[0, prompt_len: prompt_len + response_ids.size(0), :]
        if response_hidden.numel() == 0:
            response_hidden = hidden_states[0, prompt_len - 1, :].unsqueeze(0)

        # Token-level states are required by paper equation (9) for GAE.
        return token_log_probs, response_hidden

    def _compute_log_probs_from_prompt(
        self,
        llama_model,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_log_probs, _ = self._compute_response_stats_from_prompt(
            llama_model,
            prompt_embeds,
            attention_mask,
            response_ids,
        )
        return token_log_probs

    def _prepare_agent_replay_inputs(
        self,
        result: RolloutResult,
        agent_name: str,
        llama_model,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if agent_name == "q":
            img_list = result.runtime_context.get("img_list")
            if img_list is None:
                raise RuntimeError("Missing multimodal context for Query Planner replay.")
            messages = build_query_planner_messages(
                result.sample.subtitle,
                result.sample.candidate_labels,
                result.sample.lang,
                result.runtime_context.get("face_or_frame", "face"),
            )
            _, attention_mask, prompt_embeds = self._build_multimodal_prompt_inputs(
                messages,
                img_list,
                llama_model=llama_model,
            )
            response_ids = result.q_response_ids
        elif agent_name in {"f", "s"}:
            img_list = result.runtime_context.get("img_list")
            if img_list is None:
                raise RuntimeError("Missing multimodal context for Evidence Filter replay.")
            messages = build_evidence_filter_messages(
                result.sample.subtitle,
                result.candidates,
                result.sample.lang,
                result.runtime_context.get("face_or_frame", "face"),
            )
            _, attention_mask, prompt_embeds = self._build_multimodal_prompt_inputs(
                messages,
                img_list,
                llama_model=llama_model,
            )
            response_ids = result.s_response_ids
        elif agent_name == "g":
            img_list = result.runtime_context.get("img_list")
            if img_list is None:
                raise RuntimeError("Missing multimodal context for Emotion Generator replay.")
            messages = build_emotion_generator_messages(
                result.sample.subtitle,
                result.selected_evidence,
                result.sample.candidate_labels,
                result.sample.lang,
                result.runtime_context.get("face_or_frame", "face"),
            )
            # Re-run RAAF + MB-MoE so PPO keeps the fusion gradient path.
            # rollout 时的 balanced_video/audio 是 detached 的，直接用会切断融合模块梯度
            balanced_video = result.runtime_context.get("balanced_video")
            balanced_audio = result.runtime_context.get("balanced_audio")
            partner_tokens = result.runtime_context.get("partner_tokens")
            cur_tokens = result.runtime_context.get("cur_tokens")
            if (partner_tokens is not None and cur_tokens is not None
                    and self.enable_raaf and self.raaf is not None):
                cur_v, cur_a = cur_tokens
                par_v, par_a = partner_tokens
                if cur_v.dim() == 2: cur_v = cur_v.unsqueeze(0)
                if cur_a.dim() == 2: cur_a = cur_a.unsqueeze(0)
                if par_v.dim() == 2: par_v = par_v.unsqueeze(0)
                if par_a.dim() == 2: par_a = par_a.unsqueeze(0)
                fusion_dtype = next(self.raaf.parameters()).dtype
                cur_v = cur_v.to(device=self.device, dtype=fusion_dtype)
                cur_a = cur_a.to(device=self.device, dtype=fusion_dtype)
                par_v = par_v.to(device=self.device, dtype=fusion_dtype)
                par_a = par_a.to(device=self.device, dtype=fusion_dtype)
                fused_v, fused_a, _, _ = self.raaf(cur_v, cur_a, par_v, par_a)
                if self.enable_mb_moe and self.mb_moe is not None:
                    balanced_video, balanced_audio = self.mb_moe(fused_v, fused_a)
                else:
                    balanced_video, balanced_audio = fused_v, fused_a

            _, attention_mask, prompt_embeds = self._build_multimodal_prompt_inputs(
                messages,
                img_list,
                llama_model=llama_model,
                balanced_video=balanced_video,
                balanced_audio=balanced_audio,
            )
            response_ids = result.g_response_ids
        else:
            raise ValueError(f"Unknown agent name for PPO replay: {agent_name}")

        return attention_mask, prompt_embeds, response_ids

    def compute_agent_log_probs(
        self,
        result: RolloutResult,
        agent_name: str,
        llama_model,
    ) -> torch.Tensor:
        attention_mask, prompt_embeds, response_ids = self._prepare_agent_replay_inputs(
            result,
            agent_name,
            llama_model,
        )

        return self._compute_log_probs_from_prompt(
            llama_model,
            prompt_embeds,
            attention_mask,
            response_ids,
        )

    def compute_agent_log_probs_and_state(
        self,
        result: RolloutResult,
        agent_name: str,
        llama_model,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_mask, prompt_embeds, response_ids = self._prepare_agent_replay_inputs(
            result,
            agent_name,
            llama_model,
        )

        return self._compute_response_stats_from_prompt(
            llama_model,
            prompt_embeds,
            attention_mask,
            response_ids,
        )

    @torch.no_grad()
    def _generate_text_only(
        self,
        messages: List[Dict],
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Legacy text-only generation helper retained for external utilities.
        Returns: (query_ids, response_ids, response_text)
        """
        prompt = self._messages_to_prompt(messages)
        input_ids = self.chat.to_token_ids(prompt, max_length=2000)
        query_ids = input_ids.clone().cpu()

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)
        input_ids = input_ids.to(self.device)

        # 直接用 embed_tokens 获取 embeddings（无多模态替换）
        inputs_embeds = self.model.llama_model.model.model.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        outputs = self.model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.chat.stopping_criteria,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

        sequence_ids = outputs[0].cpu()
        response_ids = self._extract_generated_ids(query_ids, sequence_ids)
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        # 清理 stop tokens
        response_text = response_text.rsplit('###', 1)[0].strip()

        return query_ids, response_ids, response_text

    @torch.no_grad()
    def _generate_with_multimodal(
        self,
        messages: List[Dict],
        img_list: Dict,
        max_new_tokens: int,
        balanced_video=None,
        balanced_audio=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Shared multimodal generation helper for all three AffectAgent roles.
        将 <VideoHere>/<AudioHere> 等占位符替换为实际多模态 embeddings。

        If balanced_video/balanced_audio from RAAF + MB-MoE are provided,
        则用它们替换原始的 frame/face tokens。
        """
        prompt = self._messages_to_prompt(messages)
        prompt = self.chat.replace_token_for_multimodal(prompt)

        input_ids = self.chat.to_token_ids(prompt, max_length=2000)
        query_ids = input_ids.clone().cpu()
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)

        # Step 1: embed tokens, 将多模态 token IDs 置零
        temp_input_ids = input_ids.clone().to(self.device)
        for patch_id in [self.FRAME_PATCH_TOKEN_ID, self.FACE_PATCH_TOKEN_ID,
                         self.AUDIO_PATCH_TOKEN_ID, self.MULTI_PATCH_TOKEN_ID,
                         self.IMAGE_PATCH_TOKEN_ID]:
            if patch_id >= 0:
                temp_input_ids[temp_input_ids == patch_id] = 0
        cur_input_embeds = self.model.llama_model.model.model.embed_tokens(temp_input_ids)

        # Step 2: 替换多模态占位符为实际 embeddings
        audio_embeds = balanced_audio if balanced_audio is not None else img_list.get('audio')

        replacements = [
            (self.FRAME_PATCH_TOKEN_ID, self.chat.num_video_query_token, img_list.get('frame')),
            (self.FACE_PATCH_TOKEN_ID, self.chat.num_video_query_token, img_list.get('face')),
            (self.AUDIO_PATCH_TOKEN_ID, self.chat.num_audio_query_token, audio_embeds),
            (self.MULTI_PATCH_TOKEN_ID, self.chat.num_multi_query_token, img_list.get('multi')),
            (self.IMAGE_PATCH_TOKEN_ID, self.chat.num_image_query_token, img_list.get('image')),
        ]

        # 如果有 balanced tokens，替换 face/frame
        if balanced_video is not None:
            replacements = [
                (self.FRAME_PATCH_TOKEN_ID, self.chat.num_video_query_token, balanced_video),
                (self.FACE_PATCH_TOKEN_ID, self.chat.num_video_query_token, balanced_video),
                (self.AUDIO_PATCH_TOKEN_ID, self.chat.num_audio_query_token, balanced_audio),
                (self.MULTI_PATCH_TOKEN_ID, self.chat.num_multi_query_token, img_list.get('multi')),
                (self.IMAGE_PATCH_TOKEN_ID, self.chat.num_image_query_token, img_list.get('image')),
            ]

        offset = 0
        for (patch_token_id, query_token_number, embeds) in replacements:
            if patch_token_id < 0:
                continue
            if (input_ids == patch_token_id).sum() == 0:
                continue
            if embeds is None:
                continue
            cur_features = embeds[0] if embeds.dim() == 3 else embeds
            # detach 多模态特征，避免 PPO 多 epoch backward 时计算图已释放
            cur_features = cur_features.detach()
            masked_indices = torch.where(input_ids == patch_token_id)[0]
            if len(masked_indices) == 0:
                continue
            mask_index_start = int(masked_indices[0].item()) + offset
            cur_input_embeds = torch.cat((
                cur_input_embeds[:mask_index_start],
                cur_features,
                cur_input_embeds[mask_index_start + query_token_number:],
            ), dim=0)
            offset += cur_features.size(0) - query_token_number

        # 多模态替换后 cur_input_embeds 长度可能已变化，重建 attention_mask 以对齐
        attention_mask = torch.ones(cur_input_embeds.size(0), dtype=torch.long, device=self.device)
        cur_input_embeds = cur_input_embeds.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        # Step 3: 对齐 dtype（fusion 模块输出 fp32，LLM 权重可能为 fp16）
        model_dtype = next(self.model.llama_model.parameters()).dtype
        cur_input_embeds = cur_input_embeds.to(dtype=model_dtype)

        # Step 4: 生成
        outputs = self.model.llama_model.generate(
            inputs_embeds=cur_input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.chat.stopping_criteria,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

        sequence_ids = outputs[0].cpu()
        response_ids = self._extract_generated_ids(query_ids, sequence_ids)
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        # 清理 stop tokens
        response_text = response_text.rsplit('###', 1)[0].strip()

        return query_ids, response_ids, response_text

    # ══════════════════════════════════════════════════════
    # 多模态特征提取
    # ══════════════════════════════════════════════════════

    def extract_multimodal_features(self, sample_data: dict, face_or_frame: str = "face"):
        """
        从 AffectGPT 的 sample_data 中提取多模态 tokens 和向量。
        Returns:
            img_list: dict of multimodal embeddings for Chat
            feature_vectors: dict of numpy vectors for Channel B retrieval
        """
        audio_hiddens, audio_llms = self.chat.postprocess_audio(sample_data)
        frame_hiddens, frame_llms = self.chat.postprocess_frame(sample_data)
        face_hiddens, face_llms = self.chat.postprocess_face(sample_data)

        # 根据 face_or_frame 配置决定视频优先级
        if "face" in face_or_frame:
            video_llms = face_llms if face_llms is not None else frame_llms
            video_hiddens = face_hiddens if face_hiddens is not None else frame_hiddens
        else:
            video_llms = frame_llms if frame_llms is not None else face_llms
            video_hiddens = frame_hiddens if frame_hiddens is not None else face_hiddens

        # Multi fusion (如果配置需要)
        multi_llms = None
        if face_or_frame.startswith('multi') and video_hiddens is not None and audio_hiddens is not None:
            _, multi_llms = self.chat.postprocess_multi(video_hiddens, audio_hiddens)

        img_list = {
            'audio': audio_llms,
            'frame': frame_llms,
            'face': face_llms,
            'image': None,
            'multi': multi_llms,
        }

        # 提取用于 Channel B 检索的向量 (mean pooling + normalize)
        feature_vectors = {}
        for key, llm_out in [('video', video_llms), ('audio', audio_llms)]:
            if llm_out is not None:
                vec = llm_out.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                feature_vectors[key] = vec

        return img_list, feature_vectors, video_llms, audio_llms

    # ══════════════════════════════════════════════════════
    # Channel B: 加载 fusion partner 的预计算 tokens
    # ══════════════════════════════════════════════════════

    def load_partner_tokens(self, partner_global_id: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        从 token_store (HDF5) 加载 fusion partner 的预计算 video/audio tokens。
        """
        if self.token_store_path is None or not os.path.isfile(self.token_store_path):
            return None, None

        row_info = self.id_to_row.get(partner_global_id)
        if row_info is None:
            return None, None

        video_row = row_info if isinstance(row_info, int) else row_info.get("video_row", row_info.get("row", -1))
        audio_row = row_info if isinstance(row_info, int) else row_info.get("audio_row", row_info.get("row", -1))
        if video_row < 0 and audio_row < 0:
            return None, None

        try:
            with h5py.File(self.token_store_path, "r") as hf:
                video_tokens = None
                audio_tokens = None
                if "video_tokens" in hf and video_row >= 0:
                    video_tokens = torch.tensor(hf["video_tokens"][video_row], dtype=torch.float16)
                if "audio_tokens" in hf and audio_row >= 0:
                    audio_tokens = torch.tensor(hf["audio_tokens"][audio_row], dtype=torch.float16)
                return video_tokens, audio_tokens
        except Exception as e:
            print(f"[Orchestrator] Failed to load partner tokens for global_id={partner_global_id}: {e}")
            return None, None

    # ══════════════════════════════════════════════════════
    # RAAF + MB-MoE
    # ══════════════════════════════════════════════════════

    def run_fusion(
        self,
        cur_video: torch.Tensor,
        cur_audio: torch.Tensor,
        partner_video: torch.Tensor,
        partner_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, FusionDiagnostics]:
        """
        Run RAAF followed by MB-MoE, with explicit ablation switches.

        Returns:
            balanced_video, balanced_audio, diagnostics
        """
        # 确保 batch 维度
        if cur_video.dim() == 2:
            cur_video = cur_video.unsqueeze(0)
        if cur_audio.dim() == 2:
            cur_audio = cur_audio.unsqueeze(0)
        if partner_video.dim() == 2:
            partner_video = partner_video.unsqueeze(0)
        if partner_audio.dim() == 2:
            partner_audio = partner_audio.unsqueeze(0)

        # Move to device and align dtype with fusion module weights
        module = self.raaf if self.enable_raaf and self.raaf is not None else self.mb_moe
        if module is None:
            return cur_video, cur_audio, FusionDiagnostics()
        fusion_dtype = next(module.parameters()).dtype
        cur_video = cur_video.to(device=self.device, dtype=fusion_dtype)
        cur_audio = cur_audio.to(device=self.device, dtype=fusion_dtype)
        partner_video = partner_video.to(device=self.device, dtype=fusion_dtype)
        partner_audio = partner_audio.to(device=self.device, dtype=fusion_dtype)

        # RAAF, equations (5)-(6). The no-RAAF ablation keeps MB-MoE active.
        if self.enable_raaf and self.raaf is not None:
            fused_video, fused_audio, video_gate_mean, audio_gate_mean = self.raaf(
                cur_video, cur_audio, partner_video, partner_audio
            )
        else:
            fused_video, fused_audio = cur_video, cur_audio
            video_gate_mean, audio_gate_mean = 0.0, 0.0

        top_k_indices = None
        top_k_weights = None
        if self.enable_mb_moe and self.mb_moe is not None:
            balanced_video, balanced_audio = self.mb_moe(fused_video, fused_audio)
            with torch.no_grad():
                v_global = fused_video.mean(dim=1)
                a_global = fused_audio.mean(dim=1)
                logits = self.mb_moe.router(torch.cat([v_global, a_global], dim=-1))
                top_k_weights, top_k_indices = torch.topk(logits, self.mb_moe.top_k, dim=-1)
                top_k_weights = torch.softmax(top_k_weights, dim=-1)
        else:
            balanced_video, balanced_audio = fused_video, fused_audio

        diag = FusionDiagnostics(
            video_gate_mean=round(video_gate_mean, 4),
            audio_gate_mean=round(audio_gate_mean, 4),
            moe_experts_activated=top_k_indices[0].tolist() if top_k_indices is not None else [],
            moe_expert_weights=(
                [round(w, 4) for w in top_k_weights[0].tolist()]
                if top_k_weights is not None else []
            ),
        )

        return balanced_video, balanced_audio, diag

    # ══════════════════════════════════════════════════════
    # 各 Agent 的 Rollout 步骤
    # ══════════════════════════════════════════════════════

    def run_query_planner(
        self,
        sample: RolloutSample,
        img_list: Dict,
        face_or_frame: str = "face",
    ) -> Tuple[QueryPlannerOutput, torch.Tensor, torch.Tensor]:
        """Observe raw text, video, and audio and generate the three cognitive queries."""

        messages = build_query_planner_messages(
            sample.subtitle,
            sample.candidate_labels,
            sample.lang,
            face_or_frame,
        )
        query_ids, response_ids, text = self._generate_with_multimodal(
            messages,
            img_list,
            self.max_new_tokens_q,
        )
        return parse_query_planner_output(text), query_ids, response_ids

    def run_retrieval_channel_a(
        self,
        q_output: QueryPlannerOutput,
        exclude_sample_id: Optional[str] = None,
        fallback_subtitle: str = "",
    ) -> Dict[str, list]:
        """Channel A (E5 语义)：根据 Q 的子查询检索候选证据。"""
        if not q_output.valid:
            # Q 解析失败时，用 subtitle 生成默认查询，避免级联失败
            if fallback_subtitle:
                queries_dict = {
                    "support": {"query_text": fallback_subtitle},
                    "confusion": {"query_text": fallback_subtitle},
                    "counter": {"query_text": fallback_subtitle},
                }
            else:
                return {"support": [], "confusion": [], "counter": []}
        else:
            queries_dict = {
                "support": q_output.support,
                "confusion": q_output.confusion,
                "counter": q_output.counter,
            }
        try:
            candidates = self.retriever.retrieve_channel_A(
                queries_dict,
                top_k=self.retrieval_top_k,
                exclude_sample_id=exclude_sample_id,
            )
        except Exception as e:
            print(f"[Orchestrator] Channel A retrieval failed: {e}")
            candidates = {"support": [], "confusion": [], "counter": []}
        return candidates

    def run_retrieval_channel_b(
        self,
        feature_vectors: Dict[str, np.ndarray],
        exclude_sample_id: Optional[str] = None,
    ) -> Tuple[List[str], Optional[int]]:
        """Channel B (多模态 FAISS)：找感知最相似样本。"""
        try:
            partner_ids = self.retriever.retrieve_channel_B(
                feature_vectors,
                top_k=self.channel_b_top_k,
                exclude_sample_id=exclude_sample_id,
            )
            if partner_ids:
                # 获取 global_id
                name_to_gid = {
                    item.get("name"): gid
                    for gid, item in self.retriever.multimodal_meta.items()
                    if item.get("name")
                }
                partner_global_id = name_to_gid.get(partner_ids[0])
                return partner_ids, partner_global_id
        except Exception as e:
            print(f"[Orchestrator] Channel B retrieval failed: {e}")
        return [], None

    def run_evidence_filter(
        self,
        sample: RolloutSample,
        candidates: Dict[str, list],
        img_list: Dict,
        face_or_frame: str = "face",
    ) -> Tuple[EvidenceFilterOutput, torch.Tensor, torch.Tensor]:
        """Cross-verify retrieved cognitive evidence against the raw modalities."""
        messages = build_evidence_filter_messages(sample.subtitle, candidates, sample.lang, face_or_frame)
        query_ids, response_ids, text = self._generate_with_multimodal(
            messages, img_list, self.max_new_tokens_s
        )
        filter_output = parse_evidence_filter_output(text, candidates)
        return filter_output, query_ids, response_ids

    def resolve_evidence(
        self,
        filter_output: EvidenceFilterOutput,
        candidates: Dict[str, list],
    ) -> Dict[str, Optional[EvidenceItem]]:
        """Resolve the Evidence Filter decisions into the refined cognitive subset."""
        evidence = {"support": None, "confusion": None, "counter": None}

        for group in evidence:
            selected_id = getattr(filter_output, f"{group}_id", "")
            for item in candidates.get(group, []):
                if item["id"] == selected_id:
                    evidence[group] = EvidenceItem(
                        evidence_id=item["id"],
                        global_id=item.get("global_id"),
                        text=item.get("text", ""),
                        role=group,
                        subquery_type=group,
                        label_hint=item.get("label_hint", ""),
                    )
                    break
        return evidence

    def run_emotion_generator(
        self,
        sample: RolloutSample,
        evidence: Dict[str, Optional[EvidenceItem]],
        img_list: Dict,
        balanced_video=None,
        balanced_audio=None,
        face_or_frame: str = "face",
    ) -> Tuple[EmotionGeneratorOutput, torch.Tensor, torch.Tensor]:
        """Generate the final emotion and rationale from refined evidence and fused tokens."""
        messages = build_emotion_generator_messages(
            sample.subtitle, evidence, sample.candidate_labels, sample.lang, face_or_frame
        )
        query_ids, response_ids, text = self._generate_with_multimodal(
            messages, img_list, self.max_new_tokens_g,
            balanced_video=balanced_video,
            balanced_audio=balanced_audio,
        )
        output = parse_emotion_generator_output(text, sample.candidate_labels)
        return output, query_ids, response_ids

    @staticmethod
    def _complete_filter_fallback(
        filter_output: EvidenceFilterOutput,
        candidates: Dict[str, list],
    ) -> EvidenceFilterOutput:
        """Keep the rollout defined when structured decoding fails."""

        if filter_output.valid:
            return filter_output
        for group in ("support", "confusion", "counter"):
            items = candidates.get(group, [])
            if items:
                setattr(filter_output, f"{group}_id", str(items[0]["id"]))
        filter_output.valid = all(
            bool(getattr(filter_output, f"{group}_id"))
            for group in ("support", "confusion", "counter")
        )
        return filter_output

    @staticmethod
    def _simple_label_from_plan(
        query_output: QueryPlannerOutput,
        candidate_labels: List[str],
    ) -> str:
        """Return the simple-label replacement used by paper Score_label."""

        label = str(query_output.support.get("target_label", "")).strip()
        if label:
            return label
        query_text = str(query_output.support.get("query_text", "")).strip()
        for candidate in candidate_labels:
            if str(candidate).lower() in query_text.lower():
                return str(candidate)
        return str(candidate_labels[0]) if candidate_labels else "neutral"

    def _label_baseline_candidates(
        self,
        query_output: QueryPlannerOutput,
        sample: RolloutSample,
    ) -> Dict[str, list]:
        label = self._simple_label_from_plan(query_output, sample.candidate_labels)
        simple_queries = {
            group: {"query_text": label}
            for group in ("support", "confusion", "counter")
        }
        try:
            return self.retriever.retrieve_channel_A(
                simple_queries,
                top_k=self.retrieval_top_k,
                exclude_sample_id=sample.sample_id,
            )
        except Exception as error:
            print(f"[AffectAgent] Score_label retrieval failed: {error}")
            return {"support": [], "confusion": [], "counter": []}

    @staticmethod
    def resolve_rank_baseline(candidates: Dict[str, list]) -> Dict[str, List[EvidenceItem]]:
        """Bypass the Filter and pass ranked Top-K items directly to the Generator."""

        evidence: Dict[str, List[EvidenceItem]] = {}
        for group in ("support", "confusion", "counter"):
            evidence[group] = [
                EvidenceItem(
                    evidence_id=str(item.get("id", "")),
                    global_id=item.get("global_id"),
                    text=item.get("text", ""),
                    role=group,
                    subquery_type=group,
                    label_hint=item.get("label_hint", ""),
                )
                for item in candidates.get(group, [])
            ]
        return evidence

    def _run_reward_counterfactuals(
        self,
        result: RolloutResult,
        img_list: Dict,
        balanced_video,
        balanced_audio,
        face_or_frame: str,
    ) -> None:
        """Compute predictions required by Score_label and Score_rank."""

        label_candidates = self._label_baseline_candidates(result.query_output, result.sample)
        label_filter, _, _ = self.run_evidence_filter(
            result.sample,
            label_candidates,
            img_list,
            face_or_frame,
        )
        label_filter = self._complete_filter_fallback(label_filter, label_candidates)
        label_evidence = self.resolve_evidence(label_filter, label_candidates)
        result.label_baseline_output, _, _ = self.run_emotion_generator(
            result.sample,
            label_evidence,
            img_list,
            balanced_video=balanced_video,
            balanced_audio=balanced_audio,
            face_or_frame=face_or_frame,
        )

        rank_evidence = self.resolve_rank_baseline(result.candidates)
        result.rank_baseline_output, _, _ = self.run_emotion_generator(
            result.sample,
            rank_evidence,
            img_list,
            balanced_video=balanced_video,
            balanced_audio=balanced_audio,
            face_or_frame=face_or_frame,
        )

    # ══════════════════════════════════════════════════════
    # 完整 Rollout
    # ══════════════════════════════════════════════════════

    def full_rollout(
        self,
        sample: RolloutSample,
        sample_data: dict,
        face_or_frame: str = "face",
    ) -> RolloutResult:
        """Run Query Planner -> Retriever -> Filter -> RAAF/MB-MoE -> Generator."""
        result = RolloutResult(sample=sample)

        if sample_data is None:
            result.runtime_context = {
                "img_list": None,
                "balanced_video": None,
                "balanced_audio": None,
                "face_or_frame": face_or_frame,
            }
            result.query_output = QueryPlannerOutput(valid=False)
            result.filter_output = EvidenceFilterOutput(valid=False)
            result.generator_output = EmotionGeneratorOutput(valid=False)
            result.candidates = {"support": [], "confusion": [], "counter": []}
            result.selected_evidence = {"support": None, "confusion": None, "counter": None}
            result.fusion_diagnostics = FusionDiagnostics()
            return result

        # Step 1: 提取当前样本的多模态特征
        img_list, feature_vectors, cur_video_llms, cur_audio_llms = \
            self.extract_multimodal_features(sample_data, face_or_frame)
        result.runtime_context = {
            "img_list": img_list,
            "balanced_video": None,
            "balanced_audio": None,
            "face_or_frame": face_or_frame,
        }

        # Step 2: Query Planner observes all raw modalities.
        q_output, q_query, q_resp = self.run_query_planner(sample, img_list, face_or_frame)
        result.query_output = q_output
        result.q_query_ids = q_query
        result.q_response_ids = q_resp

        # Step 3: 双通道检索 (并行独立)
        # Channel A: Q 的子查询 → 文本证据（Q 失败时用 subtitle 做 fallback 查询）
        candidates = self.run_retrieval_channel_a(
            q_output,
            exclude_sample_id=sample.sample_id,
            fallback_subtitle=sample.subtitle,
        )
        result.candidates = candidates

        # Channel B: 当前样本向量 → 感知最似样本
        fusion_diag = FusionDiagnostics()
        balanced_video = None
        balanced_audio = None
        partner_ids, partner_global_id = [], None

        if feature_vectors and self.retriever.multimodal_ready:
            partner_ids, partner_global_id = self.run_retrieval_channel_b(
                feature_vectors,
                exclude_sample_id=sample.sample_id,
            )

        # Step 4: Evidence Filter cross-verifies cognitive evidence.
        filter_output, filter_query, filter_response = self.run_evidence_filter(
            sample,
            candidates,
            img_list,
            face_or_frame,
        )
        filter_output = self._complete_filter_fallback(filter_output, candidates)
        result.filter_output = filter_output
        result.f_query_ids = filter_query
        result.f_response_ids = filter_response

        # Step 5: Resolve evidence (角色由 Channel A 来源强绑定)
        evidence = self.resolve_evidence(filter_output, candidates)
        result.selected_evidence = evidence

        # Step 6: RAAF + MB-MoE over perceptual audiovisual evidence.
        if partner_global_id is not None and cur_video_llms is not None and cur_audio_llms is not None:
            partner_video, partner_audio = self.load_partner_tokens(partner_global_id)
            if partner_video is not None and partner_audio is not None:
                balanced_video, balanced_audio, fusion_diag = self.run_fusion(
                    cur_video_llms, cur_audio_llms, partner_video, partner_audio
                )
                fusion_diag.fusion_partner_id = partner_ids[0] if partner_ids else ""
                partner_meta = self.retriever.multimodal_meta.get(partner_global_id, {})
                fusion_diag.fusion_partner_label = str(partner_meta.get("ovlabel", ""))
                fusion_diag.fusion_partner_discrete_label = str(
                    partner_meta.get("discrete_label", "")
                )
                # detach balanced tensors 以释放 rollout 时的计算图，PPO replay 时会重新过 fusion
                result.runtime_context["balanced_video"] = balanced_video.detach()
                result.runtime_context["balanced_audio"] = balanced_audio.detach()
                # 存储原始 tokens 供 PPO replay 时重新过 fusion（保持梯度链路）
                result.runtime_context["cur_tokens"] = (cur_video_llms.detach(), cur_audio_llms.detach())
                result.runtime_context["partner_tokens"] = (partner_video.detach(), partner_audio.detach())
        elif (
            not self.enable_raaf
            and self.enable_mb_moe
            and self.mb_moe is not None
            and cur_video_llms is not None
            and cur_audio_llms is not None
        ):
            # The no-RAAF ablation still evaluates MB-MoE on the raw modalities.
            balanced_video, balanced_audio, fusion_diag = self.run_fusion(
                cur_video_llms,
                cur_audio_llms,
                cur_video_llms,
                cur_audio_llms,
            )
            result.runtime_context["balanced_video"] = balanced_video.detach()
            result.runtime_context["balanced_audio"] = balanced_audio.detach()

        result.fusion_diagnostics = fusion_diag

        # Step 7: Emotion Generator.
        g_output, g_query, g_resp = self.run_emotion_generator(
            sample, evidence, img_list,
            balanced_video=balanced_video,
            balanced_audio=balanced_audio,
            face_or_frame=face_or_frame,
        )
        result.generator_output = g_output
        result.g_query_ids = g_query
        result.g_response_ids = g_resp

        # Equations (1)-(4): two additional predictions for local increments.
        if self.compute_counterfactual_rewards:
            self._run_reward_counterfactuals(
                result,
                img_list,
                balanced_video,
                balanced_audio,
                face_or_frame,
            )

        return result

    def batch_rollout(
        self,
        samples: List[RolloutSample],
        sample_data_list: List[dict],
        face_or_frame: str = "face",
    ) -> List[RolloutResult]:
        """批量 rollout（逐样本串行，保证逻辑正确性）。"""
        results = []
        for sample, sample_data in zip(samples, sample_data_list):
            result = self.full_rollout(sample, sample_data, face_or_frame)
            results.append(result)
        return results

    def collect_ppo_trajectories(
        self,
        results: List[RolloutResult],
    ) -> List[Dict[str, object]]:
        trajectories = []

        for result in results:
            bd = result.rewards
            if bd is None:
                continue

            if result.q_response_ids is not None and result.q_response_ids.numel() > 0:
                trajectories.append({
                    "agent": "q",
                    "result": result,
                    "reward": float(bd.r_planner),
                })

            if result.f_response_ids is not None and result.f_response_ids.numel() > 0:
                trajectories.append({
                    "agent": "f",
                    "result": result,
                    "reward": float(bd.r_filter),
                })

            if result.g_response_ids is not None and result.g_response_ids.numel() > 0:
                trajectories.append({
                    "agent": "g",
                    "result": result,
                    "reward": float(bd.r_generator),
                })

        return trajectories

    # Compatibility methods for the pre-release Q/S/G API.
    def run_agent_q(self, sample, img_list=None, face_or_frame="face"):
        if img_list is None:
            raise ValueError("run_agent_q now requires multimodal img_list; use run_query_planner.")
        return self.run_query_planner(sample, img_list, face_or_frame)

    run_agent_s = run_evidence_filter
    run_agent_g = run_emotion_generator


# Backward-compatible pre-release class name.
MmoaOrchestrator = AffectAgentPipeline
