import torch
import torch.nn as nn
import torch.nn.functional as F

class RetrievalAugmentedAdaptiveFusion(nn.Module):
    """
    Retrieval-Augmented Adaptive Fusion (RAAF), equations (5)-(6).

    Current audiovisual tokens query perceptually retrieved audiovisual evidence;
    a learned sigmoid gate controls the residual injected into each modality.
    """
    def __init__(self, dim, n_heads=8):
        super().__init__()
        # 交叉注意力：Query为当前样本，Key/Value为证据(感知相似)样本
        self.video_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.audio_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        
        # 门控网络：决定吸收多少证据特征
        self.gate_video = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, cur_video, cur_audio, ev_video, ev_audio):
        """
        Args:
            cur_video: [batch_size, seq_len_v, dim] 当前样本的视频特征
            cur_audio: [batch_size, seq_len_a, dim] 当前样本的音频特征
            ev_video:  [batch_size, seq_len_ev_v, dim] 证据样本的视频特征
            ev_audio:  [batch_size, seq_len_ev_a, dim] 证据样本的音频特征
            
        Returns:
            fused_video: [batch_size, seq_len_v, dim] 增强后的视频特征
            fused_audio: [batch_size, seq_len_a, dim] 增强后的音频特征
            video_gate_mean: float, 视频门控均值（用于诊断）
            audio_gate_mean: float, 音频门控均值（用于诊断）
        """
        # 1. Video 交叉注意力融合
        v_attn, _ = self.video_cross_attn(query=cur_video, key=ev_video, value=ev_video)
        g_v = self.gate_video(torch.cat([cur_video, v_attn], dim=-1))
        fused_video = cur_video + g_v * v_attn  # 门控残差连接
        
        # 2. Audio 交叉注意力融合
        a_attn, _ = self.audio_cross_attn(query=cur_audio, key=ev_audio, value=ev_audio)
        g_a = self.gate_audio(torch.cat([cur_audio, a_attn], dim=-1))
        fused_audio = cur_audio + g_a * a_attn  # 门控残差连接
        
        video_gate_mean = g_v.detach().mean().item()
        audio_gate_mean = g_a.detach().mean().item()
        
        return fused_video, fused_audio, video_gate_mean, audio_gate_mean


class ModalityBalancingMoE(nn.Module):
    """
    Modality-Balancing Mixture of Experts (MB-MoE), equation (7).

    One global audiovisual router selects Top-K experts and shares the same
    routing weights across both modalities.
    """
    def __init__(self, dim, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # 多个专家网络，这里以简单的两层FFN为例
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for _ in range(n_experts)
        ])
        
        # 路由器网络，输入为 video 和 audio 的全局池化特征拼接
        self.router = nn.Linear(dim * 2, n_experts)
        
    def forward(self, fused_video, fused_audio):
        """
        Args:
            fused_video: [batch_size, seq_len_v, dim] 融合后的视频特征
            fused_audio: [batch_size, seq_len_a, dim] 融合后的音频特征
            
        Returns:
            balanced_video: [batch_size, seq_len_v, dim] 平衡后的视频特征
            balanced_audio: [batch_size, seq_len_a, dim] 平衡后的音频特征
        """
        # 1. 提取全局特征用于路由决策
        v_global = fused_video.mean(dim=1)  # [batch_size, dim]
        a_global = fused_audio.mean(dim=1)  # [batch_size, dim]
        
        # 2. 路由打分与 Top-K 选择
        logits = self.router(torch.cat([v_global, a_global], dim=-1))  # [batch_size, n_experts]
        top_k_weights, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # [batch_size, top_k]
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # 归一化权重
        
        # 3. 专家特征聚合 — 只计算被选中的专家，避免浪费
        balanced_video = torch.zeros_like(fused_video)
        balanced_audio = torch.zeros_like(fused_audio)
        
        activated_experts = set(top_k_indices.reshape(-1).tolist())
        expert_cache = {}
        for exp_idx in activated_experts:
            expert = self.experts[exp_idx]
            expert_cache[exp_idx] = (expert(fused_video), expert(fused_audio))

        for i in range(self.top_k):
            expert_weights = top_k_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            expert_indices = top_k_indices[:, i]  # [batch_size]
            
            for exp_idx in activated_experts:
                mask = (expert_indices == exp_idx).float().unsqueeze(-1).unsqueeze(-1)
                if mask.sum() > 0:
                    exp_video, exp_audio = expert_cache[exp_idx]
                    balanced_video += mask * expert_weights * exp_video
                    balanced_audio += mask * expert_weights * exp_audio
                    
        return balanced_video, balanced_audio


# Concise paper names used by the public API.
RAAF = RetrievalAugmentedAdaptiveFusion
MBMoE = ModalityBalancingMoE

# Backward-compatible pre-release implementation names.
SupportFusion = RetrievalAugmentedAdaptiveFusion
ModalityMoE = ModalityBalancingMoE
