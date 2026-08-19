import os
import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class DoubleChannelRetriever:
    """
    MMOA-Lite 双通道检索服务 (Service-R)
    该模块为冻结状态，不参与梯度更新。
    
    包含两个完全独立的通道：
    1. Channel A: 基于 E5 的语义检索 (为 Agent-S 提供候选和 Agent-G 提供文本 prompt)
    2. Channel B: 基于 FAISS 的多模态检索 (为 SupportFusion 提供最相似的融合特征)
    """
    
    def __init__(self, semantic_index_dir=None, multimodal_index_dir=None):
        """
        初始化双通道检索器
        
        Args:
            semantic_index_dir: Channel A 语义索引存放的目录
            multimodal_index_dir: Channel B 多模态索引存放的目录 (后续如果需要完全跑通可传入)
        """
        self.semantic_index_dir = semantic_index_dir
        self.multimodal_index_dir = multimodal_index_dir
        self.metadata = []
        self.semantic_faiss = None
        self.encoder = None
        
        # --- 初始化 Channel A (语义通道) ---
        if self.semantic_index_dir:
            print(f"Loading Semantic Index from {semantic_index_dir}...")
            self._load_semantic_index()
        
        # --- 初始化 Channel B (多模态感知通道) ---
        self.video_index = None
        self.audio_index = None
        self.video_ids = None
        self.audio_ids = None
        self.multimodal_meta = {}
        self.multimodal_id_to_row = {}
        self.multimodal_ready = False
        if self.multimodal_index_dir:
            self._load_multimodal_index()

    @staticmethod
    def _first_existing_path(candidates, expect_dir=False):
        for path in candidates:
            if not path:
                continue
            if expect_dir and os.path.isdir(path):
                return path
            if not expect_dir and os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _normalize_vec(vec):
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32)
        
    def _load_semantic_index(self):
        """加载 Channel A 的 E5 编码器、FAISS 索引以及 metadata"""
        # 加载 Metadata
        metadata_path = os.path.join(self.semantic_index_dir, "metadata.jsonl")
        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        
        # 加载 FAISS 索引
        index_path = self._first_existing_path([
            os.path.join(self.semantic_index_dir, "semantic_faiss.index"),
            os.path.join(self.semantic_index_dir, "e5_semantic.index"),
        ])
        if index_path is None:
            raise FileNotFoundError(
                f"No semantic FAISS index found under {self.semantic_index_dir}. "
                "Expected semantic_faiss.index or e5_semantic.index"
            )
        self.semantic_faiss = faiss.read_index(index_path)
        
        # 加载对应的模型名 (在构建时保存)
        self.e5_model_name = "intfloat/multilingual-e5-base"  # 默认值
        try:
            with open(os.path.join(self.semantic_index_dir, "build_config.json"), 'r') as f:
                config = json.load(f)
                self.e5_model_name = config.get("model_name", self.e5_model_name)
        except Exception:
            print(f"Warning: Could not read build_config.json, using default {self.e5_model_name}")

        # 离线环境：尝试将 HuggingFace 模型名解析为本地路径
        resolved_path = self._resolve_local_model_path(self.e5_model_name)
        print(f"Loading E5 Model: {resolved_path}...")
        self.encoder = SentenceTransformer(resolved_path)

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> str:
        """尝试将模型名解析为本地路径，找不到则原样返回（让 sentence-transformers 走在线下载）。"""
        # 如果已经是有效的本地目录，直接返回
        if os.path.isdir(model_name):
            return os.path.abspath(model_name)

        # 从 HuggingFace 风格名称提取短名，如 "intfloat/multilingual-e5-base" -> "multilingual-e5-base"
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name

        # 常见本地候选路径
        candidates = [
            short_name,                                          # CWD/multilingual-e5-base
            os.path.join(os.environ.get("AFFECTAGENT_MODEL_ROOT", "models"), short_name),
            os.path.join("models", short_name),                  # CWD/models/multilingual-e5-base
            os.path.join(os.path.dirname(__file__), "..", "..", short_name),  # 项目根/multilingual-e5-base
            os.path.join(os.path.dirname(__file__), "..", "..", "models", short_name),
        ]
        for path in candidates:
            if os.path.isdir(path):
                resolved = os.path.abspath(path)
                print(f"  -> Resolved E5 to local path: {resolved}")
                return resolved

        # 未找到本地路径，原样返回
        return model_name

    def _load_multimodal_index(self):
        """加载 Channel B 的多模态 FAISS 索引、ID 映射和 metadata。"""
        root = self.multimodal_index_dir
        index_dir = self._first_existing_path([
            os.path.join(root, "index"),
            root,
        ], expect_dir=True)
        vector_dir = self._first_existing_path([
            os.path.join(root, "vector"),
            root,
        ], expect_dir=True)
        meta_dir = self._first_existing_path([
            os.path.join(root, "meta"),
            root,
        ], expect_dir=True)
        token_dir = self._first_existing_path([
            os.path.join(root, "token_store"),
            root,
        ], expect_dir=True)

        if index_dir is None or vector_dir is None or meta_dir is None:
            raise FileNotFoundError(
                f"Invalid multimodal index dir: {root}. Expected subdirs index/, vector/, meta/"
            )

        video_index_path = os.path.join(index_dir, "video.index")
        audio_index_path = os.path.join(index_dir, "audio.index")
        video_ids_path = os.path.join(vector_dir, "video_ids.npy")
        audio_ids_path = os.path.join(vector_dir, "audio_ids.npy")
        meta_path = self._first_existing_path([
            os.path.join(meta_dir, "metadata.jsonl"),
            os.path.join(root, "metadata.jsonl"),
        ])

        if os.path.isfile(video_index_path):
            self.video_index = faiss.read_index(video_index_path)
        if os.path.isfile(audio_index_path):
            self.audio_index = faiss.read_index(audio_index_path)
        if os.path.isfile(video_ids_path):
            self.video_ids = np.load(video_ids_path)
        if os.path.isfile(audio_ids_path):
            self.audio_ids = np.load(audio_ids_path)
        if meta_path is not None:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    gid = item.get("global_id")
                    if gid is not None:
                        self.multimodal_meta[int(gid)] = item

        id_to_row_path = None
        if token_dir is not None:
            id_to_row_path = os.path.join(token_dir, "id_to_row.json")
        if id_to_row_path and os.path.isfile(id_to_row_path):
            with open(id_to_row_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.multimodal_id_to_row = {int(k): v for k, v in raw.items()}

        self.multimodal_ready = bool(
            (self.video_index is not None and self.video_ids is not None)
            or (self.audio_index is not None and self.audio_ids is not None)
        )

    def retrieve_channel_A(self, queries_dict, top_k=3, exclude_sample_id=None):
        """
        通道 A (语义通道)：根据 Agent-Q 的三条不同子查询，检索出三个相互独立的候选池。
        
        Args:
            queries_dict: Agent-Q 生成的包含 3 条子查询的字典。例如：
                {
                   "primary": {"query_text": "..."},
                   "confusion": {"query_text": "..."},
                   "counter": {"query_text": "..."}
                }
            top_k: 每个子查询召回的数量 (默认 3)
            
        Returns:
            candidates: 包含三组结果的字典，传递给 Agent-S 筛选
                {
                   "primary": [{"id": ..., "text": ..., "score": ...}, ...],
                   "confusion": [...],
                   "counter": [...]
                }
        """
        if self.semantic_faiss is None or self.encoder is None:
            raise RuntimeError(
                "Channel A semantic index is not loaded. Please provide semantic_index_dir when initializing DoubleChannelRetriever."
            )
        candidates = {"primary": [], "confusion": [], "counter": []}
        if len(self.metadata) == 0:
            return candidates
        
        # 遍历三类 query
        for q_type, q_info in queries_dict.items():
            if q_type not in candidates:
                continue
                
            query_text = q_info.get("query_text", "")
            if not query_text:
                continue
                
            # E5 编码约定：query 端需加上 "query: " 前缀
            encoded_query = f"query: {query_text}"
            
            # 推理向量
            q_emb = self.encoder.encode([encoded_query], normalize_embeddings=True)
            
            # FAISS 内积检索
            search_k = min(max(top_k * 3, top_k + 1), len(self.metadata))
            scores, indices = self.semantic_faiss.search(q_emb, search_k)
            
            # 组装结果
            for rank in range(search_k):
                idx = indices[0][rank]
                score = scores[0][rank]
                if idx < 0 or idx >= len(self.metadata):
                    continue
                    
                meta = self.metadata[idx]
                if exclude_sample_id and meta.get("name") == exclude_sample_id:
                    continue
                global_id = meta.get("global_id", idx)
                description = meta.get("description", meta.get("reason", ""))
                
                # 按照方案设计：检索出的文本证据需包含 subtitle 和 description
                evidence_text = f"{meta.get('subtitle', '')}。{description}".strip("。")
                
                candidates[q_type].append({
                    "id": meta["name"],
                    "global_id": global_id,
                    "label_hint": meta.get("ovlabel", ""),
                    "score": float(score),
                    "text": evidence_text,
                    "subquery_type": q_type,
                    "description": description
                })
                if len(candidates[q_type]) >= top_k:
                    break
                
        return candidates

    def _search_single_modality(self, index, id_array, query_vec, top_k):
        if index is None or id_array is None or query_vec is None:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        k = min(top_k, index.ntotal)
        scores, indices = index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            gid = int(id_array[idx])
            results.append({"global_id": gid, "score": float(score)})
        return results

    def retrieve_channel_B(self, current_sample_features, top_k=1, exclude_sample_id=None):
        """
        通道 B (多模态通道)：寻找感知上最相似的样本 (用于 SupportFusion)
        该通道不受 Agent-Q 影响，纯看音视频向量相似度。
        
        Args:
            current_sample_features: numpy array，当前样本的多模态特征向量
            top_k: 取 top k 用于特征融合 (按照方案建议 Top-1 或 Top-3)
            
        Returns:
            fusion_partner_ids: 检索到的最相似样本的 ID 列表
        """
        if not self.multimodal_ready:
            raise RuntimeError(
                "Channel B multimodal index is not loaded. Please pass a valid multimodal_index_dir "
                "built by retrieval/faiss/build_mercaptionplus_faiss.py"
            )
        if current_sample_features is None:
            raise ValueError(
                "Channel B retrieval requires current_sample_features. "
                "Expected a dict with keys like 'video' and/or 'audio'."
            )

        if isinstance(current_sample_features, dict):
            video_vec = self._normalize_vec(current_sample_features.get("video"))
            audio_vec = self._normalize_vec(current_sample_features.get("audio"))
        else:
            video_vec = self._normalize_vec(current_sample_features)
            audio_vec = None

        score_map = {}
        for result in self._search_single_modality(self.video_index, self.video_ids, video_vec, top_k * 3):
            score_map[result["global_id"]] = score_map.get(result["global_id"], 0.0) + 0.5 * result["score"]
        for result in self._search_single_modality(self.audio_index, self.audio_ids, audio_vec, top_k * 3):
            score_map[result["global_id"]] = score_map.get(result["global_id"], 0.0) + 0.5 * result["score"]

        if not score_map:
            return []

        ranked = sorted(score_map.items(), key=lambda x: -x[1])
        partner_ids = []
        for global_id, _ in ranked:
            meta = self.multimodal_meta.get(global_id, {})
            name = meta.get("name")
            if exclude_sample_id and name == exclude_sample_id:
                continue
            partner_ids.append(name if name else str(global_id))
            if len(partner_ids) >= top_k:
                break
        return partner_ids
