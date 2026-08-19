import argparse
import glob
import json
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss is required. Please install faiss-cpu or faiss-gpu first."
    ) from exc

# Registry side-effects (do not remove)
from my_affectgpt.tasks import *  # noqa: F401,F403
from my_affectgpt.models import *  # noqa: F401,F403
from my_affectgpt.runners import *  # noqa: F401,F403
from my_affectgpt.processors import *  # noqa: F401,F403
from my_affectgpt.datasets.builders import *  # noqa: F401,F403

from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
from my_affectgpt.processors.base_processor import BaseProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MER-Caption+ multimodal FAISS indexes with AffectGPT vectors"
    )
    parser.add_argument("--cfg-path", required=True, help="Path to AffectGPT yaml config")
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Override config options, e.g. model.ckpt_3=xxx datasets.mercaptionplus.face_or_frame=multiframe_audio_frame_text",
    )
    parser.add_argument(
        "--output-root",
        default="retrieval/faiss/artifacts/mercaptionplus",
        help="Output directory for vectors/indexes/metadata",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--text-max-features",
        type=int,
        default=4096,
        help="Max features for TF-IDF text vectors",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Only process first N samples for debugging (-1 for all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output root if exists",
    )
    return parser.parse_args()


def is_empty_like(value: Any) -> bool:
    if value is None:
        return True
    v = str(value).strip()
    return v in {"", "xxx", "None", "null"}


def first_existing_file(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def first_existing_dir(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isdir(p):
            return os.path.abspath(p)
    return None


def infer_project_root(cfg_path: str) -> str:
    cfg_abs = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_abs)
    if os.path.basename(cfg_dir) == "train_configs":
        return os.path.dirname(cfg_dir)
    return os.getcwd()


def search_for_ckpt_root(root_candidates: List[str]) -> str:
    if len(root_candidates) == 0:
        return ""

    max_count = 0
    target_root = ""
    for root in root_candidates:
        checkpoint_files = [
            p
            for p in os.listdir(root)
            if p.startswith("checkpoint_") and p.endswith(".pth")
        ]
        if len(checkpoint_files) == 0:
            checkpoint_files = [p for p in os.listdir(root) if p.endswith(".pth")]
        count = len(checkpoint_files)
        print(f"[ckpt-scan] {root} => {count}")
        if count > max_count:
            max_count = count
            target_root = root

    if target_root:
        print("[ckpt-scan] ================================================")
        print(f"[ckpt-scan] Target root: {target_root}")
        print(f"[ckpt-scan] Epoch range (estimated): 0-{max_count-1}")
        checkpoint_files = sorted(glob.glob(os.path.join(target_root, "checkpoint*.pth")))
        if len(checkpoint_files) == 0:
            checkpoint_files = sorted(glob.glob(os.path.join(target_root, "*.pth")))
        if checkpoint_files:
            file_stat = Path(checkpoint_files[-1]).stat()
            creation_time = datetime.fromtimestamp(file_stat.st_ctime)
            print(f"[ckpt-scan] Last checkpoint creation time: {creation_time}")
        print("[ckpt-scan] ================================================")

    return target_root


def get_ckpt3_candidates(ckpt3_root: str, inference_cfg: Any) -> List[str]:
    if not ckpt3_root or not os.path.isdir(ckpt3_root):
        raise FileNotFoundError(f"Invalid ckpt root: {ckpt3_root}")

    test_epoch = str(inference_cfg.get("test_epoch", "xxx"))
    test_epochs = str(inference_cfg.get("test_epochs", "xxx-xxx"))
    skip_epoch = int(inference_cfg.get("skip_epoch", 1))

    if test_epoch != "xxx":
        cur_epoch = int(test_epoch)
        ckpts = glob.glob(os.path.join(ckpt3_root, f"*{cur_epoch:06d}*.pth"))
        if len(ckpts) != 1:
            raise RuntimeError(
                f"Expected exactly one ckpt for epoch={cur_epoch}, found {len(ckpts)} in {ckpt3_root}"
            )
        return [ckpts[0]]

    if test_epochs == "xxx-xxx":
        ckpts = sorted(glob.glob(os.path.join(ckpt3_root, "*.pth")))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint .pth found in {ckpt3_root}")
        return [ckpts[-1]]

    start_epoch, end_epoch = test_epochs.split("-")
    whole_ckpts: List[str] = []
    for cur_epoch in range(int(start_epoch), int(end_epoch) + 1):
        if cur_epoch % skip_epoch == 0:
            ckpts = glob.glob(os.path.join(ckpt3_root, f"*{cur_epoch:06d}*.pth"))
            if len(ckpts) != 1:
                raise RuntimeError(
                    f"Expected exactly one ckpt for epoch={cur_epoch}, found {len(ckpts)} in {ckpt3_root}"
                )
            whole_ckpts.append(ckpts[0])

    if not whole_ckpts:
        raise FileNotFoundError(
            f"No checkpoint selected by test_epochs={test_epochs}, skip_epoch={skip_epoch}"
        )
    return whole_ckpts


def resolve_ckpt3_path(cfg: Config, cfg_path: str) -> str:
    model_cfg = cfg.model_cfg
    project_root = infer_project_root(cfg_path)
    cwd = os.getcwd()

    explicit_ckpt3 = model_cfg.get("ckpt_3", "")
    if not is_empty_like(explicit_ckpt3):
        explicit_ckpt3 = str(explicit_ckpt3)
        ckpt3_path = first_existing_file(
            [
                explicit_ckpt3,
                os.path.join(project_root, explicit_ckpt3),
                os.path.join(cwd, explicit_ckpt3),
            ]
        )
        if ckpt3_path is None:
            raise FileNotFoundError(f"model.ckpt_3 does not exist: {explicit_ckpt3}")
        print(f"[ckpt] Using explicit model.ckpt_3: {ckpt3_path}")
        return ckpt3_path

    inference_cfg = cfg.inference_cfg
    if inference_cfg is None:
        raise RuntimeError("inference config is required for automatic checkpoint search")

    ckpt_root = inference_cfg.get("ckpt_root", "")
    ckpt_name = inference_cfg.get("ckpt_name", "")
    cfg_name = os.path.basename(cfg_path).rsplit(".", 1)[0]

    if not is_empty_like(ckpt_root):
        ckpt_root = str(ckpt_root)
        resolved = first_existing_dir(
            [
                ckpt_root,
                os.path.join(project_root, ckpt_root),
                os.path.join(cwd, ckpt_root),
            ]
        )
        ckpt3_root = resolved if resolved is not None else ckpt_root
    elif not is_empty_like(ckpt_name):
        ckpt_name = str(ckpt_name)
        ckpt3_root = first_existing_dir(
            [
                os.path.join(project_root, "output", cfg_name, ckpt_name),
                os.path.join(cwd, "output", cfg_name, ckpt_name),
                os.path.join("output", cfg_name, ckpt_name),
            ]
        )
        if ckpt3_root is None:
            ckpt3_root = os.path.join("output", cfg_name, ckpt_name)
    else:
        root_candidates: List[str] = []
        output_roots = [
            os.path.join(project_root, "output", cfg_name),
            os.path.join(cwd, "output", cfg_name),
            os.path.join("output", cfg_name),
        ]
        for out_root in output_roots:
            if not os.path.isdir(out_root):
                continue

            # inference_hybird default strategy
            root_candidates.extend(glob.glob(os.path.join(out_root, cfg_name + "*")))

            # fallback 1: checkpoints directly under output/<cfg_name>
            direct_ckpts = glob.glob(os.path.join(out_root, "checkpoint*.pth"))
            if len(direct_ckpts) == 0:
                direct_ckpts = glob.glob(os.path.join(out_root, "*.pth"))
            if len(direct_ckpts) > 0:
                root_candidates.append(out_root)

            # fallback 2: recursive search for checkpoint files
            recursive_ckpts = glob.glob(os.path.join(out_root, "**", "checkpoint*.pth"), recursive=True)
            for ckpt_file in recursive_ckpts:
                root_candidates.append(os.path.dirname(ckpt_file))

        # de-duplicate while preserving order
        dedup_candidates: List[str] = []
        seen = set()
        for c in root_candidates:
            c_abs = os.path.abspath(c)
            if c_abs not in seen and os.path.isdir(c_abs):
                seen.add(c_abs)
                dedup_candidates.append(c_abs)

        root_candidates = dedup_candidates
        ckpt3_root = search_for_ckpt_root(root_candidates)

    if is_empty_like(ckpt3_root):
        raise FileNotFoundError(
            "Auto checkpoint search failed: cannot resolve ckpt3_root. "
            "Set --options model.ckpt_3=... or configure inference.ckpt_root/ckpt_name. "
            f"project_root={project_root}, cwd={cwd}, cfg_name={cfg_name}"
        )

    candidates = get_ckpt3_candidates(str(ckpt3_root), inference_cfg)
    if len(candidates) > 1:
        print(f"[ckpt] Multiple ckpt candidates found ({len(candidates)}), using the last one.")
    ckpt3_path = candidates[-1]

    if not os.path.isfile(ckpt3_path):
        raise FileNotFoundError(f"Resolved ckpt path does not exist: {ckpt3_path}")

    print(f"[ckpt] Auto-resolved model.ckpt_3: {ckpt3_path}")
    return ckpt3_path


@dataclass
class SampleFeature:
    global_id: int
    name: str
    subtitle: str
    reason: str
    ovlabel: str
    discrete_label: str
    text_raw: str
    video_path: str
    audio_path: str
    face_path: str
    has_video: bool
    has_audio: bool
    has_text: bool
    video_vector: Optional[np.ndarray]
    audio_vector: Optional[np.ndarray]
    video_tokens: Optional[np.ndarray]      # [T_v, D] raw token sequence for embedding injection
    audio_tokens: Optional[np.ndarray]      # [T_a, D] raw token sequence for embedding injection


DISCRETE_LABEL_KEYWORDS = {
    "happy":    ["happy", "happiness", "joy", "joyful", "content", "excited", "relieved", "cheerful", "delighted", "amused", "pleased"],
    "angry":    ["angry", "anger", "frustrated", "irritated", "furious", "rage", "annoyed", "mad"],
    "sad":      ["sad", "sadness", "depressed", "grief", "sorrow", "melancholy", "unhappy", "gloomy", "heartbroken"],
    "neutral":  ["neutral", "calm", "indifferent", "composed", "unemotional"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed", "stunned"],
    "fear":     ["fear", "afraid", "scared", "anxious", "terrified", "nervous", "worried", "panic"],
    "disgust":  ["disgust", "disgusted", "contempt", "repulsed", "revolted"],
}


def derive_discrete_label(ovlabel: str) -> str:
    """Map open-vocabulary ovlabel to a standard discrete emotion label via keyword matching."""
    if not ovlabel:
        return "unknown"
    ov_lower = ovlabel.lower()
    best_label = "unknown"
    best_pos = len(ov_lower) + 1
    for label, keywords in DISCRETE_LABEL_KEYWORDS.items():
        for kw in keywords:
            pos = ov_lower.find(kw)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_label = label
    return best_label


def extract_raw_tokens(llm_tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """Extract raw token sequence [T, D] without pooling, for embedding injection."""
    if llm_tensor is None:
        return None
    with torch.no_grad():
        x = llm_tensor.detach().float().cpu()
    if x.ndim == 3:
        return x[0].numpy().astype(np.float32)      # [T, D]
    elif x.ndim == 2:
        return x.numpy().astype(np.float32)          # [T, D]
    else:
        return x.reshape(-1).unsqueeze(0).numpy().astype(np.float32)  # [1, D]


def ensure_output_dirs(output_root: str, overwrite: bool) -> Dict[str, str]:
    if os.path.exists(output_root):
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_root}. Use --overwrite to continue."
            )
    os.makedirs(output_root, exist_ok=True)

    paths = {
        "vector": os.path.join(output_root, "vector"),
        "index": os.path.join(output_root, "index"),
        "meta": os.path.join(output_root, "meta"),
        "token_store": os.path.join(output_root, "token_store"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def build_runtime_cfg(args: argparse.Namespace) -> Config:
    return Config(args)


def resolve_face_or_frame(datasets_cfg: Any) -> str:
    if "mercaptionplus" not in datasets_cfg:
        raise KeyError("datasets.mercaptionplus is missing in config")
    face_or_frame = datasets_cfg["mercaptionplus"].face_or_frame
    return str(face_or_frame)


def build_model_and_chat(cfg: Config, device: str) -> Tuple[Any, Chat]:
    model_cfg = cfg.model_cfg
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device).eval()
    chat = Chat(model, model_cfg, device=device)
    return model, chat


def build_processors_for_dataset(dataset: MERCaptionPlus_Dataset, cfg: Config) -> None:
    model_cfg = cfg.model_cfg
    inference_cfg = cfg.inference_cfg

    dataset.vis_processor = BaseProcessor()
    dataset.img_processor = BaseProcessor()

    vis_processor_cfg = inference_cfg.get("vis_processor") if inference_cfg is not None else None
    img_processor_cfg = inference_cfg.get("img_processor") if inference_cfg is not None else None

    if vis_processor_cfg is None:
        vis_processor_cfg = model_cfg.get("vis_processor")
    if img_processor_cfg is None:
        img_processor_cfg = model_cfg.get("img_processor")

    if vis_processor_cfg is not None:
        dataset.vis_processor = registry.get_processor_class(vis_processor_cfg.train.name).from_config(
            vis_processor_cfg.train
        )
    if img_processor_cfg is not None:
        dataset.img_processor = registry.get_processor_class(img_processor_cfg.train.name).from_config(
            img_processor_cfg.train
        )

    dataset.n_frms = model_cfg.vis_processor.train.n_frms


def get_video_priority(face_or_frame: str) -> str:
    """
    Follow training config route:
    - startswith('multiface') or startswith('face'): use face
    - startswith('multiframe') or startswith('frame'): use frame
    - fallback: frame
    """
    fof = face_or_frame.lower()
    if fof.startswith("multiface") or fof.startswith("face"):
        return "face"
    if fof.startswith("multiframe") or fof.startswith("frame"):
        return "frame"
    return "frame"


def pool_and_normalize(llm_tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if llm_tensor is None:
        return None

    with torch.no_grad():
        x = llm_tensor.detach().float().cpu()

    # Expected shapes: [1, T, D] or [1, D]
    if x.ndim == 3:
        x = x.mean(dim=1)
    elif x.ndim == 2:
        pass
    else:
        x = x.reshape(1, -1)

    vec = x[0].numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_text_raw(subtitle: str, reason: str, ovlabel: str) -> str:
    subtitle = subtitle or ""
    reason = reason or ""
    ovlabel = ovlabel or ""
    return f"{subtitle}\n{reason}\n{ovlabel}".strip()


def safe_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def extract_multimodal_vectors(
    dataset: MERCaptionPlus_Dataset,
    chat: Chat,
    face_or_frame: str,
    max_samples: int,
) -> List[SampleFeature]:
    records: List[SampleFeature] = []

    video_priority = get_video_priority(face_or_frame)
    total = len(dataset.annotation)
    if max_samples > 0:
        total = min(total, max_samples)

    for idx in tqdm(range(total), desc="Extracting AffectGPT vectors"):
        sample = dataset.annotation[idx]

        name = sample["name"]
        subtitle = sample.get("subtitle", "")
        reason = sample.get("description", "")
        ovlabel = sample.get("ovlabel", "")

        video_path = dataset._get_video_path(sample)
        audio_path = dataset._get_audio_path(sample)
        face_path = dataset._get_face_path(sample)

        video_path_used = video_path if safe_exists(video_path) else None
        audio_path_used = audio_path if safe_exists(audio_path) else None
        face_path_used = face_path if safe_exists(face_path) else None

        try:
            sample_data = dataset.read_frame_face_audio_text(
                video_path=video_path_used,
                face_npy=face_path_used,
                audio_path=audio_path_used,
                image_path=None,
            )

            with torch.no_grad():
                _, audio_llms = chat.postprocess_audio(sample_data)
                _, frame_llms = chat.postprocess_frame(sample_data)
                _, face_llms = chat.postprocess_face(sample_data)

            if video_priority == "face":
                video_llms = face_llms if face_llms is not None else frame_llms
            else:
                video_llms = frame_llms if frame_llms is not None else face_llms

            video_vec = pool_and_normalize(video_llms)
            audio_vec = pool_and_normalize(audio_llms)
            video_tok = extract_raw_tokens(video_llms)
            audio_tok = extract_raw_tokens(audio_llms)

        except Exception as exc:
            print(f"[WARN] failed sample={name}: {exc}")
            video_vec = None
            audio_vec = None
            video_tok = None
            audio_tok = None

        record = SampleFeature(
            global_id=idx,
            name=name,
            subtitle=subtitle,
            reason=reason,
            ovlabel=ovlabel,
            discrete_label=derive_discrete_label(ovlabel),
            text_raw=build_text_raw(subtitle, reason, ovlabel),
            video_path=video_path,
            audio_path=audio_path,
            face_path=face_path,
            has_video=video_vec is not None,
            has_audio=audio_vec is not None,
            has_text=True,
            video_vector=video_vec,
            audio_vector=audio_vec,
            video_tokens=video_tok,
            audio_tokens=audio_tok,
        )
        records.append(record)

    return records


def fit_text_vectors(records: List[SampleFeature], text_max_features: int) -> np.ndarray:
    texts = [r.text_raw for r in records]

    vectorizer = TfidfVectorizer(max_features=text_max_features)
    mat = vectorizer.fit_transform(texts)
    text_vectors = mat.toarray().astype(np.float32)

    norms = np.linalg.norm(text_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    text_vectors = text_vectors / norms
    return text_vectors


def stack_vectors_and_ids(
    records: List[SampleFeature],
    modality: str,
    text_vectors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    vectors: List[np.ndarray] = []
    ids: List[int] = []

    for r in records:
        if modality == "video" and r.video_vector is not None:
            vectors.append(r.video_vector)
            ids.append(r.global_id)
        elif modality == "audio" and r.audio_vector is not None:
            vectors.append(r.audio_vector)
            ids.append(r.global_id)
        elif modality == "text" and text_vectors is not None:
            vectors.append(text_vectors[r.global_id])
            ids.append(r.global_id)

    if len(vectors) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    x = np.stack(vectors, axis=0).astype(np.float32)
    i = np.array(ids, dtype=np.int64)
    return x, i


def build_and_save_faiss(vectors: np.ndarray, out_path: str) -> None:
    if vectors.shape[0] == 0:
        return

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    faiss.write_index(index, out_path)


def save_metadata(records: List[SampleFeature], out_jsonl: str) -> None:
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            item = {
                "global_id": r.global_id,
                "name": r.name,
                "subtitle": r.subtitle,
                "reason": r.reason,
                "ovlabel": r.ovlabel,
                "discrete_label": r.discrete_label,
                "video_path": r.video_path,
                "audio_path": r.audio_path,
                "face_path": r.face_path,
                "modality_mask": {
                    "video": r.has_video,
                    "audio": r.has_audio,
                    "text": r.has_text,
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_token_store(records: List[SampleFeature], out_dir: str) -> Dict[str, int]:
    """Save raw token sequences to HDF5 for embedding injection at retrieval time.

    Structure:
        tokens.h5/
            video_tokens  [N_video, T_v, D]   — padded to max T_v
            audio_tokens  [N_audio, T_a, D]   — padded to max T_a
            video_ids     [N_video]            — global_id for each row
            audio_ids     [N_audio]            — global_id for each row
    """
    video_records = [(r.global_id, r.video_tokens) for r in records if r.video_tokens is not None]
    audio_records = [(r.global_id, r.audio_tokens) for r in records if r.audio_tokens is not None]

    h5_path = os.path.join(out_dir, "tokens.h5")

    with h5py.File(h5_path, "w") as f:
        # Video tokens
        if video_records:
            max_t_v = max(tok.shape[0] for _, tok in video_records)
            dim = video_records[0][1].shape[1]
            v_data = np.zeros((len(video_records), max_t_v, dim), dtype=np.float32)
            v_ids = np.zeros(len(video_records), dtype=np.int64)
            for i, (gid, tok) in enumerate(video_records):
                v_data[i, :tok.shape[0], :] = tok
                v_ids[i] = gid
            f.create_dataset("video_tokens", data=v_data, compression="gzip", compression_opts=4)
            f.create_dataset("video_ids", data=v_ids)
            f.attrs["video_max_seq_len"] = max_t_v
            f.attrs["video_dim"] = dim

        # Audio tokens
        if audio_records:
            max_t_a = max(tok.shape[0] for _, tok in audio_records)
            dim = audio_records[0][1].shape[1]
            a_data = np.zeros((len(audio_records), max_t_a, dim), dtype=np.float32)
            a_ids = np.zeros(len(audio_records), dtype=np.int64)
            for i, (gid, tok) in enumerate(audio_records):
                a_data[i, :tok.shape[0], :] = tok
                a_ids[i] = gid
            f.create_dataset("audio_tokens", data=a_data, compression="gzip", compression_opts=4)
            f.create_dataset("audio_ids", data=a_ids)
            f.attrs["audio_max_seq_len"] = max_t_a
            f.attrs["audio_dim"] = dim

    # Build global_id → h5 row index mapping for fast lookup
    mapping = {}
    for i, (gid, _) in enumerate(video_records):
        mapping.setdefault(gid, {})["video_row"] = i
    for i, (gid, _) in enumerate(audio_records):
        mapping.setdefault(gid, {})["audio_row"] = i

    mapping_path = os.path.join(out_dir, "id_to_row.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)

    counts = {"video_tokens": len(video_records), "audio_tokens": len(audio_records)}
    print(f"    Token store saved: {h5_path}")
    print(f"    Video: {len(video_records)} samples, Audio: {len(audio_records)} samples")
    return counts


def save_id_mapping(ids: np.ndarray, records: List[SampleFeature], out_jsonl: str) -> None:
    id_to_name = {r.global_id: r.name for r in records}
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row_id, gid in enumerate(ids.tolist()):
            item = {
                "faiss_row_id": row_id,
                "global_id": int(gid),
                "name": id_to_name[int(gid)],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_summary(
    out_path: str,
    face_or_frame: str,
    counts: Dict[str, int],
    dims: Dict[str, int],
    cfg_path: str,
) -> None:
    obj = {
        "cfg_path": cfg_path,
        "face_or_frame": face_or_frame,
        "counts": counts,
        "dims": dims,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    output_paths = ensure_output_dirs(args.output_root, args.overwrite)

    cfg = build_runtime_cfg(args)
    ckpt3_path = resolve_ckpt3_path(cfg, args.cfg_path)
    cfg.model_cfg.ckpt_3 = ckpt3_path

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is requested but not available.")

    print("[1/7] Loading model and chat...")
    _, chat = build_model_and_chat(cfg, device)

    print("[2/7] Preparing MER-Caption+ dataset helper...")
    dataset = MERCaptionPlus_Dataset()
    face_or_frame = resolve_face_or_frame(cfg.datasets_cfg)
    dataset.needed_data = dataset.get_needed_data(face_or_frame)
    build_processors_for_dataset(dataset, cfg)

    print("[3/7] Extracting multimodal vectors from AffectGPT...")
    records = extract_multimodal_vectors(dataset, chat, face_or_frame, args.max_samples)

    print("[4/7] Building text vectors (TF-IDF)...")
    text_vectors = fit_text_vectors(records, args.text_max_features)

    print("[5/7] Writing vectors and FAISS indexes...")
    video_vectors, video_ids = stack_vectors_and_ids(records, "video")
    audio_vectors, audio_ids = stack_vectors_and_ids(records, "audio")
    text_vectors_all, text_ids = stack_vectors_and_ids(records, "text", text_vectors)

    np.save(os.path.join(output_paths["vector"], "video.npy"), video_vectors)
    np.save(os.path.join(output_paths["vector"], "audio.npy"), audio_vectors)
    np.save(os.path.join(output_paths["vector"], "text.npy"), text_vectors_all)
    np.save(os.path.join(output_paths["vector"], "video_ids.npy"), video_ids)
    np.save(os.path.join(output_paths["vector"], "audio_ids.npy"), audio_ids)
    np.save(os.path.join(output_paths["vector"], "text_ids.npy"), text_ids)

    build_and_save_faiss(video_vectors, os.path.join(output_paths["index"], "video.index"))
    build_and_save_faiss(audio_vectors, os.path.join(output_paths["index"], "audio.index"))
    build_and_save_faiss(text_vectors_all, os.path.join(output_paths["index"], "text.index"))

    print("[6/7] Saving token store and metadata...")
    save_id_mapping(video_ids, records, os.path.join(output_paths["meta"], "video_id_to_name.jsonl"))
    save_id_mapping(audio_ids, records, os.path.join(output_paths["meta"], "audio_id_to_name.jsonl"))
    save_id_mapping(text_ids, records, os.path.join(output_paths["meta"], "text_id_to_name.jsonl"))
    save_metadata(records, os.path.join(output_paths["meta"], "metadata.jsonl"))
    token_counts = save_token_store(records, output_paths["token_store"])

    # Discrete label distribution (for Agent 3 confusion matrix reference)
    label_dist = {}
    for r in records:
        label_dist[r.discrete_label] = label_dist.get(r.discrete_label, 0) + 1
    print(f"    Discrete label distribution: {json.dumps(label_dist, indent=2)}")

    counts = {
        "samples_total": len(records),
        "video_index_size": int(video_vectors.shape[0]),
        "audio_index_size": int(audio_vectors.shape[0]),
        "text_index_size": int(text_vectors_all.shape[0]),
        "token_store_video": token_counts.get("video_tokens", 0),
        "token_store_audio": token_counts.get("audio_tokens", 0),
    }
    dims = {
        "video_dim": int(video_vectors.shape[1]) if video_vectors.shape[0] > 0 else 0,
        "audio_dim": int(audio_vectors.shape[1]) if audio_vectors.shape[0] > 0 else 0,
        "text_dim": int(text_vectors_all.shape[1]) if text_vectors_all.shape[0] > 0 else 0,
    }
    save_summary(
        out_path=os.path.join(args.output_root, "build_summary.json"),
        face_or_frame=face_or_frame,
        counts=counts,
        dims=dims,
        cfg_path=args.cfg_path,
    )

    print("[7/7] Done.")
    print(json.dumps({"output_root": args.output_root, "counts": counts, "dims": dims, "label_dist": label_dist}, indent=2))


if __name__ == "__main__":
    main()
