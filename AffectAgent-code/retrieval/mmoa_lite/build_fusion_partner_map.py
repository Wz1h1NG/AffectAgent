import os
import sys
import json
import argparse

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from my_affectgpt.datasets.datasets.mer2023 import MER2023_Dataset
from my_affectgpt.datasets.datasets.mer2024 import MER2024_Dataset
from my_affectgpt.datasets.datasets.meld import MELD_Dataset
from my_affectgpt.datasets.datasets.iemocap import IEMOCAPFour_Dataset
from retrieval.mmoa_lite.retriever_service import DoubleChannelRetriever
from retrieval.faiss.build_mercaptionplus_faiss import (
    build_runtime_cfg,
    resolve_ckpt3_path,
    build_model_and_chat,
    build_processors_for_dataset,
    resolve_face_or_frame,
    get_video_priority,
    pool_and_normalize,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build Phase 1c fusion_partner_map for MMOA-Lite")
    parser.add_argument("--cfg-path", required=True, help="Path to AffectGPT yaml config")
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Override config options, e.g. model.ckpt_3=xxx",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mer2023",
        help="Target dataset (mer2023, mer2024, meld, iemocap)",
    )
    parser.add_argument(
        "--multimodal-index-dir",
        type=str,
        default="retrieval/faiss/artifacts/mercaptionplus",
        help="Path to Channel B multimodal FAISS artifacts built by build_mercaptionplus_faiss.py",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="retrieval/mmoa_lite/artifacts/sft_data/fusion_partner_map.json",
        help="Output JSON path for {sample_id: fusion_partner_id}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Channel B retrieval depth before choosing the first valid partner",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Only process first N samples for debugging (-1 for all)",
    )
    return parser.parse_args()


def load_dataset(dataset_name):
    if dataset_name == "mer2023":
        return MER2023_Dataset()
    if dataset_name == "mer2024":
        return MER2024_Dataset()
    if dataset_name == "meld":
        return MELD_Dataset()
    if dataset_name == "iemocap":
        return IEMOCAPFour_Dataset()
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")


def extract_current_sample_features(dataset, chat, face_or_frame, sample):
    video_path = dataset._get_video_path(sample) if hasattr(dataset, "_get_video_path") else None
    audio_path = dataset._get_audio_path(sample) if hasattr(dataset, "_get_audio_path") else None
    face_path = dataset._get_face_path(sample) if hasattr(dataset, "_get_face_path") else None

    vp = video_path if video_path and os.path.exists(video_path) else None
    ap = audio_path if audio_path and os.path.exists(audio_path) else None
    fp = face_path if face_path and os.path.exists(face_path) else None

    sample_data = dataset.read_frame_face_audio_text(
        video_path=vp,
        face_npy=fp,
        audio_path=ap,
        image_path=None,
    )

    with torch.no_grad():
        _, audio_llms = chat.postprocess_audio(sample_data)
        _, frame_llms = chat.postprocess_frame(sample_data)
        _, face_llms = chat.postprocess_face(sample_data)

    video_priority = get_video_priority(face_or_frame)
    if video_priority == "face":
        video_llms = face_llms if face_llms is not None else frame_llms
    else:
        video_llms = frame_llms if frame_llms is not None else face_llms

    return {
        "video": pool_and_normalize(video_llms),
        "audio": pool_and_normalize(audio_llms),
    }


def build_token_store_ref(multimodal_index_dir, sample_id, global_id):
    token_store_path = os.path.abspath(os.path.join(multimodal_index_dir, "token_store", "tokens.h5"))
    id_to_row_path = os.path.abspath(os.path.join(multimodal_index_dir, "token_store", "id_to_row.json"))
    if not os.path.isfile(token_store_path) or not os.path.isfile(id_to_row_path):
        raise FileNotFoundError(
            f"Missing token store artifacts under {multimodal_index_dir}. Expected token_store/tokens.h5 and token_store/id_to_row.json"
        )
    if global_id is None:
        raise ValueError(f"global_id is required to build partner mm_ref for sample {sample_id}")
    return {
        "source": "multimodal_token_store",
        "sample_id": sample_id,
        "global_id": int(global_id),
        "token_store_path": token_store_path,
        "id_to_row_path": id_to_row_path,
    }


def override_with_test_split(dataset):
    if hasattr(dataset, "read_test_names") and hasattr(dataset, "get_test_name2gt"):
        try:
            test_names = dataset.read_test_names()
            test_name2gt = dataset.get_test_name2gt()
            test_annotation = []
            for name in test_names:
                item = {"name": name}
                if hasattr(dataset, "name2subtitle"):
                    item["subtitle"] = dataset.name2subtitle.get(name, "")
                gt = test_name2gt.get(name, "")
                item["onehot"] = gt
                item["discrete"] = gt
                item["ovlabel"] = gt
                test_annotation.append(item)
            dataset.annotation = test_annotation
            print(f"[INFO] Dynamically overridden dataset.annotation with TEST split ({len(test_annotation)} samples).")
        except Exception as e:
            print(f"[WARN] Failed to override with test split: {e}")

def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cfg = build_runtime_cfg(args)
    ckpt3_path = resolve_ckpt3_path(cfg, args.cfg_path)
    cfg.model_cfg.ckpt_3 = ckpt3_path

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is requested but not available.")

    _, chat = build_model_and_chat(cfg, args.device)

    dataset = load_dataset(args.dataset)
    override_with_test_split(dataset)

    face_or_frame = resolve_face_or_frame(cfg.datasets_cfg)
    dataset.needed_data = dataset.get_needed_data(face_or_frame)
    build_processors_for_dataset(dataset, cfg)

    retriever = DoubleChannelRetriever(multimodal_index_dir=args.multimodal_index_dir)
    multimodal_name_to_global_id = {
        item.get("name"): gid for gid, item in retriever.multimodal_meta.items() if item.get("name")
    }

    records = dataset.annotation
    if args.max_samples > 0:
        records = records[:args.max_samples]

    fusion_partner_map = {}
    missing_samples = []

    for sample in tqdm(records, desc="Building fusion partner map"):
        sample_id = sample["name"]
        try:
            current_sample_features = extract_current_sample_features(dataset, chat, face_or_frame, sample)
            partner_candidates = retriever.retrieve_channel_B(current_sample_features, top_k=max(args.top_k, 3))
            fusion_partner_id = next((pid for pid in partner_candidates if pid != sample_id), None)
            if fusion_partner_id is None and partner_candidates:
                fusion_partner_id = partner_candidates[0]
            if fusion_partner_id is None:
                missing_samples.append(sample_id)
                continue
            partner_global_id = multimodal_name_to_global_id.get(fusion_partner_id)
            fusion_partner_map[sample_id] = {
                "fusion_partner_id": fusion_partner_id,
                "partner_global_id": partner_global_id,
                "partner_mm_ref": build_token_store_ref(args.multimodal_index_dir, fusion_partner_id, partner_global_id),
            }
        except Exception as exc:
            print(f"[WARN] failed to build fusion partner for {sample_id}: {exc}")
            missing_samples.append(sample_id)

    if missing_samples:
        preview = ", ".join(missing_samples[:10])
        raise RuntimeError(
            f"Failed to build fusion partners for {len(missing_samples)} samples. Preview: {preview}"
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(fusion_partner_map, f, ensure_ascii=False, indent=2)

    print(f"fusion_partner_map saved to {args.output_path} ({len(fusion_partner_map)} samples)")


if __name__ == "__main__":
    main()
