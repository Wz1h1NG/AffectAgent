# *_*coding:utf-8 *_*
"""Portable paths for the public AffectAgent release.

Set AFFECTAGENT_MODEL_ROOT and AFFECTAGENT_DATA_ROOT to keep large models and
datasets outside the repository. Relative paths no longer depend on the shell's
current working directory.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
AFFECTGPT_ROOT = Path(os.environ.get("AFFECTAGENT_ROOT", PROJECT_ROOT)).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get("AFFECTAGENT_MODEL_ROOT", AFFECTGPT_ROOT / "models")).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("AFFECTAGENT_DATA_ROOT", AFFECTGPT_ROOT / "dataset")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("AFFECTAGENT_OUTPUT_ROOT", AFFECTGPT_ROOT / "output")).expanduser().resolve()

EMOTION_WHEEL_ROOT = str(AFFECTGPT_ROOT / "emotion_wheel")
RESULT_ROOT = str(OUTPUT_ROOT / "results")


#######################
## 所有模型的存储路径
#######################
PATH_TO_LLM = {
    'Qwen25': str(MODEL_ROOT / 'Qwen2.5-7B-Instruct'),
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': str(MODEL_ROOT / 'clip-vit-large-patch14'),

}

PATH_TO_AUDIO = {
    'HUBERT_LARGE': str(MODEL_ROOT / 'chinese-hubert-large'),
}

PATH_TO_BERT = str(MODEL_ROOT / 'bert-base-uncased')

#######################
## 所有数据集的存储路径
#######################
DATA_DIR = {
    'MER2025OV': str(DATA_ROOT / 'mer2025-dataset'),
    'MERCaptionPlus': str(DATA_ROOT / 'mer2025-dataset'),
    'OVMERD': str(DATA_ROOT / 'mer2025-dataset'),
    'MER2023': str(DATA_ROOT / 'mer2023-dataset-process'),
    'MER2024': str(DATA_ROOT / 'mer2024-dataset-process'),
    'IEMOCAPFour': str(DATA_ROOT / 'iemocap-process'),
    'CMUMOSI': str(DATA_ROOT / 'cmumosi-process'),
    'CMUMOSEI': str(DATA_ROOT / 'cmumosei-process'),
    'SIMS': str(DATA_ROOT / 'sims-process'),
    'SIMSv2': str(DATA_ROOT / 'simsv2-process'),
    'MELD': str(DATA_ROOT / 'meld-process'),
    'OVMERDPlus': str(DATA_ROOT / 'ovmerdplus-process'),
}

PATH_TO_RAW_AUDIO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'audio'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'audio'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'audio'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'audio'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subaudio'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'audio'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'audio'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'video'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'video'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'video'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'video'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideo-tgt'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subvideo'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subvideo_new'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'video'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subvideo'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'video_new'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'video'),
}
PATH_TO_RAW_FACE = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'openface_face'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'openface_face'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'openface_face'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'openface_face'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'openface_face'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'openface_face'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'openface_face'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'openface_face'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'subtitle_chieng.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'subtitle_chieng.csv'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'subtitle_chieng.csv'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription-engchi-polish.csv'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription-engchi-polish.csv'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription-engchi-polish.csv'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription-engchi-polish.csv'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'transcription-engchi-polish.csv'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'transcription-engchi-polish.csv'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'transcription-engchi-polish.csv'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'transcription_merge.csv'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'subtitle_eng.csv'),
}
PATH_TO_LABEL = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'track2_test.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'xxx'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'xxx'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'label_4way.npz'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'label.npz'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'label.npz'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'label.npz'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'label.npz'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'label.npz'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'label-6way.npz'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'ovlabel.csv'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
