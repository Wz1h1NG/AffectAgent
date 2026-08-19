import os
import tqdm
import copy
import json
import random
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Sequence

import decord
from decord import VideoReader

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from my_affectgpt.processors import transforms_video, AlproVideoTrainProcessor
from my_affectgpt.conversation.conversation_video import Conversation,SeparatorStyle
from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
from my_affectgpt.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio # 将上面的函数功能拆解为两个
from toolkit.utils.functions import string_to_list
import config

from toolkit.utils.read_files import *

#############################################################
## OVMERD+: 532 samples
#############################################################
class OVMERDPlus_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                       dataset_cfg=None, model_cfg=None):
        
        self.dataset = 'OVMERDPlus'
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data) # ['audio', 'frame', 'face']
        
        ################# 直接手动指定所有信息的存储路径 #################
        name2subtitle = {}
        subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset]
        df = pd.read_csv(subtitle_csv)
        for _, row in df.iterrows():
            name = row['name']
            subtitle = row['sentence']
            if pd.isna(subtitle): subtitle=""
            name2subtitle[name] = subtitle
        self.name2subtitle = name2subtitle

        ov_path = os.path.join(config.DATA_DIR[self.dataset], 'ovlabel.csv')
        name2openset = {}
        df = pd.read_csv(ov_path)
        for _, row in df.iterrows():
            name = row['name']
            openset = row['openset']
            openset = string_to_list(openset)
            if len(openset) == 0: openset = ['neutral']
            name2openset[name] = ", ".join(openset)
        self.name2openset = name2openset
        
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
        face_root= config.PATH_TO_RAW_FACE[self.dataset]
        ##################################################################

        # use base model initialize approach
        super().__init__(vis_processor=vis_processor, 
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=vis_root,
                         face_root=face_root,
                         wav_root=wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
        
    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.vis_root, sample['name'] + '.mp4')
        return full_video_fp
    
    def _get_audio_path(self, sample):
        full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
        return full_audio_fp

    def _get_face_path(self, sample):
        # full_face_fp = os.path.join(self.face_root, sample['name'], sample['name'] + '.npy')
        full_face_fp = os.path.join(self.face_root, sample['name'] + '.npy') # dataset-copy 过程中少了一个 subroot 
        return full_face_fp
    
    # for inference
    def read_test_names(self):
        subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset]
        test_names  = func_read_key_from_csv(subtitle_csv, 'name')
        assert len(test_names) == 532
        return test_names
    
    def get_test_name2gt(self):
        return self.name2openset