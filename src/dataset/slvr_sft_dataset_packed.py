"""
    This file is adapted from InternVL [https://github.com/OpenGVLab/InternVL/tree/main]
    We adapted and simplified the PackedDataset and IterableSupervisedDataset for our SLVR finetuning
"""


import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset, IterableDataset

from functools import partial

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    SLVR_TOKEN,
    SEM_TOKEN
)
from transformers import TrainingArguments

from .data_utils import get_image_info, llava_to_openai_slvr, pad_sequence
import numpy as np
from PIL import Image
from typing import List, Tuple
import math

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

  
def make_packed_supervised_data_module_slvr(model_id, processor, data_args, training_args: TrainingArguments,latent_end_token=False):

    """Make dataset and collator for supervised fine-tuning."""

    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()

    # we assume meta data
    meta_data = json.load(open(data_args.data_path))

    datasets = []
    total_data_len = 0
    for meta in meta_data:
        iterable_sft_dataset = IterableSupervisedDatasetSLVR(
            data_path=meta['data_path'],
            image_folder=meta['image_folder'],
            ds_name=meta['ds_name'],
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode = training_args.enable_data_packing,    # set for packed dataset
            random_seed=data_args.random_seed,
            latent_end_token=latent_end_token
        )
        datasets.append(iterable_sft_dataset)
        total_data_len += len(iterable_sft_dataset)

    packed_train_dataset = PackedDataset(
        tokenizer=processor.tokenizer,
        datasets=datasets,
        # Get rank and world size from Hugging Face Trainer arguments
        data_rank=data_rank,
        data_world_size=data_world_size,
        # --- Configure your packing parameters ---
        max_packed_tokens=training_args.max_packed_tokens,
        max_buffer_size=100,
        # long_seq_cut=training_args.long_seq_cut,
        long_seq_threshold=training_args.long_seq_threshold,
        # Limiting the number of training data per device to avoid exploded tokens_per_device when pairing long with short
        max_instance_per_batch=training_args.max_instance_per_batch,
    )

    data_collator = PackedDataCollatorForSupervisedDatasetSLVR(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=packed_train_dataset,
        eval_dataset=None,
        data_collator=data_collator,), total_data_len

import torch
from torch.utils.data import IterableDataset

class IterableSupervisedDatasetSLVR(IterableDataset):  # ✅ 继承 IterableDataset
    """
    True iterable dataset that streams samples one by one from JSONL file.
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor,
        data_args,
        ds_name: str,
        model_id,
        data_rank=0,
        data_world_size=1,
        distributed_mode=True,
        random_seed=None,
        latent_end_token=False,
    ):
        super().__init__()
        self.data_path = data_path  # 仅保存路径
        self.image_folder = image_folder
        self.processor = processor
        self.data_args = data_args
        self.ds_name = ds_name
        self.model_id = model_id
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.distributed_mode = distributed_mode
        self.random_seed = random_seed
        self.latent_end_token = latent_end_token
        
        # 图像参数（保持不变）
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self._total_length = None
        self.worker_id = None
        self.num_workers = 1

    def _should_skip(self, line_idx):
        """Determine if current worker/rank should skip this line"""
        if line_idx % self.data_world_size != self.data_rank:
            return True
        if self.worker_id is not None and line_idx % self.num_workers != self.worker_id:
            return True
        return False

    def bbox_to_token_idxs(self, bboxes, image_grid_thw):
        """保持不变，复用你的原逻辑"""
        _, h, w = image_grid_thw[0].tolist()
        token_idxs = []
        for bbox in bboxes: 
            x0, y0, x1, y1 = bbox
            x0_grid = max(0, min(int(np.floor(x0 * w)), w-1))
            x1_grid = max(0, min(int(np.ceil(x1 * w)), w))
            y0_grid = max(0, min(int(np.floor(y0 * h)), h-1))
            y1_grid = max(0, min(int(np.ceil(y1 * h)), h))

            x0_token = x0_grid // 2
            x1_token = (x1_grid + 1) // 2
            y0_token = y0_grid // 2
            y1_token = (y1_grid + 1) // 2

            H2, W2 = h // 2, w // 2
            idxs = [
                int(yy * W2 + xx)
                for yy in range(y0_token, y1_token)
                for xx in range(x0_token, x1_token)
            ]
            token_idxs.append(idxs)
        return token_idxs

    def __iter__(self):
        # 获取 worker 信息
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        else:
            self.worker_id = 0
            self.num_workers = 1
            
        logger.info(f"[{self.ds_name}] Rank {self.data_rank}, Worker {self.worker_id} starts iterating.")
        
        line_idx = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self._should_skip(line_idx):
                    line_idx += 1
                    continue
                    
                try:
                    # 逐行解析 JSON，不加载全文件
                    sources = json.loads(line.strip())
                except Exception as e:
                    logger.warning(f"Skipping invalid JSON at line {line_idx}: {e}")
                    line_idx += 1
                    continue
                
                # >>> 复用你原有的处理逻辑 <<<
                is_video = False
                videos = None
                grid_key = "image_grid_thw"
                pixel_key = "pixel_values"
                
                image_files = sources["image"]
                if isinstance(image_files, str):
                    image_files = [image_files]

                images = []
                for image_file in image_files:
                    if not os.path.exists(image_file):
                        if not image_file.startswith("http"):
                            image_file = os.path.join(self.image_folder, image_file)
                    images.append(get_image_info(
                        image_file,
                        self.image_min_pixel,
                        self.image_max_pixel,
                        self.image_resized_w,
                        self.image_resized_h
                    ))

                # Extract SLVR tokens
                inputs_for_grid = self.processor(
                    text=[""], images=images, videos=videos,
                    padding=False, do_resize=False, return_tensors='pt'
                )
                image_grid_thw = inputs_for_grid['image_grid_thw']
                slvr_token_idxs_list = self.bbox_to_token_idxs(sources['bboxes'], image_grid_thw)

                text_embedding = None
                if "emb" in sources:
                    text_embedding = torch.tensor(sources["emb"], dtype=torch.float32)

                sources = copy.deepcopy(llava_to_openai_slvr(
                    sources['conversations'],
                    is_video=is_video,
                    slvr_token_idxs_list=slvr_token_idxs_list,
                    latent_end_token=self.latent_end_token
                ))

                all_input_ids = [] 
                all_labels = []
                all_pixel_values = []
                all_image_grid_thw = []
                all_second_gird = []

                # Qwen2-VL system message
                if len(SYSTEM_MESSAGE) > 0:
                    system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
                    system_message_input_ids = self.processor.tokenizer(
                        system_message, add_special_tokens=False, return_tensors='pt'
                    )['input_ids']
                    system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)
                    
                    all_input_ids.append(system_message_input_ids.squeeze(0))
                    all_labels.append(system_labels.squeeze(0))

                for j in range(0, len(sources), 2):
                    user_input = sources[j]
                    gpt_response = sources[j + 1]

                    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
                    
                    if DEFAULT_IMAGE_TOKEN in user_input:
                        inputs = self.processor(
                            text=[user_input], images=images, videos=videos,
                            padding=False, do_resize=False, return_tensors='pt'
                        )
                        prompt_input_ids = inputs['input_ids']
                        all_pixel_values.append(inputs[pixel_key])
                        all_image_grid_thw.append(inputs[grid_key])
                    else:
                        prompt_input_ids = self.processor.tokenizer(
                            user_input, add_special_tokens=False, padding=False, return_tensors='pt'
                        )['input_ids']

                    response_input_ids = self.processor.tokenizer(
                        gpt_response, add_special_tokens=False, padding=False, return_tensors='pt'
                    )['input_ids']

                    input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                    
                    # FIX: Mask labels after </answer> tag to prevent endless token generation
                    response_labels = response_input_ids.squeeze(0).clone()
                    
                    # Find the position of </answer> token in the response
                    answer_end_token_ids = []
                    try:
                        # Get token IDs for </answer>
                        answer_end_tokens = self.processor.tokenizer.encode("</answer>", add_special_tokens=False)
                        if answer_end_tokens:
                            answer_end_token_ids = answer_end_tokens
                    except:
                        pass
                    
                    # Search for </answer> in response and mask everything after it
                    if len(answer_end_token_ids) > 0:
                        answer_end_id = answer_end_token_ids[-1]
                        for i in range(len(response_labels) - len(answer_end_token_ids) + 1):
                            if all(response_labels[i + j] == answer_end_id for j in range(len(answer_end_token_ids))):
                                if i + len(answer_end_token_ids) < len(response_labels):
                                    response_labels[i + len(answer_end_token_ids):] = IGNORE_INDEX
                                break
                    
                    labels = torch.cat(
                        [
                            torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                            response_labels,
                        ],
                        dim=0,
                    )

                    all_input_ids.append(input_ids)
                    all_labels.append(labels)
                
                input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
                labels = torch.cat(all_labels, dim=0).to(torch.long)
                attention_mask = (input_ids > -1000000).to(torch.long)

                slvr_tokens = [
                    torch.tensor(group, dtype=torch.int) for group in slvr_token_idxs_list
                ]

                data_dict = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'slvr_tokens': slvr_tokens,
                    'input_lengths': torch.tensor([input_ids.size(0)]),
                }
                
                if text_embedding is not None:
                    data_dict['text_embedding'] = text_embedding
                else:
                    continue
                if all_pixel_values:
                    data_dict[pixel_key] = torch.cat(all_pixel_values, dim=0)
                    data_dict[grid_key] = torch.cat(all_image_grid_thw, dim=0)

                yield data_dict
                line_idx += 1
    def __len__(self):
        """为兼容性提供，但 IterableDataset 不应依赖 len()"""
        if self._total_length is None:
            # 快速估算：统计 JSONL 行数
            with open(self.data_path, 'rb') as f:
                self._total_length = sum(1 for _ in f)
        return self._total_length

"""
    Below is a adaptation of 
    https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset_packed.py
    We adopted a greedy data packing logic
"""

# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import bisect
import copy
import logging
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List,
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 4096,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = False,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
        # long_seq_cut: int = 25600,           # A single instance longer than this will be truncated
        long_seq_threshold: int = 6144,      # Instance longer than this will be individually processed
        max_instance_per_batch: int = 4,     # max num of instance per device
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data

        self.max_packed_tokens = max_packed_tokens
        # self.long_seq_cut = long_seq_cut
        self.long_seq_threshold = long_seq_threshold
        self.max_instance_per_batch = max_instance_per_batch

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        self.slvr_token_id = self.tokenizer.convert_tokens_to_ids(SLVR_TOKEN)
        self.slvr_text_token_id = self.tokenizer.convert_tokens_to_ids(SEM_TOKEN)
        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [len(d) for d in datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]


        self.datasets = [ds for ds in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]

        print("#"*42)
        print(f"Training with datasets and their weights:")
        for d,w in zip(datasets,self.dataset_weight):
            print(f"{d.ds_name}:\t{w}")
        print("#"*42)

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            'sample_info': {d.ds_name:0 for d in self.datasets},
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), f'deduplicated ds_name: {ds_name_list}'

        for ds in self.datasets:
            self._state_dict[ds.ds_name] = {}

        if get_rank() == 0:
            logger.info(
                f'Loaded dataset to pack: {ds_name_list}, '
                f'{self.num_images_expected=}, {self.max_packed_tokens=}, '
                f'{self.replacement=}, {self.allow_overflow=}',
            )

            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f'{ds.ds_name:<25}: {ds_w*100:.2f}%')
            temp = '\n'.join(temp)
            logger.info(
                f'Sampling prob for each dataset:\n{temp}'
            )

        if self.allow_empty_data:
            logger.warning('allow_empty_data is enabled, note that empty data may be generated!')

    def load_state_dict(self, state_dict, custom_infos=None):

        self.worker_custom_infos = custom_infos

        self._state_dict.update(state_dict)
        for ds in self.datasets:
            if ds.ds_name in self._state_dict:
                ds.load_state_dict(self._state_dict[ds.ds_name])
                logger.info(f'{ds.ds_name=} is resumed.')
            else:
                logger.warning(f'{ds.ds_name=} is not resumed.')

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        while True:
            try:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
                break  # Exit loop if successful
            except StopIteration:
                if self.replacement:
                    # logger.info(f'[Worker id {self.worker_id}] Dataset {self.datasets[current_dataset_idx].ds_name} is exhausted, restart it.')
                    try:
                        self.dataset_iter_list[current_dataset_idx] = iter(self.datasets[current_dataset_idx])
                        current_sample = next(self.dataset_iter_list[current_dataset_idx])
                        break
                    except:
                        # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                        self.datasets.pop(current_dataset_idx)
                        self.dataset_iter_list.pop(current_dataset_idx)
                        self.dataset_weight.pop(current_dataset_idx)

                        if len(self.datasets) == 0:
                            raise StopIteration
                        current_dataset_idx = np.random.choice(len(self.datasets))
                else:
                    # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                    self.datasets.pop(current_dataset_idx)
                    self.dataset_iter_list.pop(current_dataset_idx)
                    self.dataset_weight.pop(current_dataset_idx)

                    if len(self.datasets) == 0:
                        raise StopIteration
                    current_dataset_idx = np.random.choice(len(self.datasets))
            except:
                logger.error(f'worker_id{self.worker_id} data_rank={self.data_rank} data_world_size={self.data_world_size} Unexpected error!')
                # logger.error('Unexpected error!')
                if len(self.datasets) == 0:
                    raise StopIteration
                current_dataset_idx = np.random.choice(len(self.datasets))

        current_ds_name = self.datasets[current_dataset_idx].ds_name

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}

        meta_info = current_sample.pop('meta_info', {})
        self._state_dict[current_ds_name][self.worker_state_key].update(**meta_info)
        self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1
        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        # NOTE: use `bisect` to search might be faster
        #  deleted the condition on # of images

        find = False
        find_idx = -1

        # if we see a new sample > LST, we need it to be in the buffer list,
        # instead of concatenating it to any existing buffer
        if new_sample['input_ids'].size(0) >= self.long_seq_threshold:
            return None

        for buffer_idx, buffer in enumerate(buffer_list):
            num_merged_tokens = new_sample['input_ids'].size(0) + buffer['input_ids'].size(0)
            num_instance_buffer = buffer['input_lengths'].size(0)
            if num_instance_buffer + 1 <= self.max_instance_per_batch:
                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

                if self.allow_overflow and len(buffer_list) >= self.max_buffer_size // 2:
                    find = True
                    find_idx = buffer_idx

        if find:
            return buffer_list.pop(find_idx)
        else:
            return None

    def update_buffer(self, buffer, new_sample):
        if buffer is None:
            new_sample['data_index'] = torch.zeros_like(new_sample['input_ids'])
            return new_sample

        new_sample['data_index'] = torch.ones_like(new_sample['input_ids']) + buffer['data_index'][-1].item()

        assert buffer.keys() == new_sample.keys()
        for k in buffer:
            if k == 'slvr_tokens':
                buffer[k] = buffer[k] + new_sample[k]
            else:
                buffer[k] = torch.cat([buffer[k], new_sample[k]])
        return buffer

    @staticmethod
    def split_buffer(buffer, max_tokens, img_start_token_id, img_token_id, img_end_token_id,
                 slvr_token_id, slvr_text_token_id,long_seq_threshold, max_instance_per_batch):  
        if not long_seq_threshold:
            long_seq_threshold = max_tokens // 2

        def _image_is_splitted(input_ids, cut_idx):
            if cut_idx >= input_ids.size(0):
                return False
            else:
                is_image_start = input_ids[cut_idx].item() == img_start_token_id
                is_image_token = input_ids[cut_idx].item() == img_token_id
                is_image_end = input_ids[cut_idx].item() == img_end_token_id
                return is_image_start or is_image_token or is_image_end
        
        '''
            Handles long single-/multi- instance buffer differently
        '''
        # condition 1: single instance
        if buffer['data_index'][-1].item() == 0:
            # condition 1.1: single long instance
            if buffer['input_ids'].size(0) >= long_seq_threshold:

                '''cut_id is the idx of the first token to be dropped'''
                cut_id = min(max_tokens, buffer['input_ids'].size(0))

                if not _image_is_splitted(buffer['input_ids'], cut_id):
                    # count discarded slvr tokens before slicing

                    if slvr_token_id is not None:
                        num_discarded_slvr_tokens = (buffer['input_ids'][cut_id:] == slvr_token_id).sum().item()
                        cut_id_slvr = buffer['slvr_tokens'][0].size(0)-num_discarded_slvr_tokens
                    if slvr_text_token_id is not None:
                        num_discarded_slvr_text_tokens = (buffer['input_ids'][cut_id:] == slvr_text_token_id).sum().item()
                        # 假设text_embedding存储在buffer['text_embedding']
                        if 'text_embedding' in buffer and buffer['text_embedding'] is not None:
                            buffer['text_embedding'] = buffer['text_embedding'][:buffer['text_embedding'].size(0) - num_discarded_slvr_text_tokens]
                    for k in buffer:
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index']:
                            buffer[k] = buffer[k][:cut_id]
                        elif k in ['pixel_values', 'image_flags','image_grid_thw']:
                            buffer[k] = buffer[k]
                        elif k in ['slvr_tokens']:
                            buffer[k][0] = buffer[k][0][:cut_id_slvr]
                        elif k == 'text_embedding':
                            buffer[k] = buffer[k]
                        elif k in ['input_lengths']:
                            pass
                        else:
                            raise NotImplementedError(f'find unsupported keys: {k} from {buffer.keys()}')
                    # re-assign lengths
                    buffer['input_lengths'][0] = buffer['input_ids'].size(0)
                    buffer_ready = [buffer]
                    buffer_unready = []
                else:   # if image is getting cut, discard the overlong instance
                    buffer_ready = []
                    buffer_unready = []
            
            # condition 1.2: single short instance
            else:
                buffer_ready = []
                buffer_unready = [buffer]

        # condition 2: multi instance
        else:
            # condition 2.1: < maxToken AND < max_instance_per_batch
            if (buffer['input_ids'].size(0) < max_tokens) and (buffer['input_lengths'].size(0) < max_instance_per_batch):
                buffer_ready = []
                buffer_unready = [buffer]
            # condition 2.2: < maxToken AND == max_instance_per_batch
            elif (buffer['input_ids'].size(0) < max_tokens) and (buffer['input_lengths'].size(0) == max_instance_per_batch):
                buffer_ready = [buffer]
                buffer_unready = []
            # condition 2.3: otherwise
            else:
                buffer_ready = []
                buffer_unready = []
                while buffer['input_ids'].size(0) >= max_tokens:
                    buffer_right = {}
                    cut_idx_right_size = buffer['input_lengths'][-1].item()     # number of tokens to be cut from right side
                    image_cut_idx_right_size = buffer['image_grid_thw'][-1].prod()  # number of pixels to be cut from right side
                    for k in buffer:
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index']:
                            buffer_right[k] = buffer[k][-cut_idx_right_size:]
                            buffer[k] = buffer[k][:-cut_idx_right_size]
                        elif k in ['pixel_values', 'image_flags']:
                            buffer_right[k] = buffer[k][-image_cut_idx_right_size:]
                            buffer[k] = buffer[k][:-image_cut_idx_right_size]
                        elif k in ['slvr_tokens','image_grid_thw','input_lengths']:
                            buffer_right[k] = buffer[k][-1:]
                            buffer[k] = buffer[k][:-1]
                        elif k == 'text_embedding':
                            buffer_right[k] = buffer[k][-1:]
                            buffer[k] = buffer[k][:-1]
                        else:
                            raise NotImplementedError(f'find unsupported keys: {k} from {buffer.keys()}')
                    # if left buffer is longer
                    if buffer['input_ids'].size(0) >= buffer_right['input_ids'].size(0):
                        buffer_ready.append(buffer)
                        buffer = buffer_right
                    else:   # buffer_right is longer than the accumulated left
                        buffer_ready.append(buffer_right)

                # if buffer['input_ids'].size(0) <= max_tokens and PackedDataset.check_valid(buffer):
                buffer_unready.append(buffer)

        return buffer_ready, buffer_unready

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer):
        # NOTE: in-place operation

        buffer_ready, buffer_unready = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
            slvr_token_id=self.slvr_token_id,
            slvr_text_token_id=self.slvr_text_token_id,
            long_seq_threshold=self.long_seq_threshold,
            max_instance_per_batch=self.max_instance_per_batch
        )

        for each_buffer in buffer_ready:
            buffer_max_len_list.append(each_buffer)

        for each_buffer in buffer_unready:
            find_idx = len(buffer_list)
            num_tokens_new_sample = each_buffer['input_ids'].size(0)
            for buffer_idx in range(len(buffer_list)):
                if buffer_list[buffer_idx]['input_ids'].size(0) < num_tokens_new_sample:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        return buffer_list, buffer_max_len_list

    def print_log(self, iter_idx, buffer_list):
        if iter_idx % self.log_freq != 0:
            return

        if self._should_log():
            logger.info(
                f"{iter_idx=}, {len(buffer_list)=}, {self._state_dict['sample_info']}"
            )

    def __iter__(self):
        iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        if self._should_log():
            logger.info(f'Begin to iter, {len(buffer_list)=}')

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        rng = np.random.default_rng(seed=worker_id)

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f'work_state_{self.worker_id}'
        self.datasets = [d for d in self.datasets_orig]

        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            # if not isinstance(ds, (ImageTextPairDataset, InterleavedDataset)):
            ds.worker_id = worker_id
            ds.worker_state_key = f'work_state_{self.worker_id}'
            ds.num_workers = num_workers
            if self._should_log() and worker_id == 0:
                logger.info(f'set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}')

        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if 'buffer_list' in custom_infos and isinstance(custom_infos['buffer_list'], list):
                buffer_list = custom_infos['buffer_list']
                if self._should_log() and worker_id == 0:
                    logger.info(f'[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}')
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(
            f'{self.__class__.__name__} Rank {self.data_rank} '
            f'Worker {worker_id} begin to load data'
        )

        while True:
            self.dataset_weight = [w / sum(self.dataset_weight) for w in self.dataset_weight]
            current_dataset_idx = rng.choice(len(self.dataset_iter_list), p=self.dataset_weight)

            try:
                current_sample = self.next_data(current_dataset_idx)
            except:
                logger.info(f'All datasets are exhausted, begin to empty the buffer_list ({len(buffer_list)=})')
                while len(buffer_list) > 0:
                    yield buffer_list.pop(0)
                logger.info(f'buffer_list is empty! ({len(buffer_list)=})')
                return

            # it is guaranteed in self.find_buffer() that if current_sample is >= max_tokens,
            # it will not get concatenated to any existing buffer
            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            '''
                A greedy method to balance effifiency and memory safety:
                1. if buffer stacks up tp max_packed_tokens, it is poped
                2. if buffer has >= max_instance_per_batch, it is poped
                3. if a single sample is >= long_seq_thresh, it is poped as a single buffer

                This is intended to avoid a long seq paired with multiple short seqs since
                padding them will explode the memory
            '''
            buffer_list, buffer_max_len_list = self.update_buffer_list(buffer_list, buffer_max_len_list, buffer)

            while len(buffer_max_len_list) > 0:
                
                yield buffer_max_len_list.pop(0)

            while len(buffer_list) > self.max_buffer_size:
                
                yield buffer_list.pop(0)

            self.print_log(iter_idx=iter_idx, buffer_list=buffer_list)
            iter_idx += 1

    # @staticmethod
    # def get_cu_seqlens_and_indexes(
    #     data_index: torch.LongTensor,  # (seq_len,)
    #     input_ids: torch.LongTensor,   # (seq_len,)
    #     labels: torch.LongTensor,   # (seq_len,)
    #     len2weight: callable,
    # ):
    #     indexes = []
    #     cu_seqlens = [0]
    #     loss_weight = []

    #     start = data_index.min()
    #     end = data_index.max() + 1
    #     for i in range(start, end):
    #         num_tokens = (data_index == i).sum().item()
    #         indexes.extend(list(range(num_tokens)))
    #         cu_seqlens.append(cu_seqlens[-1] + num_tokens)
    #         assert num_tokens > 0

    #         curr_data_index = data_index[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
    #         assert (curr_data_index == i).all(), data_index

    #         curr_labels = labels[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
    #         num_effective_tokens = (curr_labels != IGNORE_INDEX).sum().item()
    #         loss_weight.extend([len2weight(num_effective_tokens)] * num_tokens)

    #     assert len(indexes) == data_index.size(0), f'{len(indexes)=}, {data_index.size(0)=}'

    #     loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
    #     return cu_seqlens, indexes, loss_weight


WARNING_CNT = defaultdict(int)

class PackedDataCollatorForSupervisedDatasetSLVR(object):

    def __init__(self,pad_token_id):
        self.pad_token_id = pad_token_id 
    
    def __call__(self, features):
        # features is supposed to be a list of packed items
        # We will set batch_per_device to 1

        if not isinstance(features, list):
            features = [features]

        #  Unpack all sequences
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        all_slvr_tokens = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_text_embedding = []
        for feature in features:
            # each feature is a packed mini-batch
            all_input_ids.extend(torch.split(feature["input_ids"], feature["input_lengths"].tolist()))
            all_attention_masks.extend(torch.split(feature["attention_mask"], feature["input_lengths"].tolist()))
            all_labels.extend(torch.split(feature["labels"], feature["input_lengths"].tolist()))

            all_slvr_tokens.extend(feature['slvr_tokens'])
            all_pixel_values.append(feature['pixel_values'])
            all_image_grid_thw.append(feature['image_grid_thw'])
            all_text_embedding.append(feature['text_embedding'])
        # pad all sequences
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        for i in range(len(all_input_ids)):
            seq_len = len(all_input_ids[i])
            padding_needed = max_len - seq_len

            # Pad on the right side.
            padded_input_ids.append(torch.nn.functional.pad(
                all_input_ids[i], (0, padding_needed), value=self.pad_token_id))
            
            padded_attention_masks.append(torch.nn.functional.pad(
                all_attention_masks[i], (0, padding_needed), value=0))
            
            padded_labels.append(torch.nn.functional.pad(
                all_labels[i], (0, padding_needed), value=IGNORE_INDEX))

        data_dict = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),

            "slvr_tokens": all_slvr_tokens,
            "pixel_values": torch.cat(all_pixel_values),
            "image_grid_thw": torch.cat(all_image_grid_thw),
            "text_embedding": torch.stack(all_text_embedding)
            }
        
        return data_dict



    # def bbox_to_token_idxs_manual(
    #         self,  
    #         images: List[Image.Image], 
    #         bboxes: List[Tuple[float, float, float, float]]) -> List[np.ndarray]:
    #         """
    #         Convert bounding box coordinates to visual token indices.
    #         ATTENTION: images suppose to only contain only one image!!! although its a list
    #         bboxes is the list of bboxes that is drawn on the only image
                
    #         Returns:
    #             List of token indices, optionally with grid coordinates
    #         """
    #         # Again, only one image
    #         img = images[0]
    #         patch_size = self.processor.image_processor.patch_size
    #         image_width = img.width
    #         image_height = img.height

    #         grid_height = image_height // patch_size
    #         grid_width = image_width // patch_size

    #         token_grid_height = grid_height // self.processor.image_processor.temporal_patch_size
    #         token_grid_width = grid_width // self.processor.image_processor.temporal_patch_size

    #         token_idx_list = []
    #         for bbox in bboxes:
    #             '''
    #                 Attention: 
    #                 Even if the bbox is normalized here, it is possible to mess up the cords 
    #                 as QWEN img processing will resize the image if its beyond/below max/min pixels.
    #                 I dont wanna modify their official code for img processing tbh. So please keep in mind that
    #                 THE BBOXES ARE SUPPOSED TO BE NORMALIZED
    #             '''

    #             x1, y1, x2, y2 = bbox
    #             if max(x1, y1, x2, y2) > 1.0:
    #                 x1 /= image_width
    #                 y1 /= image_height
    #                 x2 /= image_width
    #                 y2 /= image_height
                
    #             # Clamp coordinates to valid range
    #             x1, y1 = max(0, x1), max(0, y1)
    #             x2, y2 = min(1, x2), min(1, y2)
                
    #             # Convert to token grid coordinates
    #             # Map from image coordinates to token grid coordinates
    #             token_x1 = int(x1 * token_grid_width)
    #             token_y1 = int(y1 * token_grid_height)
    #             token_x2 = min(int(math.ceil(x2 * token_grid_width)), token_grid_width)
    #             token_y2 = min(int(math.ceil(y2 * token_grid_height)), token_grid_height)
                
    #             # Ensure we have at least one token
    #             if token_x2 <= token_x1:
    #                 token_x2 = token_x1 + 1
    #             if token_y2 <= token_y1:
    #                 token_y2 = token_y1 + 1
                
    #             # Generate token indices and grid coordinates
    #             token_indices = []
                
    #             for y in range(token_y1, token_y2):
    #                 for x in range(token_x1, token_x2):
    #                     # Convert 2D grid position to 1D token index
    #                     token_idx = y * token_grid_width + x
    #                     token_indices.append(token_idx)
    #             token_idx_list.append(np.array(token_indices))
            
    #         return token_idx_list
    

# def collate_fn_for_packed_slvr(
#     features,
#     pad_token_id: int,
# ):
#     if not isinstance(features, list):
#         features = [features]

#     #  Unpack all sequences
#     all_input_ids = []
#     all_attention_masks = []
#     all_labels = []

#     all_slvr_tokens = []
#     all_pixel_values = []
#     all_image_grid_thw = []

#     for feature in features:
#         # each feature is a packed mini-batch
#         all_input_ids.extend(torch.split(feature["input_ids"], feature["input_lengths"].tolist()))
#         all_attention_masks.extend(torch.split(feature["attention_mask"], feature["input_lengths"].tolist()))
#         all_labels.extend(torch.split(feature["labels"], feature["input_lengths"].tolist()))

#         all_slvr_tokens.extend(feature['slvr_tokens'])
#         all_pixel_values.append(feature['pixel_values'])
#         all_image_grid_thw.append(feature['image_grid_thw'])
    
#     # pad all sequences
#     max_len = max(len(seq) for seq in all_input_ids)
#     padded_input_ids = []
#     padded_attention_masks = []
#     padded_labels = []
#     for i in range(len(all_input_ids)):
#         seq_len = len(all_input_ids[i])
#         padding_needed = max_len - seq_len

#         # Pad on the right side.
#         padded_input_ids.append(torch.nn.functional.pad(
#             all_input_ids[i], (0, padding_needed), value=pad_token_id))
        
#         padded_attention_masks.append(torch.nn.functional.pad(
#             all_attention_masks[i], (0, padding_needed), value=0))
        
#         padded_labels.append(torch.nn.functional.pad(
#             all_labels[i], (0, padding_needed), value=IGNORE_INDEX))
    
#     input_ids = torch.stack(padded_input_ids)
#     attention_mask = torch.stack(padded_attention_masks)
#     labels = torch.stack(padded_labels)

#     pixel_values = torch.cat(all_pixel_values)
#     image_grid_thw = torch.cat(all_image_grid_thw)

#     # data_dict = {
#     #     "input_ids": input_ids.unsqueeze(0),
#     #     "attention_mask": attention_mask.unsqueeze(0),
#     #     "labels": labels.unsqueeze(0),

#     #     "slvr_tokens": all_slvr_tokens,
#     #     "pixel_values": pixel_values.unsqueeze(0),
#     #     "image_grid_thw": image_grid_thw.unsqueeze(0),
#     #     }    

#     data_dict = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,

#         "slvr_tokens": all_slvr_tokens,
#         "pixel_values": pixel_values,
#         "image_grid_thw": image_grid_thw,
#         }    

#     return data_dict
