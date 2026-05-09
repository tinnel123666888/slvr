import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai_lvr, pad_sequence
import numpy as np
from PIL import Image
from typing import List, Tuple
import math

class SupervisedDatasetLVR(Dataset):
    """Dataset for supervised fine-tuning LVR model."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        latent_end_token=False,
    ):
        super(SupervisedDatasetLVR, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.latent_end_token = latent_end_token
        
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def make_bbox_masks_rgb(
        self,
        pil_imgs: List[Image.Image], 
        bboxes_norm: List[Tuple[float, float, float, float]]
    ) -> List[np.ndarray]:
        """
        Create RGB binary masks for multiple PIL images based on normalized bounding boxes.
        """
        assert len(pil_imgs) == len(bboxes_norm), "Images and bboxes lists must be same length"
        masks_rgb = []

        for img, bbox in zip(pil_imgs, bboxes_norm):
            w, h = img.size
            x_min, y_min, x_max, y_max = bbox

            xmin = int(round(x_min * w))
            ymin = int(round(y_min * h))
            xmax = int(round(x_max * w))
            ymax = int(round(y_max * h))

            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w), min(ymax, h)

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[ymin:ymax, xmin:xmax] = 1
            mask_rgb = np.stack([mask] * 3, axis=-1)
            masks_rgb.append(mask_rgb)

        return masks_rgb
    
    def bbox_to_token_idxs(
        self,
        images: List[Image.Image], 
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[np.ndarray]:
        image_masks = self.make_bbox_masks_rgb(images, bboxes)
        lvr_token_idxs_list = []
        
        for image_mask in image_masks:
            image_masks_processed, _ = self.processor.image_processor._preprocess(
                [image_mask],  # 注意：这里需要传入列表
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
                patch_size=self.processor.image_processor.patch_size,
                temporal_patch_size=self.processor.image_processor.temporal_patch_size,
                merge_size=self.processor.image_processor.merge_size,
                do_convert_rgb=False,
            )
            idxs = np.where(np.any(image_masks_processed != 0, axis=1))[0]
            lvr_token_idxs_list.append(idxs)
        return lvr_token_idxs_list

    def bbox_to_token_idxs_manual(
            self,  
            images: List[Image.Image], 
            bboxes: List[Tuple[float, float, float, float]]) -> List[np.ndarray]:
            """
            Convert bounding box coordinates to visual token indices.
            """
            token_idx_list = []
            for img, bbox in zip(images, bboxes):
                patch_size = self.processor.image_processor.patch_size
                image_width = img.width
                image_height = img.height

                grid_height = image_height // patch_size
                grid_width = image_width // patch_size

                token_grid_height = grid_height // self.processor.image_processor.temporal_patch_size
                token_grid_width = grid_width // self.processor.image_processor.temporal_patch_size

                x1, y1, x2, y2 = bbox
                if max(x1, y1, x2, y2) > 1.0:
                    x1 /= image_width
                    y1 /= image_height
                    x2 /= image_width
                    y2 /= image_height
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1, x2), min(1, y2)
                
                token_x1 = int(x1 * token_grid_width)
                token_y1 = int(y1 * token_grid_height)
                token_x2 = min(int(math.ceil(x2 * token_grid_width)), token_grid_width)
                token_y2 = min(int(math.ceil(y2 * token_grid_height)), token_grid_height)
                
                if token_x2 <= token_x1:
                    token_x2 = token_x1 + 1
                if token_y2 <= token_y1:
                    token_y2 = token_y1 + 1
                
                token_indices = []
                for y in range(token_y1, token_y2):
                    for x in range(token_x1, token_x2):
                        token_idx = y * token_grid_width + x
                        token_indices.append(token_idx)
                token_idx_list.append(np.array(token_indices))
            
            return token_idx_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        is_video = False
        processor = self.processor
        
        # 提取text_embedding（如果存在）
        text_embedding = None
        if "emb" in sources:
            text_embedding = torch.tensor(sources["emb"], dtype=torch.float32)
        
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, 
                                           self.image_resized_w, self.image_resized_h))
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        # 提取bounding boxes
        bboxes = sources['bboxes']
        lvr_token_idxs_list_manual = self.bbox_to_token_idxs_manual(images, bboxes)

        # 使用latent_end_token参数
        sources = copy.deepcopy(llava_to_openai_lvr(
            sources['conversations'], 
            is_video=is_video,
            lvr_token_idxs_list=lvr_token_idxs_list_manual,
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
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for j in range(0, len(sources), 2):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, 
                                 padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            elif DEFAULT_VIDEO_TOKEN in user_input:
                '''Not Implemented'''
                pass
            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, 
                                                     padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, 
                                                   padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            
            # FIX: Mask labels after </answer> tag to prevent endless token generation
            # The model should learn to STOP after </answer>, not continue generating padding tokens
            response_labels = response_input_ids.squeeze(0).clone()
            
            # Find the position of </answer> token in the response
            answer_end_token_ids = []
            try:
                # Get token IDs for </answer>
                answer_end_tokens = processor.tokenizer.encode("</answer>", add_special_tokens=False)
                if answer_end_tokens:
                    answer_end_token_ids = answer_end_tokens
            except:
                pass
            
            # Search for </answer> in response and mask everything after it
            if len(answer_end_token_ids) > 0:
                answer_end_id = answer_end_token_ids[-1]  # Use the last token of </answer>
                answer_end_pos = -1
                
                # Find where </answer> ends in response_labels
                for i in range(len(response_labels) - len(answer_end_token_ids) + 1):
                    if all(response_labels[i + j] == answer_end_id for j in range(len(answer_end_token_ids))):
                        # Found </answer>, mask everything after it
                        if i + len(answer_end_token_ids) < len(response_labels):
                            # Also mask the closing >
                            answer_end_pos = i + len(answer_end_token_ids)
                            response_labels[answer_end_pos:] = IGNORE_INDEX
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

        lvr_tokens = []
        for item_img in lvr_token_idxs_list_manual:
            group_lst = []
            for group in item_img:
                group_lst.append(torch.tensor(group))
            lvr_tokens.append(group_lst)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            lvr_tokens=lvr_tokens
        )

        # 添加text_embedding到data_dict
        if text_embedding is not None:
            data_dict['text_embedding'] = text_embedding

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

class DataCollatorForSupervisedDatasetLVR(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_lvr_tokens = []
        batch_text_embeddings = []  # 新增：收集text_embedding

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
            
            # 收集text_embedding
            if "text_embedding" in keys:
                batch_text_embeddings.append(example["text_embedding"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        lvr_tokens = [example['lvr_tokens'] for example in examples]
        lvr_tokens_all_local_indices = [torch.tensor(idx) for group in lvr_tokens for idx in group]

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'lvr_tokens': lvr_tokens_all_local_indices
        }

        # 如果有text_embedding，添加到data_dict
        if batch_text_embeddings:
            data_dict['text_embedding'] = torch.stack(batch_text_embeddings)

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    
def make_supervised_data_module_lvr(model_id, processor, data_args, latent_end_token=False):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDatasetLVR(
        data_path=data_args.data_path, 
        processor=processor, 
        data_args=data_args, 
        model_id=model_id,
        latent_end_token=latent_end_token
    )
    data_collator = DataCollatorForSupervisedDatasetLVR(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)