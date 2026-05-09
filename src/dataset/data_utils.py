import re
import torch

from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,

    VISION_START_TOKEN,
    VISION_END_TOKEN,
    LVR_TOKEN,
    LVR_LATENT_END_TOKEN,
    LVR_PLACEHOLDER,SEM_START_TOKEN,SEM_TOKEN,SEM_END_TOKEN,SEM_PLACEHOLDER

)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def replace_lvr_tokens(input_string, lvr_token_idxs_list, latent_end_token, fixed_num_of_lvr_tokens):
    '''Video not implemented'''
    # 处理普通LVR tokens
    if LVR_PLACEHOLDER in input_string:
        parts = input_string.split(LVR_PLACEHOLDER)
        result_parts = [parts[0]]  # 保留第一部分（占位符前的内容）
        
        for i, seg in enumerate(parts[1:], 1):
            if fixed_num_of_lvr_tokens is not None:
                num_tokens = fixed_num_of_lvr_tokens
            else:
                # 确保有足够的lvr_token_idxs
                if i-1 < len(lvr_token_idxs_list):
                    num_tokens = len(lvr_token_idxs_list[i-1])
                else:
                    num_tokens = 0
                
            if latent_end_token:
                replacement = VISION_START_TOKEN + LVR_TOKEN*num_tokens + LVR_LATENT_END_TOKEN + VISION_END_TOKEN
            else:
                replacement = VISION_START_TOKEN + LVR_TOKEN*num_tokens + VISION_END_TOKEN
            result_parts.append(replacement + seg)
        
        input_string = "".join(result_parts)
    
    # 处理LVR text tokens
    if SEM_PLACEHOLDER in input_string:
        parts = input_string.split(SEM_PLACEHOLDER)
        result_parts = [parts[0]]  # 保留第一部分
        
        for seg in parts[1:]:
            replacement = SEM_START_TOKEN + SEM_TOKEN + SEM_END_TOKEN
            result_parts.append(replacement + seg)
        
        input_string = "".join(result_parts)
    
    return input_string



def llava_to_openai_lvr(conversations, is_video=False, lvr_token_idxs_list=None, latent_end_token=False, fixed_num_of_lvr_tokens=None):

    # assert lvr_token_idxs_list is not None

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_content = replace_lvr_tokens(transformed_content,lvr_token_idxs_list,latent_end_token,fixed_num_of_lvr_tokens)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        # print(transformed_entry)
        transformed_data.append(transformed_entry)

    return transformed_data

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future


    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, width, height, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs