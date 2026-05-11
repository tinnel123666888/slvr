# from .dpo_trainer import QwenDPOTrainer
from .sft_trainer import QwenSFTTrainer
from .grpo_trainer import QwenGRPOTrainer
from .slvr_trainer import QwenSLVRSFTTrainer
from .qq_grpo_trainer import QwenGRPO2QTrainer
from .mgrpo_trainer import QwenMGRPOTrainer
# __all__ = ["QwenSFTTrainer", "QwenDPOTrainer", "QwenGRPOTrainer"]
# __all__ = ["QwenSFTTrainer", "QwenSLVRSFTTrainer"]
__all__ = ["QwenSFTTrainer", "QwenSLVRSFTTrainer","QwenGRPOTrainer", "QwenGRPO2QTrainer", "QwenMGRPOTrainer"]