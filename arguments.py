from dataclasses import dataclass, field
from typing import Optional, List
import transformers

@dataclass
class DataTrainingArguments:

    dataset_path: str = "dataset_path"

    arbitrary_resolution: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If true, images will have arbitrary resolutions."
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    one_minus_one_data_transform: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If true, the data will be scaled to [-1, 1] instead of [0, 1]."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """
    
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: "},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    image_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The size (resolution) of each image. If not specified, will use `image_size` of the configuration."
            )
        },
    )
    
    fixed_image_size: Optional[bool] = field(
        default=True,
    )

    patch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The size (resolution) of each patch. If not specified, will use `patch_size` of the configuration."
            )
        },
    )

    tublet_size: Optional[List[int]] = field(
        default_factory=lambda: [2, 16, 16],
        metadata={
            "help": (
                "The size of each tubelet (3D patch size). If not specified, will use `tubelet_size` of the configuration."
            )
        },
    )

    cost_gradient_penalty: Optional[float] = field(
        default=0, # 0.2
    )

    enable_flash: Optional[bool] = field(
        default=False,
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    multiple_optimizer_training: Optional[float] = field( default=False, metadata={ "help": "will become true if `gan_loss_weight` in `model_args` is set to allow multiple optimizers" } )

    wandb_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "wandb api key"
        }
    )

    train_steps: Optional[int] = field(default=1,)

    visual_pattern: Optional[str] = field(default=None,)

    clip_image_size: Optional[int] = field(default=224,)
    
    metaclip_version: Optional[str] = field(default=None,)