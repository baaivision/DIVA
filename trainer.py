import pickle
import time
import wandb
from PIL import Image
import numpy as np
import gc

import torch
from torch.utils.data import Sampler, DataLoader

from transformers import Trainer
# from hf_trainer import Trainer

from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.utils import is_datasets_available
from transformers.integrations import WandbCallback

# from vq_model import unpack_images
if is_datasets_available():
    import datasets
from transformers.trainer_utils import seed_worker
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    AcceleratorConfig,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

ASPECT_RATIOS = {}

class CustomTrainer(Trainer):

    def monitor_aspect_raio(self, aspect_ratio):
        if aspect_ratio not in ASPECT_RATIOS:
            ASPECT_RATIOS[aspect_ratio] = 0
        ASPECT_RATIOS[aspect_ratio] += 1
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Qiying overrides for logging training-related metrics
        
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
        
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if hasattr(eval_dataset, "root") and eval_dataset.root == "/share/project/datasets/ImageNet/val":
            data_collator = self.eval_collator['in1k']
            self.args.eval_set_name = "in1k"
            self.model.eval_set_name = "in1k"
        else:
            data_collator = self.eval_collator['highres']
            self.args.eval_set_name = "highres"
            self.model.eval_set_name = "highres"

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
    
    def create_optimizer(self):
        """
        Qiying overrides to allow multiple optimizers for GAN training
        
        Setup the optimizer.
        """
        if not self.args.multiple_optimizer_training:
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("discriminator" not in n and 'perceptual' not in n) and (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("discriminator" not in n and 'perceptual' not in n) and (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = [
                optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs),
                torch.optim.Adam(opt_model.discriminator.parameters(), **optimizer_kwargs)  # TODO
            ]
            
            # print(f"{optimizer_grouped_parameters=}, {opt_model.discriminator=}")
            
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Qiying overrides to allow multiple schedulers for GAN training
        
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if not self.args.multiple_optimizer_training:
            return super().create_scheduler(num_training_steps, optimizer)
        
        if self.lr_scheduler is None:
            self.lr_scheduler = []
            for optim in self.optimizer:
                self.lr_scheduler.append(
                    get_scheduler(
                        self.args.lr_scheduler_type,
                        optimizer=optim,
                        num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                        num_training_steps=num_training_steps,
                        scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
                    )
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _get_learning_rate(self):
        if not self.args.multiple_optimizer_training:
            return super()._get_learning_rate()

        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[0]["lr"]
            else:
                last_lr = self.lr_scheduler[0].get_last_lr()[0]
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if not self.args.multiple_optimizer_training:
            return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
        
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        
    def _save_optimizer_and_scheduler(self, output_dir):
        if not self.args.multiple_optimizer_training:
            return super()._save_optimizer_and_scheduler(output_dir)
        
        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
            )
            if accept_exclude_frozen_parameters and _is_peft_model(self.model):
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                self.model_wrapped.save_checkpoint(output_dir)
        elif self.is_fsdp_enabled:
            # save fsdp specific ckpt for resuming from ckpt
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir, **_get_fsdp_ckpt_kwargs()
            )
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        elif self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            for i, optimizer in enumerate(self.optimizer):
                torch.save(optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME + f"_{i}"))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_tpu_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                for i, scheduler in enumerate(self.lr_scheduler):
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME + f"_{i}"))
            reissue_pt_warnings(caught_warnings)
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        if not self.args.multiple_optimizer_training:
            return super()._load_optimizer_and_scheduler(checkpoint)
        
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
                or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            if is_torch_tpu_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                optimizer_state = torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location="cpu")
                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu")
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(checkpoint, "user_content.pt")):
                        # Optimizer checkpoint was saved with smp >= 1.10
                        def opt_load_hook(mod, opt):
                            opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    else:
                        # Optimizer checkpoint was saved with smp < 1.10
                        def opt_load_hook(mod, opt):
                            if IS_SAGEMAKER_MP_POST_1_10:
                                opt.load_state_dict(
                                    smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True, back_compat=True)
                                )
                            else:
                                opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    if self.is_fsdp_enabled:
                        load_fsdp_optimizer(
                            self.accelerator.state.fsdp_plugin,
                            self.accelerator,
                            self.optimizer,
                            self.model,
                            checkpoint,
                            **_get_fsdp_ckpt_kwargs(),
                        )
                    else:
                        self.optimizer[0].load_state_dict(
                            torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location)
                        )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler[0].load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if not self.args.multiple_optimizer_training:
            return super()._inner_training_loop(batch_size=batch_size, args=args, resume_from_checkpoint=resume_from_checkpoint, trial=trial, ignore_keys_for_eval=ignore_keys_for_eval)
        
        """
        Overrides to allow mutiple optimizer and scheduler step
        """
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False
        
        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        
        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if ((isinstance(self.lr_scheduler, list)) and hasattr(self.lr_scheduler[0], "step")) or hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                if isinstance(self.optimizer, list):
                    model, self.optimizer[0], self.optimizer[1], self.lr_scheduler[0], self.lr_scheduler[1] = self.accelerator.prepare(
                        self.model, self.optimizer[0], self.optimizer[1], self.lr_scheduler[0], self.lr_scheduler[1]
                    )
                else:
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                        self.model, self.optimizer, self.lr_scheduler
                    )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                print(">> before skip_first_batches..")
                print(">> Remove skip_first_batches(..)")
                # epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch) # this will cause the error (on-hold) when resuming training
                print(">> after skip_first_batches..")
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    
                def toggle_optimizer(model, optimizer):
                    """
                    Makes sure only the gradients of the current optimizer's parameters are calculated
                    in the training step to prevent dangling gradients in multiple-optimizer setup.

                    .. note:: Only called when using multiple optimizers

                    Override for your own behavior

                    Args:
                        optimizer:
                        optimizer_idx:
                    """
                    for param in model.parameters():
                        param.requires_grad = False

                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True
                
                # new stuff
                for i, (optimizer, lr_scheduler) in enumerate(zip(self.optimizer, self.lr_scheduler)):
                    
                    if self.state.global_step < self.model.config.disc_start:
                        if i == 1:
                            continue
                        else:
                            i = None
                    
                    toggle_optimizer(model, optimizer)
                    
                    # print(f"{optimizer.param_groups=}, {optimizer.param_groups[0]['params'][0].requires_grad=}")
                    
                    inputs['optimizer_idx'] = i

                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                            else:
                                grad_norm = _grad_norm.item() if _grad_norm is not None else None

                        # Optimizer step
                        optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                lr_scheduler.step()

                        model.zero_grad()
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_tpu_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # QY: add to avoid gather input images
        label_names = self.label_names
        self.label_names = []
        
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        self.model.save_min_encoding_indices = None
        self.model.save_features = None
        start_time = time.time()
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.gather_function((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        end_time = time.time()
        print(f"eval_time={end_time-start_time}!!")
        # if self.model.save_min_encoding_indices is not None:
        #     save_name = Path(self.model.config.codesplit_save_dir) / f"saveimages_rank{dist.get_rank()}_{step}.pkl"
        #     with open(save_name, "wb") as f:
        #         pickle.dump(
        #             {
        #                 # "recon_pixel_values": recon_pixel_values.cpu(),
        #                 "min_encoding_indices": self.model.save_min_encoding_indices,
        #                 "features": self.model.save_features
        #             }, f
        #         )
        #     del self.model.save_features, self.model.save_min_encoding_indices
        #     self.model.save_min_encoding_indices = None
        #     self.model.save_features = None
        #     gc.collect()
        
        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        # QY: modify
        if self.compute_metrics is not None and all_preds is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs), self.args.logging_dir
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), self.args.logging_dir)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        # QY: add to avoid gather input images
        self.label_names = label_names

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

            

OPENAI_DATASET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
OPENAI_DATASET_STD = np.array([0.26862954, 0.26130258, 0.27577711])

def organize_wandb_image(ori_images, recon_images, n_images=8):
    
    ori, recon = ori_images[:n_images // 2].detach().cpu().float().permute(0, 2, 3, 1), recon_images[:n_images // 2].detach().cpu().float().permute(0, 2, 3, 1)  # [8, 224, 224, 3]
    ori = torch.clip((ori * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
    recon = torch.clip((recon * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
    example1 = torch.cat([ori, recon], dim=2).contiguous()  # 8, 224, 448, 3
    
    ori, recon = ori_images[n_images // 2: n_images].detach().cpu().float().permute(0, 2, 3, 1), recon_images[n_images // 2: n_images].detach().cpu().float().permute(0, 2, 3, 1)  # [8, 224, 224, 3]
    ori = torch.clip((ori * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
    recon = torch.clip((recon * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
    example2 = torch.cat([ori, recon], dim=2).contiguous()  # 8, 224, 448, 3
    
    blank = torch.zeros(example1.shape[0], example1.shape[1], 100, 3)
    
    example = torch.cat([example1, blank, example2], dim=2)
    
    _, _, w, c = example.shape
    example = example.view(-1, w, c).numpy().astype("uint8")
    example = Image.fromarray(example)
    return example


def organize_wandb_video(ori_images, recon_images, one_minus_one_data_transform=False):

    # b, t, c, h, w

    b, t, c, h, w = ori_images.shape

    if b >= 8:
        num_select = 4
    elif b >= 2:
        num_select = 2
    else:
        num_select = 1

    ori, recon = ori_images[:num_select].detach().cpu().float().permute(0, 1, 3, 4, 2), recon_images[:num_select].detach().cpu().float().permute(0, 1, 3, 4, 2)  # [8, t, 224, 224, 3]
    if one_minus_one_data_transform:
        ori = torch.clip((ori + 1) * 127.5, 0, 255)
        recon = torch.clip((recon + 1) * 127.5, 0, 255)
    else:
        ori = torch.clip((ori * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
        recon = torch.clip((recon * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
    example1 = torch.cat([ori, recon], dim=3).contiguous()  # 8, t, 224, 448, 3
    
    ori, recon = ori_images[num_select:num_select*2].detach().cpu().float().permute(0, 1, 3, 4, 2), recon_images[num_select:num_select*2].detach().cpu().float().permute(0, 1, 3, 4, 2)  # [8, 224, 224, 3]
    if one_minus_one_data_transform:
        ori = torch.clip((ori + 1) * 127.5, 0, 255)
        recon = torch.clip((recon + 1) * 127.5, 0, 255)
    else:
        ori = torch.clip((ori * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
        recon = torch.clip((recon * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN) * 255, 0, 255)
    example2 = torch.cat([ori, recon], dim=3).contiguous()  # 8, t, 224, 448, 3
    
    blank = torch.zeros(example1.shape[0], recon.shape[1], example1.shape[2], 100, 3)
    
    try:
        example = torch.cat([example1, blank, example2], dim=3)
    except:
        example = example1
    
    # size=(10, 3, 100, 100)
    # view(-1, w, c)
    b, t, h, w, c = example.shape
    # t, c, b, h, w -> t, c, b*h, w
    example = example.permute(1, 4, 0, 2, 3).reshape(t, c, b*h, w).numpy().astype("uint8")
    # example = Image.fromarray(example)
    # wandb.Video(frames, fps=4)
    return example