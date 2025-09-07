import functools
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.distributed
from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.module import MegatronModule

from ..checkpointing import (
    ensure_directory_exists,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    load_state_dict_from_checkpoint,
)
from ..utils import get_logger
from .converter.convert_utils import MAX_SHARD_SIZE
from .converter.model_converter import ModelConverter
from .model_config import McaModelConfig
from .model_utils import ModuleUtilsMixin, RMSNorm, exists_hf_config, exists_mca_config


class ValueHeadWrapper(torch.nn.Module):
    """Wrapper to adapt value_head to output_layer interface.
    
    The parent GPTModel.forward() expects output_layer to be callable with:
    logits, bias = self.output_layer(hidden_states, weight=..., runtime_gather_output=...)
    
    This wrapper ignores the extra parameters and returns the expected tuple format.
    Includes dropout to match TRL's ValueHead implementation.
    """
    
    def __init__(self, value_head, dropout_prob=0.1):
        super().__init__()
        self.value_head = value_head
        # Match TRL's dropout behavior (default 0.1)
        self.dropout = torch.nn.Dropout(dropout_prob) if dropout_prob else torch.nn.Identity()
    
    def forward(self, hidden_states, weight=None, runtime_gather_output=None):
        """Forward pass matching output_layer interface.
        
        Args:
            hidden_states: Hidden states from transformer [batch, seq_len, hidden_size]
            weight: Optional weight for embedding sharing (ignored)
            runtime_gather_output: Optional flag for parallel output (ignored)
            
        Returns:
            Tuple of (values, None) where values has shape [batch, seq_len, 1]
        """
        # Apply dropout before the linear layer (matching TRL)
        hidden_states = self.dropout(hidden_states)
        values = self.value_head(hidden_states)
        return values, None  # Return (logits, bias) tuple format


if TYPE_CHECKING:
    from ..training_args import TrainingArguments


logger = get_logger(__name__)


class VirtualModels:
    # a wrapper for model list to support virtual pipeline model parallel
    def __init__(self, cls, config: "McaModelConfig", *args, **kwargs):
        self.models: List["McaGPTModel"] = []
        self.config = config
        for i in range(config.virtual_pipeline_model_parallel_size or 1):
            if (config.virtual_pipeline_model_parallel_size or 1) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(i)
            self.models.append(cls(config, *args, **kwargs))

    def save_pretrained(self, save_directory: str):
        if len(self.models) == 1:
            return self.models[0].save_pretrained(save_directory)
        state_dict = {f"model{i}": model.state_dict_for_save_checkpoint() for i, model in enumerate(self.models)}
        return self.models[0].save_pretrained(save_directory, state_dict=state_dict)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        if len(self.models) == 1:
            if "model" in state_dict:
                state_dict = state_dict["model"]
            return self.models[0].load_state_dict(state_dict, strict=strict)
        all_missing_keys, all_unexpected_keys = [], []
        for i, model in enumerate(self.models):
            ret = model.load_state_dict(state_dict[f"model{i}"], strict=strict)
            if not strict:
                all_missing_keys.extend(ret[0])
                all_unexpected_keys.extend(ret[1])
        return all_missing_keys, all_unexpected_keys

    def state_dict(self, *args, **kwargs):
        if len(self.models) == 1:
            return self.models[0].state_dict(*args, **kwargs)
        return {f"model{i}": model.state_dict(*args, **kwargs) for i, model in enumerate(self.models)}

    def get_models(self):
        return self.models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        return self.models[index]

    def __iter__(self):
        return iter(self.models)

    def parameters(self):
        for model in self.models:
            yield from model.parameters()

    def named_parameters(self, *args, **kwargs):
        for model in self.models:
            yield from model.named_parameters(*args, **kwargs)

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]):
        return self.models[0].estimate_tokens(input_dict)

    @functools.lru_cache(maxsize=4)
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False):
        return sum(model.num_parameters(only_trainable, exclude_embeddings) for model in self.models)

    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)

    def zero_grad(self):
        for model in self.models:
            model.zero_grad()

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self, *args, **kwargs):
        for model in self.models:
            model.train(*args, **kwargs)

    def to(self, *args, **kwargs):
        for model in self.models:
            model.to(*args, **kwargs)
        return self

    @property
    def main_input_name(self):
        return self.models[0].main_input_name

    def save_pretrained_as_hf(
        self, save_directory: str, save_safetensors: bool = True, max_shard_size: Union[int, str] = MAX_SHARD_SIZE
    ):
        os.makedirs(save_directory, exist_ok=True)
        converter = ModelConverter(self.config)
        converter.save_model_as_hf_inflight(
            self.models, save_directory, save_safetensors=save_safetensors, max_shard_size=max_shard_size
        )

    def all_gather_weights_as_hf_inflight(self, models=None):
        models = models or self.models
        converter = ModelConverter(self.config)
        yield from converter.all_gather_weights_as_hf_inflight(models)

    def all_gather_weights_as_hf_bucket(self, models=None, bucket_size: int = None):
        models = models or self.models
        converter = ModelConverter(self.config)
        yield from converter.all_gather_weights_as_hf_bucket(models, bucket_size=bucket_size)

    def get_batch_on_this_cp_rank(self, *args, **kwargs):
        return self.models[0].get_batch_on_this_cp_rank(*args, **kwargs)

    def sharded_state_dict(self, prefix: str = "", *args, **kwargs):
        state_dict = {}
        if len(self.models) == 1:
            state_dict['model'] = self.models[0].sharded_state_dict(prefix, *args, **kwargs)
        else:
            for i in range(len(self.models)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = self.models[i].sharded_state_dict(prefix, *args, **kwargs)
        return state_dict


class PretrainedModel(MegatronModule, ModuleUtilsMixin):
    config_class = McaModelConfig

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, args: "TrainingArguments" = None, use_cpu_initialization: bool = False
    ) -> "VirtualModels":
        load_start_time = time.time()
        config = cls.config_class.from_pretrained(model_name_or_path, args)
        config.use_cpu_initialization = use_cpu_initialization
        models = VirtualModels(cls, config=config)

        logger.info(
            f"number of parameters on (tensor, pipeline, expert) model parallel rank "
            f"({mpu.get_tensor_model_parallel_rank()}, {mpu.get_pipeline_model_parallel_rank()}, "
            f"{mpu.get_expert_model_parallel_rank()}): {sum(p.nelement() for p in models.parameters())}"
        )

        mca_ckpt_exist = exists_mca_config(model_name_or_path)
        dist_config_match = False
        if mca_ckpt_exist:
            old_mca_config = cls.config_class.from_pretrained(model_name_or_path)
            dist_config_match = config.distribute_config_match(old_mca_config)

        if mca_ckpt_exist and dist_config_match:
            state_dict = load_state_dict_from_checkpoint(model_name_or_path)
            models.load_state_dict(state_dict)
        else:
            if not exists_hf_config(model_name_or_path):
                raise ValueError(
                    f"{model_name_or_path} is not valid for current training, because not exists hf ckpt "
                    f"and not mca_ckpt_exist: {mca_ckpt_exist} or not dist_config_match: {dist_config_match}"
                )
            state_dict = {}
            converter = ModelConverter(config, model_name_or_path=model_name_or_path)
            for i in range(len(models)):
                key = "model"
                if len(models) > 1:
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    key = f"{key}{i}"
                state_dict[key] = converter.load_mca_state_dict_from_hf()
            missing_keys, unexpected_keys = models.load_state_dict(state_dict, strict=False)
            if missing_keys:
                missing_keys = [key for key in missing_keys if not key.endswith("._extra_state")]
            if unexpected_keys and config.tie_embeddings_and_output_weights:
                unexpected_keys = [key for key in unexpected_keys if not key.endswith("output_layer.weight")]
            assert unexpected_keys is None or len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
            assert missing_keys is None or len(missing_keys) == 0, f"missing_keys: {missing_keys}"
        logger.info(f"End loading, cost: {time.time() - load_start_time:0.3f}s")
        return models

    def save_pretrained(self, save_directory: str, state_dict=None):
        os.makedirs(save_directory, exist_ok=True)
        # TODO: better directory structure
        tracker_file = get_checkpoint_tracker_filename(save_directory)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.config.save_pretrained(save_directory)
            with open(tracker_file, "w") as f:
                f.write("1")
        if not torch.distributed.is_initialized() or mpu.get_expert_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(save_directory)
            ensure_directory_exists(checkpoint_name)
            if state_dict is None:
                state_dict = {"model": self.state_dict_for_save_checkpoint()}
            torch.save(state_dict, checkpoint_name)
            logger.info(f"Saving model checkpoint to {checkpoint_name}")

    def get_batch_on_this_cp_rank(self, batch: Dict[str, "torch.Tensor"], dim3_keys: List[str] = ["attention_mask"]):
        # copy from Megatron-LM
        """Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        """
        # With causal masking, each token only attends to its prior tokens. Simply split
        # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
        # at the end of sequence have bigger workload than others. To address this issue,
        # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
        # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
        # that we can get balanced workload among GPUs in a context parallel group.
        cp_size = self.config.context_parallel_size
        if cp_size > 1:
            cp_rank = mpu.get_context_parallel_rank()
            for key, val in batch.items():
                if val is not None and isinstance(val, torch.Tensor):
                    seq_dim = 2 if key in dim3_keys else 1
                    val = val.view(
                        *val.shape[0:seq_dim],
                        2 * cp_size,
                        val.shape[seq_dim] // (2 * cp_size),
                        *val.shape[(seq_dim + 1) :],
                    )
                    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                        non_blocking=True
                    )
                    val = val.index_select(seq_dim, index)
                    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                    batch[key] = val

        return batch


class McaGPTModel(GPTModel, PretrainedModel):
    main_input_name: str = "input_ids"
    config_class = McaModelConfig

    def __init__(self, config: "McaModelConfig", **kwargs):
        transformer_layer_spec = self._get_transformer_layer_spec(config)
        pre_process = kwargs.pop("pre_process", mpu.is_pipeline_first_stage())
        post_process = kwargs.pop("post_process", mpu.is_pipeline_last_stage())
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=config.tie_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            mtp_block_spec=kwargs.get("mtp_block_spec", None),
        )
        for param in self.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        if not config.use_cpu_initialization:
            self.cuda(torch.cuda.current_device())

    def _get_transformer_layer_spec(self, config: Optional["McaModelConfig"]=None):
        config = config or self.config
        use_te = config.transformer_impl == "transformer_engine"
        if config.num_moe_experts:
            transformer_block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            if not use_te and config.normalization == "RMSNorm":
                transformer_block_spec.layer_norm = RMSNorm
            for transformer_layer_spec in transformer_block_spec.layer_specs:
                if not use_te and config.normalization == "RMSNorm":
                    transformer_layer_spec.submodules.input_layernorm = RMSNorm
                    transformer_layer_spec.submodules.pre_mlp_layernorm = RMSNorm
                if hasattr(transformer_layer_spec.submodules.mlp.submodules, "shared_experts"):
                    transformer_layer_spec.submodules.mlp.submodules.shared_experts.params["gate"] = config.moe_use_shared_expert_gate
            return transformer_block_spec
        if use_te:
            return get_gpt_layer_with_transformer_engine_spec(config.num_moe_experts, config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm)
        else:
            module_spec = get_gpt_layer_local_spec(config.num_moe_experts, config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm)
            if config.normalization == "RMSNorm":
                module_spec.submodules.input_layernorm = RMSNorm
                module_spec.submodules.pre_mlp_layernorm = RMSNorm
            return module_spec


class McaValueModel(GPTModel, PretrainedModel):
    """
    Megatron value model for critic training.
    
    Replaces the language modeling head with a value head that outputs
    scalar values for each token position. Used in PPO and other RL algorithms
    for value function estimation.
    
    Args:
        config: McaModelConfig with model configuration
        **kwargs: Additional arguments including pre_process and post_process flags
    """
    main_input_name: str = "input_ids"
    config_class = McaModelConfig
    
    def __init__(self, config: "McaModelConfig", **kwargs):
        """
        Initialize value model with custom value head.
        
        Critical: Store pre/post process flags before super().__init__ 
        as parent class pops them from kwargs.
        """
        # CRITICAL: Store and pop flags BEFORE super().__init__
        self.pre_process = kwargs.pop("pre_process", mpu.is_pipeline_first_stage())
        self.post_process = kwargs.pop("post_process", mpu.is_pipeline_last_stage())
        
        transformer_layer_spec = self._get_transformer_layer_spec(config)
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_sequence_length,
            pre_process=self.pre_process,
            post_process=self.post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=config.tie_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            mtp_block_spec=kwargs.get("mtp_block_spec", None),
        )
        
        # Replace language modeling head with value head for last pipeline stage
        if self.post_process:
            # Add value head: hidden_size → 1
            # Match TRL/DeepSpeed: use bias=True (TRL default)
            value_head = torch.nn.Linear(
                config.hidden_size, 1, bias=True, dtype=config.params_dtype
            )
            
            # Initialize value head weights to match TRL/DeepSpeed behavior
            # IMPORTANT: TRL's default (v_head_init_strategy=None) does NOT initialize weights
            # This means PyTorch's default nn.Linear initialization is used:
            # - Weights: uniform(-sqrt(1/fan_in), sqrt(1/fan_in)) 
            # - Bias: uniform(-sqrt(1/fan_in), sqrt(1/fan_in)) - NOT zero!
            if config.perform_initialization:
                # Match PyTorch's default Kaiming uniform initialization
                import math
                bound = math.sqrt(1.0 / config.hidden_size)
                value_head.weight.data.uniform_(-bound, bound)
                if value_head.bias is not None:
                    # CRITICAL: PyTorch default does NOT zero the bias!
                    # It uses the same uniform distribution as weights
                    value_head.bias.data.uniform_(-bound, bound)
                logger.info(f"Initialized value_head to match TRL default (PyTorch nn.Linear defaults)")
                logger.info(f"Range: [{-bound:.6f}, {bound:.6f}] for both weights and bias")
                logger.info(f"Bias value: {value_head.bias.data.item():.6f} (NOT zero!)")
                logger.info(f"Sample weights: {value_head.weight.data.flatten()[:5].tolist()}")
            
            # Set tensor parallel attributes for value head
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                value_head.weight
            )
            if value_head.bias is not None:
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                    value_head.bias
                )
            
            # Create a module wrapper that matches output_layer interface
            # This allows parent's forward() to work unchanged
            # Use dropout_prob=0.1 to match TRL's default
            self.output_layer = ValueHeadWrapper(value_head, dropout_prob=0.1)
        
        for param in self.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        if not config.use_cpu_initialization:
            self.cuda(torch.cuda.current_device())
    
    
    def state_dict_for_save_checkpoint(self) -> Dict[str, torch.Tensor]:
        """
        Override to include value_head weights in checkpoint.
        
        Returns:
            Dict containing all model weights including value_head
        """
        # Get parent class state dict - this will include output_layer.value_head.weight
        state_dict = super().state_dict_for_save_checkpoint() if hasattr(super(), 'state_dict_for_save_checkpoint') else self.state_dict()
        
        # The value_head is now at output_layer.value_head.weight
        # No need to add it separately as it's already included
        return state_dict
    
    def save_pretrained(self, save_directory: str, state_dict: Optional[Dict] = None, **kwargs):
        """
        Save model in HuggingFace format for compatibility with DeepSpeed backend.
        
        Args:
            save_directory: Directory to save model
            state_dict: Optional state dict to save
            **kwargs: Additional save arguments
        """
        if state_dict is None:
            state_dict = self.state_dict_for_save_checkpoint()
        # Call parent's save_pretrained for HuggingFace compatibility
        return super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """
        Override to handle missing value_head weights when loading from GPT checkpoints.
        
        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce matching keys
            
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Filter out value_head from missing keys if loading from GPT checkpoint
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
        
        # Only raise error if there are missing keys other than value_head
        # The value_head is now at output_layer.value_head.weight and output_layer.value_head.bias
        filtered_missing = [k for k in missing_keys if not ("value_head" in k or "output_layer" in k)]
        
        if strict and filtered_missing:
            raise RuntimeError(f"Missing keys in state_dict: {filtered_missing}")
        
        return missing_keys, unexpected_keys
    
    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, args: "TrainingArguments" = None, use_cpu_initialization: bool = False
    ) -> "VirtualModels":
        """
        Override from_pretrained to handle missing value_head.weight gracefully.
        This allows loading from GPT checkpoints that don't have the value head.
        """
        load_start_time = time.time()
        config = cls.config_class.from_pretrained(model_name_or_path, args)
        config.use_cpu_initialization = use_cpu_initialization
        models = VirtualModels(cls, config=config)

        logger.info(
            f"number of parameters on (tensor, pipeline, expert) model parallel rank "
            f"({mpu.get_tensor_model_parallel_rank()}, {mpu.get_pipeline_model_parallel_rank()}, "
            f"{mpu.get_expert_model_parallel_rank()}): {sum(p.nelement() for p in models.parameters())}"
        )

        mca_ckpt_exist = exists_mca_config(model_name_or_path)
        dist_config_match = False
        if mca_ckpt_exist:
            old_mca_config = cls.config_class.from_pretrained(model_name_or_path)
            dist_config_match = config.distribute_config_match(old_mca_config)

        if mca_ckpt_exist and dist_config_match:
            state_dict = load_state_dict_from_checkpoint(model_name_or_path)
            models.load_state_dict(state_dict)
        else:
            if not exists_hf_config(model_name_or_path):
                raise ValueError(
                    f"{model_name_or_path} is not valid for current training, because not exists hf ckpt "
                    f"and not mca_ckpt_exist: {mca_ckpt_exist} or not dist_config_match: {dist_config_match}"
                )
            state_dict = {}
            converter = ModelConverter(config, model_name_or_path=model_name_or_path)
            for i in range(len(models)):
                key = "model"
                if len(models) > 1:
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    key = f"{key}{i}"
                state_dict[key] = converter.load_mca_state_dict_from_hf()
            missing_keys, unexpected_keys = models.load_state_dict(state_dict, strict=False)
            if missing_keys:
                missing_keys = [key for key in missing_keys if not key.endswith("._extra_state")]
            if unexpected_keys and config.tie_embeddings_and_output_weights:
                unexpected_keys = [key for key in unexpected_keys if not key.endswith("output_layer.weight")]
            
            # For McaValueModel, filter out expected missing value_head/output_layer weights
            if missing_keys:
                value_head_missing = [key for key in missing_keys if "value_head" in key or "output_layer" in key]
                other_missing = [key for key in missing_keys if not ("value_head" in key or "output_layer" in key)]
                if value_head_missing:
                    logger.info(f"Expected missing keys for value model (will be randomly initialized): {value_head_missing}")
                missing_keys = other_missing  # Only check non-value_head missing keys
            
            assert unexpected_keys is None or len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
            assert missing_keys is None or len(missing_keys) == 0, f"missing_keys: {missing_keys}"
        logger.info(f"End loading, cost: {time.time() - load_start_time:0.3f}s")
        return models
    
    def _get_transformer_layer_spec(self, config: Optional["McaModelConfig"]=None):
        """Reuse the same transformer layer spec as McaGPTModel."""
        config = config or self.config
        use_te = config.transformer_impl == "transformer_engine"
        if config.num_moe_experts:
            transformer_block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            if not use_te and config.normalization == "RMSNorm":
                transformer_block_spec.layer_norm = RMSNorm
            for transformer_layer_spec in transformer_block_spec.layer_specs:
                if not use_te and config.normalization == "RMSNorm":
                    transformer_layer_spec.submodules.input_layernorm = RMSNorm
                    transformer_layer_spec.submodules.pre_mlp_layernorm = RMSNorm
                if hasattr(transformer_layer_spec.submodules.mlp.submodules, "shared_experts"):
                    transformer_layer_spec.submodules.mlp.submodules.shared_experts.params["gate"] = config.moe_use_shared_expert_gate
            return transformer_block_spec
        if use_te:
            return get_gpt_layer_with_transformer_engine_spec(config.num_moe_experts, config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm)
        else:
            module_spec = get_gpt_layer_local_spec(config.num_moe_experts, config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm)
            if config.normalization == "RMSNorm":
                module_spec.submodules.input_layernorm = RMSNorm
                module_spec.submodules.pre_mlp_layernorm = RMSNorm
            return module_spec
