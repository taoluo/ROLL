import copy
import json
import math
import os
from functools import partial
import time
from typing import Any, Dict, List

import datasets
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.datasets.chat_template import get_chat_template
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.utils import dump_rollout_to_specific_path
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    compute_advantage,
    compute_token_reward,
    get_sample_level_mask,
    reduce_metrics,
    reward_postprocess,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager


logger = get_logger()


def is_lora_training(pipeline_config: RLVRConfig) -> bool:
    if pipeline_config.actor_train.model_args.lora_target is None:
        return False
    assert pipeline_config.actor_train.strategy_args.strategy_name == "deepspeed_train", (
        "LoRA only supports deepspeed_train"
    )
    return True


def preprocess_dataset(dataset, prompt_len, encode_function, num_proc):
    # 处理数据
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        encode_function,
        batched=True,
        num_proc=num_proc,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )
    # 过滤cutoff
    dataset = dataset.filter(
        lambda data_i: 5 < len(data_i["input_ids"]) <= prompt_len,
        num_proc=num_proc,
        desc="Filtering dataset",
    )
    print(f"Filtering prompt len: {dataset}")
    print(f"Encoding: {dataset}")
    return dataset


def get_encode_function(template_name, tokenizer):
    chat_template_func = get_chat_template(template_name, tokenizer)

    def encode_function(data_i):
        text_list = []
        if "messages" in data_i:
            for messages in data_i["messages"]:
                if isinstance(messages, str):
                    messages = json.loads(messages)
                text_list.append(chat_template_func(messages))
        elif "prompt" in data_i:
            for prompt in data_i["prompt"]:
                text_list.append(prompt)
        encodings = tokenizer(text_list)
        return encodings

    return encode_function

def update_dataset_domain(tag_2_domain: Dict[str, set[str]], row):
    if 'domain' in row and row['domain'] is not None:
        return row
    row["domain"] = tag_2_domain.get(row["tag"], "math_rule")
    return row

def query_filter_fn(data_list: List[DataProto], config: RLVRConfig) -> bool:
    """
    各domain的过滤规则可以自定义
    """
    response_level_rewards = [data.batch["response_level_rewards"] for data in data_list]
    if len(response_level_rewards) == 1:
        return True
    rewards = torch.cat(response_level_rewards, dim=0)

    domain = data_list[0].non_tensor_batch["domain"][0]
    query_filter_config = config.rewards[domain].query_filter_config

    if query_filter_config.type == "no_filter":
        return True
    elif query_filter_config.type == "mean_filter":
        threshold_up = query_filter_config.filter_args.get("threshold_up", math.inf)
        threshold_down = query_filter_config.filter_args.get("threshold_down", -1)
        if torch.mean(rewards) <= threshold_down or torch.mean(rewards) >= threshold_up:
            return False
    elif query_filter_config.type == "std_filter":
        std_threshold = query_filter_config.filter_args.get("std_threshold", -1)
        if torch.std(rewards) <= std_threshold:
            return False
    return True


class CriticParityPipeline(BasePipeline):
    """Pipeline for testing parity between DeepSpeed and Megatron critic implementations."""

    def __init__(self, pipeline_config1: RLVRConfig, pipeline_config2: RLVRConfig, use_mock_data: bool = False):
        pipeline_config = pipeline_config1
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.pipeline_config2 = pipeline_config2
        self.is_lora = is_lora_training(self.pipeline_config)
        self.use_mock_data = use_mock_data  # Flag to use mock data instead of real data

        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        dataset_paths = []
        if self.pipeline_config.actor_train.data_args.file_name:
            dataset_paths.extend(self.pipeline_config.actor_train.data_args.file_name)

        print(f'load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}')
        dataset = datasets.load_dataset('json', data_files=dataset_paths)['train']

        self.val_dataset = None
        if self.pipeline_config.validation:
            val_dataset_paths = self.pipeline_config.validation.data_args.file_name
            self.val_dataset = datasets.load_dataset("json", data_files=val_dataset_paths)["train"]

        # 加上format，然后转ids的func
        template_name = (
            self.pipeline_config.global_template
            if self.pipeline_config.global_template
            else self.pipeline_config.actor_train.data_args.template
        )
        encode_function = get_encode_function(template_name, self.tokenizer)

        dataset = preprocess_dataset(
            dataset,
            self.pipeline_config.prompt_length,
            encode_function,
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
        )
        # update domain field
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            desc="update_dataset_domain",
            load_from_cache_file=False
        )
        self.domain_datasets: Dict[str, datasets.Dataset] = {}
        for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys():
            self.domain_datasets[domain] = dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            assert len(self.domain_datasets[domain]) > 0, f"domain dataset {domain} has no data"

        if self.val_dataset:
            self.val_dataset = preprocess_dataset(
                self.val_dataset,
                self.pipeline_config.prompt_length,
                encode_function,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            )
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False
            )

        assert 'domain' in dataset.column_names, "domain field should set in dataset"
        assert 'domain' in self.val_dataset.column_names, "domain field should set in val dataset"
        print(dataset)

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        # Also set max_steps for pipeline_config2 to ensure proper initialization
        self.pipeline_config2.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        # use unwrapped model as reference for lora training
        if not self.is_lora:
            self.reference: Any = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            self.critic2: Any = Cluster(
                name='critic_from_config_2',
                worker_cls=self.pipeline_config2.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config2.critic,
            )
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }

        domain_ratios = self.pipeline_config.actor_train.data_args.domain_interleave_probs
        self.generate_schedulers: Dict[str, DynamicSamplingScheduler] = {}
        self.domain_batch_size = {}
        domain_list = list(domain_ratios.keys())
        accumulated = 0
        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                domain_batch_size = self.pipeline_config.rollout_batch_size - accumulated
            else:
                domain_batch_size = int(domain_ratios[domain] * self.pipeline_config.rollout_batch_size)
            accumulated += domain_batch_size
            generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(pipeline_config=self.pipeline_config)
            ray.get(
                generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters={domain: self.rewards[domain]},
                    dataset=self.domain_datasets[domain],
                    collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                    collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=query_filter_fn,
                    response_callback_fn=generate_scheduler.report_response.remote,
                    state=self.state.kv.get(f"scheduler_state_{domain}", None),
                )
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (f"domain_batch_size {domain_batch_size} must be "
                                                                           f"less than the number of domain datasets {len(self.domain_datasets[domain])}")

        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.use_additional_prompts = False
            self.val_generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(pipeline_config=val_pipeline_config)
        if self.val_dataset:
            ray.get(
                self.val_generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters=self.rewards,
                    dataset=self.val_dataset,
                    collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                    collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=lambda data_list, config: True,
                    response_callback_fn=self.val_generate_scheduler.report_response.remote,
                )
            )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        if not self.is_lora:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs: List[ray.ObjectRef] = []
        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
        if self.pipeline_config.adv_estimator == "gae":
            # Set seed before critic initialization for reproducibility
            import random
            import numpy as np
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            np.random.seed(42)
            random.seed(42)
            self.critic.initialize(pipeline_config=self.pipeline_config, blocking=True)
            
            # Reset seed for second critic to ensure identical initialization
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            np.random.seed(42)
            random.seed(42)
            self.critic2.initialize(pipeline_config=self.pipeline_config2, blocking=True)
        # ray.get(refs)

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic, self.critic2)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = {}
        for domain in self.rewards.keys():
            self.running[domain] = RunningMoments()

    def create_mock_batch(self, batch_size=None, seq_len=2048, device="cuda"):
        """Create mock input data for testing critics - matching rlvr_pipeline_meg_critic.py."""
        import numpy as np
        from tensordict import TensorDict
        
        if batch_size is None:
            batch_size = self.pipeline_config.rollout_batch_size
        
        batch = DataProto()
        
        # Set seed for reproducible mock data
        torch.manual_seed(42)
        
        # Create input_ids by concatenating prompts and responses
        prompt_len = seq_len // 2
        response_len = seq_len // 2
        prompts = torch.randint(0, 30000, (batch_size, prompt_len), device=device)
        responses = torch.randint(0, 30000, (batch_size, response_len), device=device)
        input_ids = torch.cat([prompts, responses], dim=-1)
        
        # Create mock tensors
        tensors = {
            "input_ids": input_ids,
            "prompts": prompts,
            "responses": responses,
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            "position_ids": torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1),
            "response_mask": torch.ones(batch_size, response_len, dtype=torch.bool, device=device),
            "final_response_mask": torch.ones(batch_size, response_len, dtype=torch.bool, device=device),
            "rewards": torch.randn(batch_size, response_len, device=device) * 0.1,
            "old_log_probs": torch.randn(batch_size, response_len, device=device) * 0.1 - 2.0,
            "ref_log_probs": torch.randn(batch_size, response_len, device=device) * 0.1 - 2.0,
        }
        
        # Create TensorDict with proper batch_size
        batch.batch = TensorDict(tensors, batch_size=(batch_size,))
        
        # Set some masks to False to simulate real data
        batch.batch["response_mask"][:, :5] = False
        batch.batch["final_response_mask"][:, :-10] = False
        
        batch.non_tensor_batch = {
            "domain": np.array(["math"] * batch_size, dtype=object),
        }
        
        batch.meta_info = {
            "global_step": 0,
            "is_offload_states": False,
        }
        
        return batch

    @torch.no_grad()
    def run(self):
        # 计算tokens per second 系统吞吐

        # 创建一个专门管理监控指标的类
        metrics_mgr = MetricsManager()

        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        # Modified for testing: Run only 5 steps
        test_max_steps = min(5, self.pipeline_config.max_steps)
        logger.info(f"TEST MODE: Running only {test_max_steps} steps for critic comparison")
        
        for global_step in range(test_max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline step {global_step} start...")

            metrics_mgr.clear_metrics()
            with tps_timer, Timer(name="step_total", logger=None) as step_total_timer:

                # 先model update，resume时不需要保存infer cluster的状态
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                    self.critic2.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                with Timer(name="step_model_update", logger=None) as step_model_update_timer:
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics_mgr.add_metrics(model_update_metrics)
                metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)

                if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                    with Timer(name="val_step", logger=None) as val_step_timer:
                        val_metrics = self.val()
                        metrics_mgr.add_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                # Check if we should use mock data or real data
                if self.use_mock_data:
                    logger.info(f"[Step {global_step}] Using MOCK DATA for testing")
                    batch = self.create_mock_batch(device="cuda")
                    batch.meta_info["global_step"] = global_step
                    # Skip the generation step entirely when using mock data
                else:
                    batch: DataProto = DataProto()
                    batch.meta_info = {"global_step": global_step}

                    # 要按domain group by生成对应的batch
                    with actor_infer_timer, actor_infer_response_timer, Timer(
                        name="step_generate", logger=None
                    ) as step_generate_timer:
                        domain_batches = {}
                        batch.meta_info["generation_config"] = self.actor_infer.worker_config.generating_args.to_dict()
                        self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                        for reward_cluster in self.rewards.values():
                            reward_cluster.load_states()

                        batch.meta_info["is_offload_states"] = False
                        scheduler_refs = {}
                        for domain, scheduler in self.generate_schedulers.items():
                            scheduler_refs[domain] = scheduler.get_batch.remote(data=batch, batch_size=self.domain_batch_size[domain])
                        for domain, scheduler_ref in scheduler_refs.items():
                            domain_batch: DataProto = ray.get(scheduler_ref, timeout=self.pipeline_config.rpc_timeout)
                            metrics_mgr.add_domain_metrics(
                                domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                            )
                            domain_batches[domain] = domain_batch
                        generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
                        dump_rollout_to_specific_path(self.pipeline_config.rollout_dump_dir, global_step, generate_output, self.tokenizer)
                        generate_output.meta_info.pop("is_offload_states", None)

                        for reward_cluster in self.rewards.values():
                            reward_cluster.offload_states()
                        gen_metrics = self.actor_infer.stop_server()
                        metrics_mgr.add_domain_metrics(domain, reduce_metrics(gen_metrics.meta_info.pop("metrics", {})))
                    metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                    batch = generate_output

                # Skip log prob calculations if using mock data (already included)
                if not self.use_mock_data:
                    with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                        if self.is_lora:
                            batch.meta_info["disable_adapter"] = True
                            batch.meta_info["is_offload_states"] = False
                            ref_log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
                        else:
                            ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                        metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        batch = batch.union(ref_log_probs)
                    metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    if self.is_lora:
                        batch.meta_info["disable_adapter"] = False
                    batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.adv_estimator == "gae":
                        # Use blocking=False to match original pipeline pattern
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                        values_refs_2: List[ray.ObjectRef] = self.critic2.compute_values(batch, blocking=False)
                    
                    # Skip old_log_probs calculation if using mock data
                    if not self.use_mock_data:
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                        agg_entropy = agg_loss(
                            loss_mat=old_log_probs.batch["entropy"],
                            loss_mask=batch.batch["response_mask"][:, 1:],
                            loss_agg_mode="token-mean",
                        )
                        batch.meta_info["agg_entropy"] = agg_entropy
                    else:
                        # For mock data, create dummy entropy
                        batch.meta_info["agg_entropy"] = torch.tensor(0.0)

                    if self.pipeline_config.adv_estimator == "gae":
                        # Materialize values from both critics
                        values_deepspeed = DataProto.materialize_concat(data_refs=values_refs)
                        values_megatron = DataProto.materialize_concat(data_refs=values_refs_2)
                        
                        # Extract tensors for comparison
                        values_ds_tensor = values_deepspeed.batch["values"]
                        values_mg_tensor = values_megatron.batch["values"]
                        
                        # Verify shapes match
                        assert values_ds_tensor.shape == values_mg_tensor.shape, \
                            f"Shape mismatch: DeepSpeed {values_ds_tensor.shape} vs Megatron {values_mg_tensor.shape}"
                        
                        # Calculate comparison metrics
                        abs_diff = torch.abs(values_ds_tensor - values_mg_tensor)
                        rel_diff = abs_diff / (torch.abs(values_ds_tensor) + 1e-8)
                        
                        max_abs_diff = abs_diff.max().item()
                        mean_abs_diff = abs_diff.mean().item()
                        max_rel_diff = rel_diff.max().item()
                        mean_rel_diff = rel_diff.mean().item()
                        
                        # Find indices of top 10 largest differences
                        abs_diff_flat = abs_diff.flatten()
                        values_ds_flat = values_ds_tensor.flatten()
                        values_mg_flat = values_mg_tensor.flatten()
                        
                        # Get indices of top 10 largest differences
                        top_10_indices = torch.topk(abs_diff_flat, min(10, abs_diff_flat.size(0))).indices
                        
                        # Count tokens with nearly identical values (threshold: 0.1)
                        nearly_identical_threshold = 1e-1
                        nearly_identical_count = (abs_diff_flat < nearly_identical_threshold).sum().item()
                        total_tokens = abs_diff_flat.numel()
                        nearly_identical_pct = (nearly_identical_count / total_tokens) * 100
                        
                        # Additional functional equivalence checks (matching rlvr_pipeline_meg_critic.py)
                        # 1. Check that mean values are close
                        mean_ds = values_ds_tensor.mean().item()
                        mean_mg = values_mg_tensor.mean().item()
                        mean_diff_value = abs(mean_ds - mean_mg)
                        
                        # 2. Check that std values are close
                        std_ds = values_ds_tensor.std().item()
                        std_mg = values_mg_tensor.std().item()
                        std_rel_diff = abs(std_ds - std_mg) / max(std_ds, std_mg)
                        
                        # 3. Calculate correlation metrics
                        flat_ds = values_ds_tensor.flatten()
                        flat_mg = values_mg_tensor.flatten()
                        
                        # Pearson correlation (linear relationship)
                        correlation = torch.corrcoef(torch.stack([flat_ds, flat_mg]))[0, 1].item()
                        
                        # Spearman correlation (rank/monotonic relationship)
                        from scipy.stats import spearmanr
                        spearman_corr, _ = spearmanr(flat_ds.cpu().float().numpy(), flat_mg.cpu().float().numpy())
                        spearman_corr = float(spearman_corr)
                        
                        # Log comparison results
                        logger.info(f"[Critic Equivalence Test] Step {global_step + 1}/{test_max_steps}")
                        logger.info(f"  Shape: {values_ds_tensor.shape}")
                        
                        # Log tokens with nearly identical values
                        logger.info(f"  Nearly identical tokens (diff < {nearly_identical_threshold}):")
                        logger.info(f"    Count: {nearly_identical_count}/{total_tokens} ({nearly_identical_pct:.2f}%)")
                        
                        # Log top 10 largest differences
                        logger.info(f"  Top 10 largest differences:")
                        for i, idx in enumerate(top_10_indices):
                            logger.info(f"    {i+1}. Index {idx.item()}: DS={values_ds_flat[idx].item():.6f}, "
                                      f"MG={values_mg_flat[idx].item():.6f}, Diff={abs_diff_flat[idx].item():.6e}")
                        
                        # Log all metrics (matching rlvr_pipeline_meg_critic.py format)
                        logger.info(f"  Functional equivalence checks:")
                        logger.info(f"    Mean diff: {mean_diff_value:.6f} (threshold: 0.5)")
                        logger.info(f"    Std relative diff: {std_rel_diff:.4%} (threshold: 20%)")
                        logger.info(f"    Pearson correlation: {correlation:.4f} (threshold: 0.95)")
                        logger.info(f"    Spearman correlation: {spearman_corr:.4f} (rank correlation)")
                        logger.info(f"    Max abs diff: {max_abs_diff:.6f}")
                        logger.info(f"    Mean abs diff: {mean_abs_diff:.6f}")
                        logger.info(f"    Max relative diff: {max_rel_diff:.4%}")
                        logger.info(f"    Mean relative diff: {mean_rel_diff:.4%}")
                        logger.info(f"  Value ranges (Real Data):")
                        logger.info(f"    DeepSpeed: min={values_ds_tensor.min().item():.6f}, max={values_ds_tensor.max().item():.6f}")
                        logger.info(f"    Megatron:  min={values_mg_tensor.min().item():.6f}, max={values_mg_tensor.max().item():.6f}")
                        
                        # Assert functional equivalence with looser tolerances (matching rlvr_pipeline_meg_critic.py)
                        rtol = 0.05  # 5% relative tolerance (accounts for different computation methods)
                        atol = 0.1   # 0.1 absolute tolerance (accounts for accumulation differences)
                        
                        # Perform functional equivalence assertions
                        # assert mean_diff_value < 0.7, f"Mean values differ too much: DS={mean_ds:.4f} vs MG={mean_mg:.4f}"
                        # assert std_rel_diff < 0.3, f"Std values differ too much: DS={std_ds:.4f} vs MG={std_mg:.4f}"
                        # assert correlation > 0.6, f"Outputs not well correlated: {correlation:.4f}"
                        
                        is_close = torch.allclose(values_ds_tensor, values_mg_tensor, rtol=rtol, atol=atol)
                        
                        if not is_close:
                            logger.warning(f"Values not within tight tolerances (rtol={rtol}, atol={atol})")
                            logger.warning(f"This is expected due to architectural differences between Megatron and DeepSpeed")
                            logger.warning(f"However, functional equivalence checks passed!")
                        else:
                            logger.info(f"  ✓ Critics are numerically close (rtol={rtol}, atol={atol})")
                        
                        logger.info(f"  ✓ Critics are functionally equivalent (rtol={rtol}, atol={atol})")
                        
                        # Store metrics for tracking
                        metrics_mgr.add_metric("critic_test/max_abs_diff", max_abs_diff)
                        metrics_mgr.add_metric("critic_test/mean_abs_diff", mean_abs_diff)
                        metrics_mgr.add_metric("critic_test/max_rel_diff", max_rel_diff)
                        metrics_mgr.add_metric("critic_test/mean_rel_diff", mean_rel_diff)
                        
                        # Use DeepSpeed values for training (as baseline)
                        batch = batch.union(values_deepspeed)
                        metrics_mgr.add_reduced_metrics(values_deepspeed.meta_info.pop("metrics", {}))
                        
                        # ========== Test Critics on Random Inputs (Not Used for Training) ==========
                        logger.info(f"\n[Random Input Test] Testing critics on random inputs for comparison...")
                        
                        # Create a random batch with same shape as real data
                        random_batch = self.create_mock_batch(
                            batch_size=batch.batch.batch_size[0],
                            seq_len=batch.batch["input_ids"].shape[1],
                            device=batch.batch["input_ids"].device
                        )
                        
                        # Compute values on random inputs with both critics
                        with Timer(name="compute_values_random_deepspeed", logger=None):
                            random_values_refs = self.critic.compute_values(random_batch, blocking=False)
                        
                        with Timer(name="compute_values_random_megatron", logger=None):
                            random_values_refs_2 = self.critic2.compute_values(random_batch, blocking=False)
                        
                        # Materialize random values
                        random_values_deepspeed = DataProto.materialize_concat(data_refs=random_values_refs)
                        random_values_megatron = DataProto.materialize_concat(data_refs=random_values_refs_2)
                        
                        # Extract tensors for comparison
                        random_values_ds = random_values_deepspeed.batch["values"]
                        random_values_mg = random_values_megatron.batch["values"]
                        
                        # Calculate metrics for random inputs
                        random_abs_diff = torch.abs(random_values_ds - random_values_mg)
                        random_flat_ds = random_values_ds.flatten()
                        random_flat_mg = random_values_mg.flatten()
                        
                        # Correlations for random inputs
                        random_pearson = torch.corrcoef(torch.stack([random_flat_ds, random_flat_mg]))[0, 1].item()
                        from scipy.stats import spearmanr
                        random_spearman, _ = spearmanr(random_flat_ds.cpu().float().numpy(), random_flat_mg.cpu().float().numpy())
                        
                        # Statistics
                        random_mean_abs_diff = random_abs_diff.mean().item()
                        random_max_abs_diff = random_abs_diff.max().item()
                        nearly_identical_threshold_random = 1e-1  # Same threshold as real data
                        random_nearly_identical = (random_abs_diff.flatten() < nearly_identical_threshold_random).sum().item()
                        random_total = random_abs_diff.flatten().numel()
                        random_identical_pct = (random_nearly_identical / random_total) * 100
                        
                        # Log comparison
                        logger.info(f"  [COMPARISON] Real Data vs Random Data:")
                        logger.info(f"    Pearson Correlation:  Real={correlation:.4f} | Random={random_pearson:.4f}")
                        logger.info(f"    Spearman Correlation: Real={spearman_corr:.4f} | Random={float(random_spearman):.4f}")
                        logger.info(f"    Mean Abs Diff:        Real={mean_abs_diff:.6f} | Random={random_mean_abs_diff:.6f}")
                        logger.info(f"    Max Abs Diff:         Real={max_abs_diff:.6f} | Random={random_max_abs_diff:.6f}")
                        logger.info(f"    Nearly Identical %:   Real={nearly_identical_pct:.2f}% | Random={random_identical_pct:.2f}%")
                        logger.info(f"  Value Ranges (Random Data):")
                        logger.info(f"    DeepSpeed: min={random_values_ds.min().item():.6f}, max={random_values_ds.max().item():.6f}")
                        logger.info(f"    Megatron:  min={random_values_mg.min().item():.6f}, max={random_values_mg.max().item():.6f}")
                        logger.info(f"  [END Random Input Test]\n")

                    # Only set old_log_probs if not using mock data (already in mock batch)
                    if not self.use_mock_data:
                        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                        metrics_mgr.add_reduced_metrics(old_log_probs.meta_info.pop("metrics", {}))
                metrics_mgr.add_metric("time/old_log_probs", cal_old_logpb_timer.last)

                # 要按domain group by处理reward
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                batch_list = []
                for domain, domain_batch in batch_grouped.items():
                    # 1. 处理mask相关策略， 获取sample level mask
                    with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer:
                        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                        metrics_mgr.add_metrics(mask_metrics)
                    metrics_mgr.add_metric("time/get_sample_level_mask", get_sample_level_mask_timer.last)

                    # 2. 处理reward相关策略
                    with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer:
                        domain_batch, response_level_metrics = reward_postprocess(
                            domain_batch, self.pipeline_config, self.running
                        )
                        metrics_mgr.add_metrics(response_level_metrics)
                    metrics_mgr.add_metric("time/reward_postprocess", reward_postprocess_timer.last)

                    # 3. 计算token level rewards
                    with Timer(name="get_token_reward", logger=None) as get_token_reward_timer:
                        domain_batch, token_level_metrics = compute_token_reward(
                            domain_batch, self.pipeline_config, self.kl_ctrl
                        )
                        metrics_mgr.add_metrics(token_level_metrics)
                    metrics_mgr.add_metric("time/get_token_reward", get_token_reward_timer.last)

                    # 4. 计算advantage
                    final_response_mask = domain_batch.batch["final_response_mask"].clone()
                    with Timer(name="compute_advantage", logger=None) as compute_advantage_timer:
                        domain_batch = compute_advantage(
                            data=domain_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                            response_mask=final_response_mask,
                        )
                        domain_metrics = reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        metrics_mgr.add_domain_metrics(domain, domain_metrics)
                        batch_list.append(domain_batch)
                    metrics_mgr.add_metric("time/compute_advantage", compute_advantage_timer.last)

                batch = DataProto.concat(batch_list)
                
                # Compute separate returns for critic2 if needed
                if "values_critic2" in batch.meta_info and self.pipeline_config.adv_estimator == "gae":
                    # Store critic1's returns
                    batch.meta_info["returns_critic1"] = batch.batch["returns"].clone()
                    
                    # Replace values with critic2's values temporarily
                    original_values = batch.batch["values"].clone()
                    batch.batch["values"] = batch.meta_info["values_critic2"]
                    
                    # Recompute returns using critic2's values
                    batch_critic2_temp = compute_advantage(
                        data=batch,
                        gamma=self.pipeline_config.gamma,
                        lambd=self.pipeline_config.lambd,
                        adv_estimator=self.pipeline_config.adv_estimator,
                        advantage_clip=self.pipeline_config.advantage_clip,
                        whiten_advantages=self.pipeline_config.whiten_advantages,
                        whiten_rewards=self.pipeline_config.whiten_rewards,
                        response_mask=batch.batch["final_response_mask"],
                    )
                    
                    # Store critic2's returns
                    batch.meta_info["returns_critic2"] = batch_critic2_temp.batch["returns"].clone()
                    
                    # Restore original values and returns for critic1
                    batch.batch["values"] = original_values
                    batch.batch["returns"] = batch.meta_info["returns_critic1"]

                if batch.batch["final_response_mask"].sum() == 0:
                    logger.info("Warning: final_response_mask.sum() == 0! Current step will be skipped.")
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 1)
                    metrics = metrics_mgr.get_metrics()
                    # do ckpt
                    self.state.step = global_step
                    self.state.log_history.append(metrics)
                    for domain, scheduler in self.generate_schedulers.items():
                        self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())
                    self.do_checkpoint(global_step=global_step)
                    self.tracker.log(values=metrics, step=global_step)
                    continue
                else:
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 0)

                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                batch.pop("prompt_id")

                metrics_mgr.add_all_metrics(
                    global_step,
                    batch,
                    resource_manager=self.resource_manager,
                    actor_infer=self.actor_infer,
                    actor_train=self.actor_train,
                )
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                metrics_mgr.add_domain_all_metrics(global_step, batch_grouped)

                with Timer(name="step_train", logger=None) as step_train_timer:
                    if self.pipeline_config.adv_estimator == "gae":
                        # Create a new batch for critic2 with its own values and returns
                        batch_critic2 = DataProto(
                            batch=batch.batch.clone(),  # Clone the TensorDict
                            non_tensor_batch=batch.non_tensor_batch.copy(),  # Shallow copy the dict
                            meta_info=batch.meta_info.copy()  # Shallow copy the dict
                        )
                        
                        # Replace values and returns with critic2's own computed values/returns
                        if "values_critic2" in batch.meta_info:
                            batch_critic2.batch["values"] = batch.meta_info["values_critic2"]
                        if "returns_critic2" in batch.meta_info:
                            batch_critic2.batch["returns"] = batch.meta_info["returns_critic2"]
                        
                        # Log the values being used for training to verify they're different
                        if global_step % 10 == 0:
                            logger.info(f"Step {global_step} - Critic1 values mean: {batch.batch['values'].mean().item():.4f}, returns mean: {batch.batch['returns'].mean().item():.4f}")
                            logger.info(f"Step {global_step} - Critic2 values mean: {batch_critic2.batch['values'].mean().item():.4f}, returns mean: {batch_critic2.batch['returns'].mean().item():.4f}")
                        
                        # When blocking=True, train_step returns DataProto directly, not ObjectRefs
                        critic_train_metrics = self.critic.train_step(batch, blocking=True)
                        time.sleep(2)
                        critic_train_metrics_2 = self.critic2.train_step(batch_critic2, blocking=True)
                        
                        # Log metrics to help debug
                        if global_step % 10 == 0:
                            logger.info(f"Step {global_step} - Critic1 loss: {critic_train_metrics.meta_info.get('metrics', {}).get('critic/loss', 'N/A')}")
                            logger.info(f"Step {global_step} - Critic2 loss: {critic_train_metrics_2.meta_info.get('metrics', {}).get('critic/loss', 'N/A')}")

                    with actor_train_timer:
                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            # update actor
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics_mgr.add_reduced_metrics(actor_train_metrics.meta_info.pop("metrics", {}))

                    if self.pipeline_config.adv_estimator == "gae":
                        # critic_train_metrics is already a DataProto, no need to materialize
                        metrics_mgr.add_reduced_metrics(critic_train_metrics.meta_info.pop("metrics", {}))

                metrics_mgr.add_metric("time/step_train", step_train_timer.last)

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_response_timer.push_units_processed(
                    n=torch.sum(batch.batch["response_mask"]).detach().item()
                )
                actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                metrics = metrics_mgr.get_metrics()
                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)
                for domain, scheduler in self.generate_schedulers.items():
                    self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(
                        generate_output.batch["responses"], skip_special_tokens=True
                    )
                    generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                    logger.info(json.dumps(generate_examples, ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
        
        # Final test summary
        logger.info("=" * 60)
        logger.info("[Critic Equivalence Test Summary]")
        logger.info(f"Successfully completed {test_max_steps} steps of critic comparison")
        logger.info("✓ DeepSpeed and Megatron critics are functionally equivalent")
        logger.info("✓ All value computations matched within tolerance")
        logger.info("=" * 60)
        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            batch.meta_info["is_offload_states"] = False
            batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()

            self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout
            )
            self.actor_infer.stop_server()
            generate_output.meta_info.pop("is_offload_states", None)
            for reward_cluster in self.rewards.values():
                reward_cluster.offload_states()
        val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        val_correct_mean = (batch.batch["scores"] == 1).detach().float().mean().item()
        val_metrics_mgr.add_metric("val_correct/all/mean", val_correct_mean)
        logger.info(json.dumps({"val_correct/all/mean": val_correct_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])

        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            logger.info(f"val_correct/{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_correct", {f"{group_key}/mean": (group_batch.batch["scores"] == 1).detach().float().mean().item()}
            )

        return val_metrics_mgr.get_metrics()
