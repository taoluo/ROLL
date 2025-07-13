import copy
import itertools
import queue
import random
import threading
import time
from collections import defaultdict
from typing import Any, Union, Optional, Dict, List, Set

import numpy as np
import ray
import torch
from datasets import Dataset
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto, collate_fn
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import (
    postprocess_generate,
    reduce_metrics,
    concatenate_input_and_output,
    GenerateRequestType,
)
from roll.utils.logging import get_logger
from roll.utils.multi_thread_utils import ThreadSafeDict
from pprint import pprint
logger = get_logger()


@ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 128})
class GenerateScheduler:

    def __init__(self, pipeline_config=None):
        self.cluster: Union[Any, Cluster] = None
        self.pipeline_config = pipeline_config
        self.progress_bar: Optional[tqdm] = None
        self.request_counter = itertools.count()
        self.dp_fetch_count = {}
        self.load_balance_coordinator = {}
        self.mp_rank_zero = {}
        self.data: Optional[DataProto] = None
        self.responses: Dict[int, List[DataProto]] = defaultdict(list)
        self.request_id_2_prompt_id: Dict[str, int] = {}
        self.prompt_id_2_request_ids: Dict[int, set] = defaultdict(set)
        self.response_batch_size: Optional[int] = None
        self.abort_request_ids: set[str] = set()
        self.input_data: Optional[DataProto] = None
        self.is_completed = False
        self.request_id_2_dp_rank = {}
        self.completed_count = set()
        self.prompt_count = 0
        self.max_running_requests = 128
        self.alive_check_interval = 10
        self.last_alive_check = time.time()
        self.lock = threading.Lock()
        self.response_callback_fn = None

    def generate(self, data: DataProto, actor_cluster: Union[Any, Cluster], pipeline_config) -> DataProto:
        self.response_callback_fn = data.meta_info["response_callback_fn"]
        self.pipeline_config = pipeline_config
        self.cluster = actor_cluster
        if len(self.mp_rank_zero) == 0:
            dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.cluster.worker_rank_info]
            for i, dp_rank in enumerate(dp_ranks):
                rank_info = self.cluster.get_rank_info(rank=i)
                if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                    self.mp_rank_zero[dp_rank] = self.cluster.workers[i]
        self.dp_fetch_count = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.load_balance_coordinator = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.request_id_2_prompt_id.clear()
        self.prompt_id_2_request_ids.clear()
        self.abort_request_ids.clear()
        self.request_id_2_dp_rank.clear()
        self.completed_count.clear()

        generate_opt_level = pipeline_config.generate_opt_level
        num_return_sequences = actor_cluster.worker_config.generating_args.num_return_sequences

        is_num_return_sequences_expand = pipeline_config.is_num_return_sequences_expand
        if generate_opt_level == 0 and is_num_return_sequences_expand:
            logger.warning("is_num_return_sequences_expand=True and generate_opt_level may reduce performance.")

        data.batch["prompt_id"] = torch.arange(data.batch.batch_size[0], device=data.batch.device)
        self.input_data = data
        data.meta_info["is_num_return_sequences_expand"] = is_num_return_sequences_expand
        data.meta_info["num_return_sequences"] = num_return_sequences

        self.prompt_count = self.input_data.batch.batch_size[0]

        generation_config = self.cluster.worker_config.generating_args.to_dict()
        generation_config["num_return_sequences"] = num_return_sequences
        if is_num_return_sequences_expand:
            generation_config["num_return_sequences"] = 1
        data.meta_info["generation_config"] = generation_config

        if generate_opt_level == 0:
            if is_num_return_sequences_expand:
                batch_size = data.batch.batch_size[0]
                output_batch_size = batch_size * num_return_sequences
                input_ids = data.batch["input_ids"]
                attention_mask = data.batch["attention_mask"]
                position_ids = data.batch["position_ids"]
                input_ids = input_ids.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                attention_mask = (
                    attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                )
                if position_ids.dim() == 3:  # (bsz, 3, seqlen)
                    # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                    position_ids = (
                        position_ids.unsqueeze(1)
                        .repeat(1, num_return_sequences, 1, 1)
                        .view(output_batch_size, *position_ids.shape[-2:])
                    )
                else:
                    position_ids = (
                        position_ids.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                    )

                non_tensor_batch = dict(
                    (k, np.repeat(v, num_return_sequences)) for k, v in data.non_tensor_batch.items()
                )

                data = DataProto(
                    batch=TensorDict(
                        {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
                        batch_size=output_batch_size,
                    ),
                    non_tensor_batch=non_tensor_batch,
                    meta_info=data.meta_info,
                )
            ret = self.cluster.generate(data)
            self.input_data = None
            return ret
        elif generate_opt_level == 1:
            # async + load balance
            if is_num_return_sequences_expand:
                batch_size = data.batch.batch_size[0]
                output_batch_size = batch_size * num_return_sequences
                input_ids = data.batch["input_ids"]
                attention_mask = data.batch["attention_mask"]
                position_ids = data.batch["position_ids"]
                prompt_ids = data.batch["prompt_id"]
                input_ids = input_ids.repeat(num_return_sequences, 1)
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
                if position_ids.dim() == 3:  # (bsz, 3, seqlen)
                    position_ids = position_ids.repeat(num_return_sequences, 1, 1)
                    non_tensor_batch = dict(
                        (k, np.tile(v, num_return_sequences))
                        for k, v in data.non_tensor_batch.items())
                else:
                    position_ids = position_ids.repeat(num_return_sequences, 1)
                    non_tensor_batch = {}
                prompt_ids = prompt_ids.unsqueeze(-1).repeat(num_return_sequences, 1)

                data = DataProto(
                    batch=TensorDict(
                        {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "position_ids": position_ids,
                            "prompt_id": prompt_ids,
                        },
                        batch_size=output_batch_size,
                    ),
                    non_tensor_batch=non_tensor_batch,
                    meta_info=data.meta_info,
                )
            self.is_completed = False
            ret = self.generate_opt_level_1(data)
            self.input_data = ret
            return ret
        else:
            raise NotImplementedError(f"not support generate_opt_level {generate_opt_level}")

    def get_available_dp_rank(self):
        while True:
            # 负载均衡逻辑，期望各dp 正在处理的条数基本接近
            sorted_ranks = sorted(
                self.load_balance_coordinator.keys(), key=lambda rank: (self.load_balance_coordinator[rank], rank)
            )
            if self.load_balance_coordinator[sorted_ranks[0]] < self.max_running_requests:
                yield sorted_ranks[0]

    def send_request_to_one_worker(self, data: DataProto):
        dp_rank = next(self.get_available_dp_rank())
        ray.get(self.cluster.workers[dp_rank].add_request.remote(command=GenerateRequestType.ADD, data=data))
        self.load_balance_coordinator[dp_rank] += 1
        self.dp_fetch_count[dp_rank] += 1

    def generate_opt_level_1(self, data: DataProto):
        # async++
        is_num_return_sequences_expand = self.pipeline_config.is_num_return_sequences_expand
        num_return_sequences = self.cluster.worker_config.generating_args.num_return_sequences

        response_batch_size = 1 if is_num_return_sequences_expand else num_return_sequences
        self.response_batch_size = response_batch_size
        self.progress_bar = tqdm(
            total=self.prompt_count, desc="generate progress(prompt)", mininterval=int(self.prompt_count * 0.1) + 1
        )

        self.data = data
        self.responses: Dict[int, List[DataProto]] = defaultdict(list)

        logger.info(
            f"request id size: {data.batch.batch_size[0]} "
            f"response_batch_size: {response_batch_size} "
            f"is_num_return_sequences_expand: {is_num_return_sequences_expand}"
        )
        self.cluster.start_server(data=DataProto(meta_info=data.meta_info), blocking=True)

        # 分发数据至收到target rollout 完成
        # 无限循环，把所有的response发送给dp worker
        send_request_count = 0
        request_refs = []
        data_index_counter = itertools.count()
        last_alive_check = time.time()
        while not self.is_completed:

            # 探测dp worker是否存活，dp worker的server thread可能由于异常退出，造成hang
            current_time = time.time()
            if current_time - last_alive_check >= self.alive_check_interval:
                self.cluster.add_request(command=GenerateRequestType.ALIVE_CHECK, data=DataProto())
                last_alive_check = current_time

            if send_request_count < data.batch.batch_size[0]:
                # 取一个可以发送request的dp worker
                dp_rank = next(self.get_available_dp_rank())

                # 还有数据需要发送, 取需要发送的数据
                # request_id 全局递增，否则vllm/sglang scheduler状态不对
                request_id = next(self.request_counter)
                data_index = next(data_index_counter)
                request_data = collate_fn([self.data[data_index]])
                request_data.meta_info["request_id"] = str(request_id)
                prompt_id = self.data[data_index].batch["prompt_id"].item()
                self.request_id_2_prompt_id[request_data.meta_info["request_id"]] = request_data.batch[
                    "prompt_id"
                ].item()
                self.request_id_2_dp_rank[request_data.meta_info["request_id"]] = dp_rank
                self.prompt_id_2_request_ids[prompt_id].add(request_data.meta_info["request_id"])
                # 需要注意上面的调用顺序, report_response中会更新request_id索引dp_rank，所以这里需要最后add request_id
                request_data.meta_info["response_callback_fn"] = self.response_callback_fn
                request_data.meta_info["generation_config"] = data.meta_info["generation_config"]
                request_refs.append(
                    self.cluster.workers[dp_rank].add_request.remote(
                        command=GenerateRequestType.ADD, data=request_data
                    )
                )
                with self.lock:
                    self.load_balance_coordinator[dp_rank] += 1
                self.dp_fetch_count[dp_rank] += 1
                send_request_count += 1
                if len(request_refs) % self.cluster.world_size == 0:
                    ray.get(request_refs)
                    request_refs = []

        gen_metrics = self.cluster.stop_server()
        generate_return_num = num_return_sequences
        response_ids_list_of_list = []
        eos_token_id = None
        pad_token_id = None
        for sample_index in range(len(self.responses)):
            response_ids_list = []
            for response in self.responses[sample_index]:
                eos_token_id = response.meta_info["eos_token_id"]
                pad_token_id = response.meta_info["pad_token_id"]
                response_ids_list.extend(response.meta_info["output_token_ids"])
            assert (
                len(response_ids_list) >= generate_return_num
            ), f"response_ids_list length {len(response_ids_list)} < generate_return_num {generate_return_num}"
            response_ids_list_of_list.extend(response_ids_list[:generate_return_num])

        response_ids_list_of_list = [torch.tensor(token_ids) for token_ids in response_ids_list_of_list]
        output_tensor = pad_sequence(response_ids_list_of_list, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=self.input_data.batch["input_ids"],
            output_ids=output_tensor,
            num_return_sequences=generate_return_num,
        )
        output: DataProto = postprocess_generate(
            prompts=self.input_data,
            output=output_tensor,
            num_return_sequences=generate_return_num,
            sequence_length=self.pipeline_config.sequence_length,
            canonical_prompt_length=self.pipeline_config.prompt_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        _, sorted_indices = torch.sort(output.batch["prompt_id"])
        output.reorder(indices=sorted_indices)
        output.pop("prompt_id")
        self.data = None
        output.meta_info["metrics"] = reduce_metrics(gen_metrics.meta_info.pop("metrics", {}))
        logger.info(f"dp_fetch_count: {self.dp_fetch_count}")
        return output

    @ray.method(concurrency_group="single_thread")
    def report_response(self, data: DataProto):
        """
        本质上也是维护了一个状态机
        """
        request_id = data.meta_info["request_id"]
        prompt_id = self.request_id_2_prompt_id[request_id]
        dp_rank = self.request_id_2_dp_rank[request_id]
        with self.lock:
            self.load_balance_coordinator[dp_rank] -= 1

        if self.is_completed:
            return

        self.responses[prompt_id].append(data)
        required_response_count = self.cluster.worker_config.generating_args.num_return_sequences
        self.prompt_id_2_request_ids[prompt_id].remove(data.meta_info["request_id"])
        if len(self.responses[prompt_id]) * self.response_batch_size >= required_response_count:
            # 取已经完成的prompt_id，对应的request_ids，需要都取消
            if prompt_id not in self.completed_count:
                self.progress_bar.update(1)
            self.completed_count.add(prompt_id)
            abort_refs = []
            for request_id in self.prompt_id_2_request_ids[prompt_id]:
                with self.lock:
                    self.load_balance_coordinator[dp_rank] -= 1
                abort_refs.append(
                    self.cluster.workers[dp_rank].add_request.remote(
                        command=GenerateRequestType.ABORT, data=DataProto(meta_info={"request_id": request_id})
                    )
                )
        if len(self.completed_count) >= self.prompt_count:
            self.is_completed = True


# @ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 256})
@ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 1})
class DynamicSamplingScheduler:

    def __init__(self, pipeline_config=None):
        self.pipeline_config = pipeline_config
        set_seed(seed=pipeline_config.seed)
        self.progress_bar: Optional[tqdm] = None
        self.request_counter = None
        self.dp_fetch_count = {}
        self.load_balance_coordinator = {}
        self.mp_rank_zero = {}
        self.request_id_2_prompt_id: Dict[str, int] = {}
        self.prompt_id_2_request_ids: Dict[int, set] = defaultdict(set)
        self.response_batch_size: Optional[int] = None
        self.abort_request_ids: set[str] = set()
        self.request_id_2_dp_rank = {}
        self.requests_buffers: Dict[str, DataProto] = {}
        self.lock = threading.Lock()
        self.last_alive_check = time.time()
        self.dataset_iter_count = 0
        self.exception_queue = queue.Queue()
        self.running = False
        self.dataset_epoch = 0

        # Flow control measures. max_running_requests limits the maximum number of concurrent requests for each dp.
        # max_additional_running_prompts limits the number of prompts running simultaneously to avoid excessive consumption of prompts.
        self.max_running_requests = self.pipeline_config.max_running_requests
        self.max_additional_running_prompts = self.pipeline_config.max_additional_running_prompts
        self.is_use_additional_prompts = self.pipeline_config.is_use_additional_prompts
        self.alive_check_interval = self.pipeline_config.alive_check_interval

        self.actor_cluster = None
        self.reward_clusters = None
        self.reward_worker_iters = None
        self.dataset = None
        self.indices = []
        self.batch_size = None
        self.dataset_iter = None
        self.collect_fn_cls = None
        self.collect_fn_kwargs = None
        self.collect_fn = None
        self.tokenizer = None
        self.response_filter_fn = None
        self.query_filter_fn = None
        self.response_callback_fn = None
        self.generation_config = None

        self.completed_buffers = None
        self.query_group_buffers = None

        self.query_filter_count = 0
        self.response_filter_count = 0
        self.running_prompts = 0
        self.response_cache: Dict[str, List] = None
        self.prompt_use_count = 0
        self.postprocessed_requests_count = 0

    def set_scheduler(
        self,
        actor_cluster: Union[Any, Cluster],
        reward_clusters: Dict[str, Union[Any, Cluster]],
        dataset: Dataset,
        collect_fn_cls,
        collect_fn_kwargs,
        response_filter_fn=None,
        query_filter_fn=None,
        response_callback_fn=None,
        state: Dict[str, Any] = None,
    ):
        """
        GenerateScheduler可以由多个实例，不再局限于单例
        """
        self.actor_cluster = actor_cluster
        self.reward_clusters = reward_clusters
        self.reward_worker_iters = {}
        for domain, cluster in reward_clusters.items():
            self.reward_worker_iters[domain] = itertools.cycle(cluster.workers)

        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if state is not None and state.get("dataset_iter_count", 0) > 0:
            for _ in range(state["dataset_iter_count"]):
                self.get_next_dataset_item()

        self.collect_fn_cls = collect_fn_cls
        self.collect_fn_kwargs = collect_fn_kwargs
        self.tokenizer = default_tokenizer_provider(model_args=self.actor_cluster.worker_config.model_args)
        self.collect_fn = self.collect_fn_cls(tokenizer=self.tokenizer, **self.collect_fn_kwargs)

        if self.is_use_additional_prompts:
            self.response_filter_fn = response_filter_fn
            self.query_filter_fn = query_filter_fn
        else:
            self.response_filter_fn = lambda data_list, config: True
            self.query_filter_fn = lambda data_list, config: True
            logger.info(f"use_additional_prompts is False, disable query and response filtering.")
        self.response_callback_fn = response_callback_fn
        dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.actor_cluster.worker_rank_info]
        for i, dp_rank in enumerate(dp_ranks):
            rank_info = self.actor_cluster.get_rank_info(rank=i)
            if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                self.mp_rank_zero[dp_rank] = self.actor_cluster.workers[i]

        self.request_counter = GlobalCounter.options(
            name=f"DynamicSchedulerRequestCounter",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        import os
        os.environ.setdefault("PYDEVD_USE_CYTHON", "NO")
        os.environ.setdefault("PYDEVD_USE_FRAME_EVAL", "NO")
        import pydevd_pycharm

        # Differentiate schedulers by use_additional_prompts config
        if hasattr(self.pipeline_config, 'use_additional_prompts') and not self.pipeline_config.use_additional_prompts:
            # Validation scheduler (use_additional_prompts = False)
            debug_port = 12346
            scheduler_type = "VALIDATION"
        else:
            # Training scheduler (use_additional_prompts = True or default)
            debug_port = 12344
            scheduler_type = "TRAINING"
            logger.info(f"Connecting PyCharm debugger on port {debug_port}")
            if os.getenv("PYCHARM", "0") == "1":
                pydevd_pycharm.settrace('localhost', port=debug_port, stdoutToServer=True, stderrToServer=True, suspend=False)
            logger.info(f"PyCharm debugger attached to {scheduler_type} scheduler on port {debug_port}")
            print(f"PyCharm debugger attached to {scheduler_type} scheduler on port {debug_port}")


    def reset_status(self):
        self.completed_buffers: Dict[int, List[DataProto]] = defaultdict(list)
        self.query_group_buffers: Dict[int, List[DataProto]] = defaultdict(list)

        self.dp_fetch_count = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.load_balance_coordinator = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.request_id_2_prompt_id.clear()
        self.prompt_id_2_request_ids.clear()
        self.abort_request_ids.clear()
        self.request_id_2_dp_rank.clear()
        self.requests_buffers.clear()
        self.response_filter_count = 0
        self.query_filter_count = 0
        self.running_prompts = 0
        self.prompt_use_count = 0
        self.response_cache = defaultdict(list)
        self.exception_queue = queue.Queue()
        bar_name = "-".join(self.reward_clusters.keys())
        self.progress_bar = tqdm(
            total=self.batch_size,
            desc=f"{bar_name} generate progress(prompt)",
            mininterval=int(self.batch_size * 0.1) + 1,
        )
        self.interrupted_query_group_buffers: Dict[int, List[DataProto]] = defaultdict(list)


    def get_batch(self, data: DataProto, batch_size: int) -> DataProto:
        """
        从dataset里，按给定策略sample batch
        1. 常规无过滤
        2. 动态过滤
        """
        self.batch_size = batch_size
        self.reset_status()
        self.running = True
        prompt_id_counter = itertools.count()
        self.generation_config = copy.deepcopy(data.meta_info["generation_config"])
        num_return_sequences = self.generation_config["num_return_sequences"]
        has_interrupted_any = False
        enable_migration = True
        # enable_migration = False
        # interrupt_timeout_threshold = [ 2, 4, 6]
        interrupt_timeout_threshold = [ 2, ]
        last_interrupt_threshold_count = 0
        start = time.time()
        while True:
            if (
                sum([len(v) for v in list(self.completed_buffers.values())[:]])
                >= self.batch_size * num_return_sequences
            ):
                self.running = False
                break
            self.check_worker_alive(self.actor_cluster)
            self.check_response_callback()


            elapse_time = time.time() - start
            current_timeout_threshold_count = sum([ elapse_time> threshold for threshold in interrupt_timeout_threshold])
            # if enable_migration and (not has_interrupted_any) and self.postprocessed_requests_count > 0:
            if enable_migration and  current_timeout_threshold_count > last_interrupt_threshold_count:
                # todo if interrupted buffer >0, resend interrupted.
                # logger.info(f"Migration: postprocessed_requests_count {self.postprocessed_requests_count} > 0, check if need to migrate interrupted query group buffers.")
                logger.info(f"Migration: interrupt after {elapse_time=}sec {current_timeout_threshold_count=}")
                # self.interrupt_all_requests_by_dp_rank(0)  # interrupt all even ranks
                # has_interrupted_any = True
                # interrupt req 0, keep req 1 for comparison
                self.actor_cluster.workers[0].add_request.remote(
                    command=GenerateRequestType.INTERRUPT, data=DataProto(meta_info={"request_id": '1'})
                )
                # interrupt req 1 and 0
                self.actor_cluster.workers[0].add_request.remote(
                    command=GenerateRequestType.INTERRUPT, data=DataProto(meta_info={"request_id": '0'})
                )
                last_interrupt_threshold_count = current_timeout_threshold_count

            if self.interrupted_query_group_buffers:

                # assert False,  f"Migration: Sending interrupted query group to DP rank {list(self.interrupted_query_group_buffers.keys())}"

                target_dp_rank = next(self.get_available_dp_rank())

                self.send_one_interrupted_query_group_to_dp_new(target_dp_rank)
                logger.info(f"Migration: Sending interrupted query group to DP rank {target_dp_rank} and skip sending any new request.")

                continue

            if not self.check_send_new_request():
                time.sleep(1)
                continue

            # get a query from dataset
            prompt_id = next(prompt_id_counter)
            dataset_item = self.get_fixed_dataset_item(0)  # Use fixed dataset item for testing
            # dataset_item = self.get_next_dataset_item()  # Use different dataset items
            # tao: rvst hardcode for testing debug interrupt, remove this later
            # dataset_item = self.get_fixed_dataset_item(0)  # This was causing identical input_ids!
            
            domain = dataset_item.get("domain", "default")
            collect_data = self.collect_fn([dataset_item])
            request_data: DataProto = DataProto.from_single_dict(collect_data, meta_info=data.meta_info)
            
            # replica, redundancy
            request_data_list = self.expand_requests(request_data)

            dp_rank = next(self.get_available_dp_rank())
            with self.lock:
                self.prompt_use_count += 1
                self.running_prompts += 1
                for req in request_data_list:
                    # get a available worker, 需要控制max_running_request, 当前策略会始终保持worker的满载
                    request_id = ray.get(self.request_counter.get_value.remote())
                    req.meta_info["request_id"] = f"{request_id}"
                    req.meta_info["response_callback_fn"] = self.response_callback_fn
                    self.request_id_2_prompt_id[req.meta_info["request_id"]] = prompt_id
                    self.request_id_2_dp_rank[req.meta_info["request_id"]] = dp_rank
                    self.prompt_id_2_request_ids[prompt_id].add(req.meta_info["request_id"])  # 用于replica情况
                    self.requests_buffers[req.meta_info["request_id"]] = req
                    ray.get(
                        self.actor_cluster.workers[dp_rank].add_request.remote(
                            command=GenerateRequestType.ADD, data=req
                        )
                    )
                    req.meta_info.pop("response_callback_fn")
                    self.load_balance_coordinator[dp_rank] += 1
                    self.dp_fetch_count[dp_rank] += 1

        completed_buffers = {k: v for k, v in self.completed_buffers.items() if len(v) > 0}
        collect_data = [item for sublist in list(completed_buffers.values())[:] for item in sublist]
        
        # **LOG ALL INDIVIDUAL REQUESTS**: Before concatenation, log each request's detailed info
        logger.info(f"SCHEDULER_COLLECT_DATA: Found {len(collect_data)} total responses from {len(completed_buffers)} queries")
        for i, data_item in enumerate(collect_data):
            request_id = data_item.meta_info.get("request_id", f"unknown_{i}")
            finish_status = data_item.meta_info.get("finish_status", "UNKNOWN")
            is_continued = data_item.meta_info.get("is_continued_request", False)
            migration_count = data_item.meta_info.get("migration_count", 0)
            original_request_id = data_item.meta_info.get("original_request_id", request_id)
            domain = data_item.non_tensor_batch.get("domain", ["UNKNOWN"])[0] if "domain" in data_item.non_tensor_batch else "UNKNOWN"
            
            # Decode prompt and response for logging
            if "prompts" in data_item.batch:
                prompt_text = self.tokenizer.decode(data_item.batch["prompts"][0], skip_special_tokens=True)
            else:
                prompt_text = "NO_PROMPT_IN_BATCH"
            
            if "responses" in data_item.batch:
                response_text = self.tokenizer.decode(data_item.batch["responses"][0], skip_special_tokens=True)
                response_length = len(data_item.batch["responses"][0])
            else:
                response_text = "NO_RESPONSE_IN_BATCH"
                response_length = 0
            
            logger.info(f"COLLECT_request_id={request_id}, original_id={original_request_id}, domain={domain}, is_continued={is_continued}, migrations={migration_count}, finish_status={finish_status}, response_length={response_length}")
            logger.info(f"COLLECT_PROMPT_{request_id}: \n{prompt_text}")
            logger.info(f"COLLECT_RESPONSE_{request_id}: \n{response_text}")
        
        query_use_count = next(prompt_id_counter)
        logger.info(
            f"total collect data: {len(collect_data)}, collect queries: {len(completed_buffers)} "
            f"used queries: {query_use_count}  query_filter_count: {self.query_filter_count} "
            f"response_filter_count: {self.response_filter_count}"
        )

        # TODO: 这里 len(collect_data) > rollout_batch_size, 可以尝试动态扩大batch_size
        batch = DataProto.concat(collect_data[: self.batch_size * num_return_sequences])
        batch.meta_info["metrics"] = {
            f"scheduler/query_filter_count": self.query_filter_count,
            f"scheduler/response_filter_count": self.response_filter_count,
            f"scheduler/collect_query_count": len(completed_buffers),
            f"scheduler/query_use_count": query_use_count,
        }

        # 统计全部response metrics
        metrics = {}
        for domain, response_batches in self.response_cache.items():
            response_batch = DataProto.concat(response_batches[:])
            sequence_score = response_batch.batch["scores"]
            metrics[f"scheduler/{domain}/score/mean"] = torch.mean(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/max"] = torch.max(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/min"] = torch.min(sequence_score).detach().item()

        batch.meta_info["metrics"].update(metrics)
        self.reset_status()

        return batch

    def send_one_interrupted_query_group_to_dp_new(self, target_dp_rank: int):
        """
        Send interrupted query group to a target DP rank for continuation.
        Enhanced to handle multiple interruptions/migrations by tracking original prompt length
        and cumulative partial output length.
        """
        with self.lock:
            assert self.interrupted_query_group_buffers, "Migration: No interrupted query groups in buffer to migrate"

            prompt_id, interrupted_batches = self.interrupted_query_group_buffers.popitem()
            assert len(interrupted_batches) > 0, f"Migration: Empty interrupted batches for prompt_id {prompt_id}"

            # --- AGGREGATE ALL PARTIAL OUTPUTS ---
            # The core fix is to process all partial outputs for a request together,
            # not individually in a loop.

            # 1. Get the original request details from the first interrupted batch.
            #    All batches for a prompt_id share the same original request.
            first_batch = interrupted_batches[0]
            original_request_id = first_batch.meta_info["request_id"]
            assert original_request_id in self.requests_buffers, f"Original request not found for {original_request_id}"
            original_request = self.requests_buffers[original_request_id]
            original_prompt_ids = original_request.batch["input_ids"]
            original_attention_mask = original_request.batch["attention_mask"]
            original_prompt_length = original_attention_mask.sum().item()

            # 2. Concatenate all partial token chunks from all interruptions.
            all_partial_tokens = []
            for batch in interrupted_batches:
                partial_tokens = batch.meta_info.get("output_token_ids", [])
                if partial_tokens and len(partial_tokens) > 0 and len(partial_tokens[0]) > 0:
                    all_partial_tokens.extend(partial_tokens[0])
            
            cumulative_partial_output_length = len(all_partial_tokens)

            # 3. Build the new, fully continued input.
            if cumulative_partial_output_length > 0:
                partial_output_tensor = torch.tensor(all_partial_tokens, device=original_prompt_ids.device)
                continued_input_ids = torch.cat([original_prompt_ids.squeeze(0), partial_output_tensor], dim=0).unsqueeze(0)
                
                # Extend the attention mask to cover the new tokens.
                partial_output_mask = torch.ones((1, cumulative_partial_output_length), device=original_prompt_ids.device, dtype=torch.long)
                continued_attention_mask = torch.cat([original_attention_mask, partial_output_mask], dim=1)
            else:
                # No new tokens, resubmit the original prompt.
                continued_input_ids = original_prompt_ids
                continued_attention_mask = original_attention_mask

            continued_position_ids = torch.arange(continued_input_ids.shape[1], device=continued_input_ids.device).unsqueeze(0)
            
            # --- NEW ASSERTIONS to validate the fix ---
            expected_total_length = original_prompt_length + cumulative_partial_output_length
            actual_total_length = continued_attention_mask.sum().item()
            assert actual_total_length == expected_total_length, \
                f"Assertion Failed: Reconstructed length mismatch. Expected {expected_total_length}, got {actual_total_length}"
            assert continued_input_ids.shape[1] == continued_attention_mask.shape[1], \
                f"Assertion Failed: Mismatch between input_ids shape ({continued_input_ids.shape[1]}) and attention_mask shape ({continued_attention_mask.shape[1]})"

            # 4. Create the single new migrated request.
            #    Use metadata from the *last* interrupted batch as it's the most recent.
            last_batch = interrupted_batches[-1]
            migrated_request = DataProto()
            
            batch_tensors = {
                    "input_ids": continued_input_ids,
                    "attention_mask": continued_attention_mask,
                    "position_ids": continued_position_ids,
                }
                # Copy other tensor fields from the original request
            for key in original_request.batch.keys():
                if key not in batch_tensors:
                    batch_tensors[key] = original_request.batch[key]
            
            migrated_request.batch = TensorDict(source=batch_tensors, batch_size=[1])
            migrated_request.non_tensor_batch = copy.deepcopy(original_request.non_tensor_batch)

            migrated_request.meta_info = last_batch.meta_info.copy()
            migrated_request.meta_info.pop('finish_status', None)
            migrated_request.meta_info["response_callback_fn"] = self.response_callback_fn
            migrated_request.meta_info["is_continued_request"] = True
            migrated_request.meta_info["original_prompt_length"] = original_prompt_length
            migrated_request.meta_info["cumulative_partial_output_length"] = cumulative_partial_output_length
            migrated_request.meta_info["migration_count"] = len(interrupted_batches)

            # Adjust max_new_tokens for the continued generation.
            generation_config = migrated_request.meta_info["generation_config"].copy()
            max_sequence_length = self.pipeline_config.sequence_length
            safety_buffer = 4
            max_allowed_new_tokens = max(1, max_sequence_length - actual_total_length - safety_buffer)
            original_max_new_tokens = generation_config.get("max_new_tokens", 512)
            generation_config["max_new_tokens"] = min(original_max_new_tokens, max_allowed_new_tokens)
            migrated_request.meta_info["generation_config"] = generation_config
                
            # Reuse the original request ID for the resumed request.
            migrated_request.meta_info["request_id"] = original_request_id

            # Update the buffers and mappings with the new state for the original request ID.
            self.requests_buffers[original_request_id] = migrated_request
            self.request_id_2_prompt_id[original_request_id] = prompt_id
            self.request_id_2_dp_rank[original_request_id] = target_dp_rank
            # The original_request_id should already be in self.prompt_id_2_request_ids, so no need to re-add.

            ray.get(self.actor_cluster.workers[target_dp_rank].add_request.remote(command=GenerateRequestType.ADD,
                                                                             data=migrated_request))
            self.load_balance_coordinator[target_dp_rank] += 1
            logger.info(
                f"Successfully resumed prompt {prompt_id} to dp rank {target_dp_rank} with original request id {original_request_id}"
            )

    def send_one_interrupted_query_group_to_dp(self, target_dp_rank: int):
        """
        Send interrupted query group to a target DP rank for continuation.
        This method resends interrupted requests with their original input + partial output
        concatenated as the new input, allowing the model to continue generation from
        where it was interrupted while preserving the original request_id and metadata.
        """
        # assert False, "reroute not working now"
        with self.lock:
            # 1. get the request outputs from the interrupted buffer for one prompt
            assert self.interrupted_query_group_buffers, "Migration: No interrupted query groups in buffer to migrate"

            prompt_id, interrupted_batches = self.interrupted_query_group_buffers.popitem()
            logger.info(
                f"Migration: Migrating prompt_id {prompt_id} with {len(interrupted_batches)} batches to DP rank {target_dp_rank}")

            old_dp_rank = None  # Initialize to track the original DP rank
            successfully_migrated = 0  # Count successfully migrated requests

            for i, processed_batch in enumerate(interrupted_batches):
                pprint(processed_batch)
                assert processed_batch.meta_info[
                           "finish_status"] == "interrupted", "interrupted_batches should be interrupted"
                # Keep the original request_id
                original_request_id = processed_batch.meta_info["request_id"]

                # Create migrated request with same request_id
                migrated_request = DataProto()

                # CRITICAL: For interrupted requests, we must build the continuation input correctly
                # by reconstructing from the original input + partial output tokens.

                # Get the original request to extract the original input_ids
                if original_request_id not in self.requests_buffers:
                    # Original request not found - skip this migration
                    logger.error(f"Migration: Original request not found for {original_request_id}, skipping migration")
                    assert False, f"Original request not found for {original_request_id}, skipping migration"


                original_request = self.requests_buffers[original_request_id]
                original_input_ids = original_request.batch["input_ids"]  # Original prompt tokens
                logger.info(
                    f"Migration: Found original request for {original_request_id}, input_shape={original_input_ids.shape}")

                # Get partial output tokens from the processed response
                partial_output_tokens = processed_batch.meta_info.get("output_token_ids", [])

                if partial_output_tokens and len(partial_output_tokens) > 0 and len(partial_output_tokens[0]) > 0:
                    # We have partial output, concatenate original input + partial output for continuation
                    logger.info(
                        f"Migration: Found partial output tokens for {original_request_id}: {len(partial_output_tokens[0])} tokens")

                    # Build continuation input: original_input + partial_output
                    partial_output_tensor = torch.tensor(partial_output_tokens[0], device=original_input_ids.device)
                    continued_input_ids = torch.cat([original_input_ids.squeeze(0), partial_output_tensor],
                                                    dim=0).unsqueeze(0)

                    # Build corresponding attention mask and position_ids
                    continued_seq_len = continued_input_ids.shape[1]
                    continued_attention_mask = torch.ones((1, continued_seq_len), device=continued_input_ids.device,
                                                          dtype=torch.long)
                    continued_position_ids = torch.arange(continued_seq_len,
                                                          device=continued_input_ids.device).unsqueeze(0)

                    logger.info(f"Migration: Built continuation input for {original_request_id}: "
                                f"original_len={original_input_ids.shape[1]}, partial_len={len(partial_output_tokens[0])}, "
                                f"continued_len={continued_seq_len}")
                    
                    # **CRITICAL FIX**: Store the original prompt length for correct prompt extraction
                    # This ensures postprocess_generate knows how to extract the original prompt correctly
                    original_prompt_length = original_input_ids.shape[1]
                    
                else:
                    # No partial output, just use original input
                    logger.warning(
                        f"Migration: No partial output found for interrupted request {original_request_id}, using original input")
                    continued_input_ids = original_input_ids

                    continued_attention_mask = original_request.batch["attention_mask"]
                    continued_position_ids = original_request.batch.get("position_ids",
                                                                        torch.arange(continued_input_ids.shape[1],
                                                                                     device=continued_input_ids.device).unsqueeze(
                                                                            0))
                    original_prompt_length = original_input_ids.shape[1]

                logger.info(f"Migration: Using continued input for request {original_request_id}: "
                            f"continued_input_shape={continued_input_ids.shape}, max_token_id={continued_input_ids.max().item()}")

                batch_tensors = {
                    "input_ids": continued_input_ids,  # Original input + partial output for continuation
                    "attention_mask": continued_attention_mask,
                    "position_ids": continued_position_ids,
                }

                # Copy other tensor fields from the original request
                source_batch = original_request.batch
                for key in source_batch.keys():
                    if key not in batch_tensors and key not in ["prompts"]:
                        batch_tensors[key] = source_batch[key]
                        logger.info(f"Migration: Copied tensor field '{key}' from original request.")

                # **CRITICAL FIX**: Create a special "prompts" tensor that contains ONLY the original prompt
                # This ensures consistent tensor shapes across normal and interrupted requests
                batch_tensors["prompts"] = original_input_ids
                logger.info(f"Migration: Set prompts tensor to original shape {original_input_ids.shape} for consistency")

                # Create TensorDict
                migrated_request.batch = TensorDict(source=batch_tensors, batch_size=(continued_input_ids.shape[0],))

                # **FIX: Use original request's non_tensor_batch instead of processed_batch**
                # The processed_batch might have incomplete or modified non_tensor_batch data
                migrated_request.non_tensor_batch = copy.deepcopy(original_request.non_tensor_batch)

                # Verify that the domain key exists in the original request
                if "domain" not in original_request.non_tensor_batch:
                    logger.error(f"Migration: 'domain' key missing in original request {original_request_id}")
                    logger.error(
                        f"Migration: Original request non_tensor_batch keys: {list(original_request.non_tensor_batch.keys())}")
                    assert False, f"'domain' key missing in original request {original_request_id}"

                logger.info(
                    f"Migration: Copied non_tensor_batch from original request. Keys: {list(migrated_request.non_tensor_batch.keys())}")

                # Keep original metadata but update callback and adjust generation config
                assert processed_batch.meta_info['finish_status'] == "interrupted"
                migrated_request.meta_info = processed_batch.meta_info.copy()
                migrated_request.meta_info.pop('finish_status', None)
                migrated_request.meta_info["response_callback_fn"] = self.response_callback_fn
                
                # **CRITICAL FIX**: Store metadata for postprocess_generate to handle correctly
                migrated_request.meta_info["is_continued_request"] = True
                migrated_request.meta_info["original_prompt_length"] = original_prompt_length
                migrated_request.meta_info["partial_output_length"] = len(partial_output_tokens[0]) if partial_output_tokens and len(partial_output_tokens) > 0 and len(partial_output_tokens[0]) > 0 else 0

                # Adjust max_new_tokens for continued generation
                generation_config = migrated_request.meta_info["generation_config"].copy()
                current_input_length = continued_input_ids.shape[1]
                max_sequence_length = self.pipeline_config.sequence_length
                safety_buffer = 4
                max_allowed_new_tokens = max_sequence_length - current_input_length - safety_buffer
                original_max_new_tokens = generation_config.get("max_new_tokens", 512)
                adjusted_max_new_tokens = max(1, min(original_max_new_tokens, max_allowed_new_tokens))
                generation_config["max_new_tokens"] = adjusted_max_new_tokens
                migrated_request.meta_info["generation_config"] = generation_config
                logger.info(f"Migration: max_new_tokens for request {original_request_id}: "
                            f"original={original_max_new_tokens}, adjusted={adjusted_max_new_tokens}")

                migrated_request.non_tensor_batch = processed_batch.non_tensor_batch.copy()
                assert "domain" in processed_batch.non_tensor_batch.keys(), f"{prompt_id=} {original_request_id=} processed_batch.non_tensor_batch keys: {list(processed_batch.non_tensor_batch.keys())} should contain domain"
                assert "domain" in migrated_request.non_tensor_batch.keys(), f"{prompt_id=} {original_request_id=} batch.non_tensor_batch keys: {list(migrated_request.non_tensor_batch.keys())} should contain domain"

                # Update bookkeeping data structures

                old_dp_rank = self.request_id_2_dp_rank[original_request_id]
                self.request_id_2_dp_rank[original_request_id] = target_dp_rank
                self.load_balance_coordinator[target_dp_rank] += 1
                self.requests_buffers[original_request_id] = migrated_request
                logger.info(f"Migration: Added migrated request {original_request_id} to DP rank {target_dp_rank}")
                # Send to new DP worker
                self.actor_cluster.workers[target_dp_rank].add_request.remote(
                    command=GenerateRequestType.ADD, data=migrated_request
                )

                migrated_request.meta_info.pop("response_callback_fn", None)
                successfully_migrated += 1

            if successfully_migrated == 0:
                logger.error(f"Migration: Failed to migrate any requests for prompt_id {prompt_id} - all were skipped.")
                assert False, "Failed to migrate any requests for prompt_id {prompt_id} - all were skipped."
                return 0

            logger.info(
                f"Migration: Successfully migrated {successfully_migrated} out of {len(interrupted_batches)} requests for prompt_id {prompt_id} from rank {old_dp_rank} to rank {target_dp_rank}")
            return successfully_migrated

    def interrupt_all_requests_by_dp_rank(self, interrupted_rank):
        # assert False, "Migration: interrupt_all_requests_by_dp_rank is not implemented yet"
        # return
        # 1. remove the interrupted rank from the active dp ranks
        logger.info(f"Migration: Removing DP rank {interrupted_rank} from ready ranks")
        # self.ready_dp_ranks.remove(interrupted_rank)

        # some might be interrupted, some might be aborted or interrupted_rank
        request_ids = self.get_running_request_ids_for_dp_rank(interrupted_rank)
        assert len(request_ids) > 0, "no requests are informed interruption"
        logger.info(
            f"Migration: inform interrupting {len(request_ids)} requests from DP rank {interrupted_rank}  request list: {request_ids} ")
        interrupt_refs = []

        for request_id in request_ids:
            # dp_rank = self.request_id_2_dp_rank[request_id]
            interrupt_refs.append(
                self.actor_cluster.workers[interrupted_rank].add_request.remote(
                    command=GenerateRequestType.INTERRUPT, data=DataProto(meta_info={"request_id": request_id})
                )
            )

    def get_running_request_ids_for_dp_rank(self, target_dp_rank: int) -> List[str]:
        """Get all request_ids currently assigned to a specific DP rank"""
        running_request_ids = []
        with self.lock:
            for request_id in self.requests_buffers.keys():
                if self.request_id_2_dp_rank[request_id] == target_dp_rank:
                    running_request_ids.append(request_id)

        return running_request_ids

    @ray.method(concurrency_group="multi_thread")
    def report_response(self, data: DataProto):
        """
        这里需要考虑多线程数据访问
        data 返回可能有多条的
        """

        import pydevd_pycharm


        try:
            logger.info(f"report_response: {data.meta_info['request_id']} {data.meta_info['finish_status']}")
            request_id = data.meta_info["request_id"]
            # if request_id == '5':
            # if False:
            #     pydevd_pycharm.settrace(
            #         'localhost',
            #         port=12332,
            #         stdoutToServer=True,
            #         stderrToServer=True,
            #         suspend=False,
            #         trace_only_current_thread=True
            #     )
            # else:
            #     # pydevd_pycharm.settrace(
            #     #     'localhost',
            #     #     port=9999,
            #     #     stdoutToServer=True,
            #     #     stderrToServer=True,
            #     #     suspend=False,
            #     #     trace_only_current_thread=True
            #     # )
            #
            #     while True:
            #         pass

            prompt_id = self.request_id_2_prompt_id[request_id]
            num_return_sequences = self.generation_config["num_return_sequences"]
            assert data.meta_info["finish_status"] in ["interrupted", 'finished']
            with self.lock:
                if data.meta_info["finish_status"] == "interrupted":
                    # **ENHANCED LOGGING**: Track prompt length and output lengths for interrupted requests
                    
                    # Get the original request to analyze lengths
                    original_request = self.requests_buffers.get(request_id, None)
                    original_prompt_length = 0
                    cumulative_partial_output_length = 0
                    migration_count = 0
                    
                    if original_request:
                        original_input_ids = original_request.batch["input_ids"]
                        concatenated_input_length = original_input_ids.shape[1]
                        
                        # Check if this is a continued request
                        is_continued_request = original_request.meta_info.get("is_continued_request", False)
                        if is_continued_request:
                            # For continued requests, get the actual original prompt length from metadata
                            original_prompt_length = original_request.meta_info.get("original_prompt_length", 1024)
                            cumulative_partial_output_length = original_request.meta_info.get("cumulative_partial_output_length", 0)
                            migration_count = original_request.meta_info.get("migration_count", 0)
                        else:
                            # For original requests, the input length is the original prompt length
                            original_prompt_length = concatenated_input_length
                    
                    # Get the newly generated output tokens from this interruption
                    output_token_ids = data.meta_info.get("output_token_ids", [])
                    newly_generated_length = 0
                    if output_token_ids and len(output_token_ids) > 0 and len(output_token_ids[0]) > 0:
                        newly_generated_length = len(output_token_ids[0])
                    
                    # Single comprehensive log entry
                    logger.info(f"Migration: BUFFERING interrupted request {request_id}: "
                               f"original_prompt_length={original_prompt_length}, "
                               f"cumulative_partial_output_length={cumulative_partial_output_length}, "
                               f"newly_generated_length={newly_generated_length}, "
                               f"migration_count={migration_count}")
                    
                    self.interrupted_query_group_buffers[prompt_id].append(data)
                    logger.info(
                        f"Migration: Added interrupted batch for prompt_id {prompt_id} to buffer {list(self.interrupted_query_group_buffers.keys())}")
                    # assert False, "can interrupt and buffer the response, but not complete it yet"
                    self.load_balance_coordinator[self.request_id_2_dp_rank[request_id]] -= 1
                    return

            assert data.meta_info["finish_status"] ==  "finished"

            # with lock
            batch = self.postprocess_output_ids(data)
            output_count = batch.batch.batch_size[0]

            with self.lock:
                self.load_balance_coordinator[self.request_id_2_dp_rank[request_id]] -= 1
                self.prompt_id_2_request_ids[prompt_id].remove(request_id)
                domain = "default"
                assert "domain" in batch.non_tensor_batch.keys(), f"{prompt_id=} {request_id=} batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())} should contain domain"

                if "domain" in batch.non_tensor_batch.keys():
                    domain = batch.non_tensor_batch["domain"][0]

                logger.info(
                    f"{request_id=} batch.non_tensor_batch: {list(batch.non_tensor_batch.keys())} self.reward_worker_iters {list(self.reward_worker_iters.keys())}")

                if domain == "default":
                    import pydevd_pycharm
                    import os
                    if os.getenv("PYCHARM", "0") == "1":
                        pydevd_pycharm.settrace('localhost', port=12332, stdoutToServer=True, stderrToServer=True,
                                            suspend=False)
                    assert False, f"batch.non_tensor_batch : {list(batch.non_tensor_batch.keys())} self.reward_worker_iters {list(self.reward_worker_iters.keys())}"

                reward_worker = next(self.reward_worker_iters[domain])

            if not self.running:
                return

            # call reward
            # reward worker得能支持单条数据计算, dynamic sampling对需要batch计算reward的需要注意...
            # 多域的时候,llm as judge, 需要单独为reward worker分配gpu
            rewards: DataProto = ray.get(reward_worker.compute_rewards.remote(batch))
            batch.union(rewards)

            response_buffers: List[DataProto] = []
            batch_expanded = [batch[[idx]] for idx in range(output_count)]

            # response_filter, 不太需要response filter
            for batch_item in batch_expanded:
                if self.response_filter_fn(batch_item, self.pipeline_config):
                    response_buffers.append(batch_item)
                else:
                    self.response_filter_count += 1

            with self.lock:
                self.response_cache[domain].extend(batch_expanded)

                if len(response_buffers) == 0:
                    if len(self.prompt_id_2_request_ids[prompt_id]) == 0:
                        self.running_prompts -= 1
                    return

                if len(self.completed_buffers[prompt_id]) > 0:
                    return

                # expand batch to response
                self.query_group_buffers[prompt_id].extend(response_buffers)

                # query_filter, query has n responses
                if len(self.query_group_buffers[prompt_id]) >= num_return_sequences:
                    if not self.query_filter_fn(self.query_group_buffers[prompt_id], self.pipeline_config):
                        self.query_filter_count += 1
                        del self.query_group_buffers[prompt_id]
                        self.abort_requests(self.prompt_id_2_request_ids[prompt_id])
                        return

                    assert len(self.query_group_buffers[prompt_id]) >= num_return_sequences, (
                        f"expect to generate {num_return_sequences} results from one prompt, "
                        f"but get {len(self.query_group_buffers[prompt_id])}."
                    )

                    self.completed_buffers[prompt_id] = self.query_group_buffers[prompt_id][:num_return_sequences]
                    self.progress_bar.update()

                    # abort uncompleted request
                    self.abort_requests(self.prompt_id_2_request_ids[prompt_id])
        except Exception as e:
            self.exception_queue.put(e)

        pydevd_pycharm.stoptrace()

    def get_fixed_dataset_item(self, dataset_index=0):
        """Fixed dataset item for testing - always returns the same item"""
        dataset_item = self.dataset[dataset_index]
        logger.info(f"FIXED_DATASET_DEBUG: Using fixed dataset item at index {dataset_index}")
        
        # Log the fixed dataset item details
        for key in ['prompt', 'text', 'messages', 'ground_truth', 'input_ids']:
            if key in dataset_item:

                data = dataset_item[key]
                if key == 'input_ids':
                    logger.info(f"FIXED_DATASET_DEBUG: {key}_len={len(data)}, first_10={data[:10]}")
                else:
                    sample_text = str(data)[:100] if data else "None"
                    logger.info(f"FIXED_DATASET_DEBUG: {key}_sample='{sample_text}'")
        
        return dataset_item

    def get_next_dataset_item(self):
        if self.dataset_iter is None:
            random.seed(self.pipeline_config.seed + self.dataset_epoch)
            random.shuffle(self.indices)
            self.dataset_iter = iter(self.indices)
            logger.info(f"{'-'.join(self.reward_clusters.keys())} dataset epoch: {self.dataset_epoch}")

        try:
            item_index = next(self.dataset_iter)
            logger.info(f"Dataset length: {len(self.dataset)}, retrieving get_next_dataset_item at index: {item_index}")
            dataset_item = self.dataset[item_index]
            
            # dataset_item = self.dataset[next(self.dataset_iter)]
            # tao: rvst hardcode for testing debug interrupt, remove this later
            # dataset_item = self.dataset[0]
        except StopIteration:
            self.dataset_epoch += 1
            random.seed(self.pipeline_config.seed + self.dataset_epoch)
            random.shuffle(self.indices)
            self.dataset_iter = iter(self.indices)
            dataset_item = self.dataset[next(self.dataset_iter)]
            logger.info(f"{'-'.join(self.reward_clusters.keys())} dataset epoch: {self.dataset_epoch}")
        self.dataset_iter_count += 1
        return dataset_item

    def get_scheduler_state(self):
        return {"dataset_iter_count": self.dataset_iter_count}

    def abort_requests(self, request_ids: Set[str]):
        abort_refs = []
        self.running_prompts -= 1
        for request_id in request_ids:
            dp_rank = self.request_id_2_dp_rank[request_id]
            self.load_balance_coordinator[dp_rank] -= 1
            abort_refs.append(
                self.actor_cluster.workers[dp_rank].add_request.remote(
                    command=GenerateRequestType.ABORT, data=DataProto(meta_info={"request_id": request_id})
                )
            )

    def postprocess_output_ids(self, data: DataProto) -> DataProto:
        # postprocess_generate, input_ids, attention_mask, left pad
        request_id = data.meta_info["request_id"]
        logger.info(f"postprocess_output_ids: {request_id=}")
        with self.lock:
            request: DataProto = self.requests_buffers.pop(request_id)
            self.postprocessed_requests_count +=1
        eos_token_id = data.meta_info["eos_token_id"]
        pad_token_id = data.meta_info["pad_token_id"]
        output_token_ids = data.meta_info["output_token_ids"]
        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]
        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=request.batch["input_ids"], output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        output: DataProto = postprocess_generate(
            prompts=request,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=self.pipeline_config.sequence_length,
            canonical_prompt_length=self.pipeline_config.prompt_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        request_repeat = request.repeat(repeat_times=len(output_tokens))
        output.non_tensor_batch = request_repeat.non_tensor_batch
        output.meta_info = request_repeat.meta_info
        return output

    def expand_requests(self, data: DataProto):
        """
        replica, 以及redundancy
        """
        generate_opt_level = self.pipeline_config.generate_opt_level
        is_num_return_sequences_expand = self.pipeline_config.is_num_return_sequences_expand
        num_return_sequences = self.generation_config["num_return_sequences"]

        assert generate_opt_level > 0, (
            f"generate_opt_level {generate_opt_level} should > 0, " f"in dynamic sampling scheduler."
        )
        assert "generation_config" in data.meta_info, f"data {data.meta_info} should have key 'generation_config'"
        generation_config = data.meta_info["generation_config"]

        target_requests = []
        if is_num_return_sequences_expand:
            generation_config["num_return_sequences"] = 1
            for _ in range(num_return_sequences):
                target_requests.append(copy.deepcopy(data))
        else:
            generation_config["num_return_sequences"] = num_return_sequences
            target_requests.append(copy.deepcopy(data))

        return target_requests

    def check_worker_alive(self, cluster):
        # 探测dp worker是否存活，dp worker的server thread可能由于异常退出，造成hang
        current_time = time.time()
        if current_time - self.last_alive_check >= self.alive_check_interval:
            cluster.add_request(command=GenerateRequestType.ALIVE_CHECK, data=DataProto())
            self.last_alive_check = current_time

    def check_response_callback(self):
        if self.exception_queue.qsize() > 0:
            e = self.exception_queue.get()
            logger.error(f"report_response get exception {e}")
            raise e

    def check_send_new_request(self) -> bool:
        if self.running_prompts >= (self.batch_size + self.max_additional_running_prompts):
            return False
        if not self.is_use_additional_prompts and self.prompt_use_count >= self.batch_size:
            return False
        return True

    def get_available_dp_rank(self):
        while True:
            # 负载均衡逻辑，期望各dp 正在处理的条数基本接近
            sorted_ranks = sorted(
                self.load_balance_coordinator.keys(), key=lambda rank: (self.load_balance_coordinator[rank], rank)
            )
            if self.load_balance_coordinator[sorted_ranks[0]] < self.max_running_requests:
                yield sorted_ranks[0]


@ray.remote
class GlobalCounter:
    def __init__(self):
        self.value = -1

    def get_value(self):
        self.value += 1
        return self.value


@ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 2048})
class RequestScheduler:
    def __init__(self, infer_cluster, pipeline_config):
        self.infer_cluster = infer_cluster
        self.pipeline_config = pipeline_config
        self.request_dict = ThreadSafeDict()
        self.request_id_2_dp_rank = {}
        self.src_rank2_dp_rank = {}
        self.worker_iter = itertools.cycle(range(self.infer_cluster.world_size))

    @ray.method(concurrency_group="multi_thread")
    def generate_one_request(self, data: DataProto):
        assert "request_id" in data.meta_info, f"data {data.meta_info} should have key 'request_id'"

        request_id = data.meta_info["request_id"]
        src_rank = data.meta_info["src_rank"]
        if src_rank not in self.src_rank2_dp_rank:
            dp_rank = next(self.worker_iter)
            self.src_rank2_dp_rank[src_rank] = dp_rank

        dp_rank = self.src_rank2_dp_rank[src_rank]
        # send request to one worker
        ray.get(self.infer_cluster.workers[dp_rank].add_request.remote(command=GenerateRequestType.ADD, data=data))
        data.meta_info.pop("response_callback_fn")
        self.request_id_2_dp_rank[request_id] = dp_rank

        response_data: DataProto = self.request_dict.pop(data.meta_info["request_id"])
        self.request_id_2_dp_rank.pop(data.meta_info["request_id"])
        if response_data is None:
            # request aborted
            return None

        # postprocess_generate, input_ids, attention_mask, left pad
        eos_token_id = response_data.meta_info["eos_token_id"]
        pad_token_id = response_data.meta_info["pad_token_id"]
        output_token_ids = response_data.meta_info["output_token_ids"]
        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]
        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=data.batch["input_ids"], output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        output: DataProto = postprocess_generate(
            prompts=data,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=output_tensor.shape[-1],
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        request_repeat = data.repeat(repeat_times=len(output_tokens))
        output.non_tensor_batch = request_repeat.non_tensor_batch
        output.meta_info = request_repeat.meta_info
        return output

    @ray.method(concurrency_group="multi_thread")
    def report_response(self, data: DataProto):
        """
        这里需要考虑多线程数据访问
        data 返回可能有多条的
        """

        import pydevd_pycharm


        try:
            logger.info(f"report_response: {data.meta_info['request_id']} {data.meta_info['finish_status']}")
            request_id = data.meta_info["request_id"]
            # if request_id == '5':
            # if False:
            #     pydevd_pycharm.settrace(
            #         'localhost',
            #         port=12332,
            #         stdoutToServer=True,
            #         stderrToServer=True,
            #         suspend=False,
            #         trace_only_current_thread=True
            #     )
            # else:
            #     # pydevd_pycharm.settrace(
            #     #     'localhost',
            #     #     port=9999,
            #     #     stdoutToServer=True,
            #     #     stderrToServer=True,
            #     #     suspend=False,
            #     #     trace_only_current_thread=True
            #     # )
            #
            #     while True:
            #         pass

            prompt_id = self.request_id_2_prompt_id[request_id]
            num_return_sequences = self.generation_config["num_return_sequences"]
            assert data.meta_info["finish_status"] in ["interrupted", 'finished']
            with self.lock:
                if data.meta_info["finish_status"] == "interrupted":
                    # **ENHANCED LOGGING**: Track prompt length and output lengths for interrupted requests
                    
                    # Get the original request to analyze lengths
                    original_request = self.requests_buffers.get(request_id, None)
                    original_prompt_length = 0
                    cumulative_partial_output_length = 0
                    migration_count = 0
                    
                    if original_request:
                        original_input_ids = original_request.batch["input_ids"]
                        concatenated_input_length = original_input_ids.shape[1]
                        
                        # Check if this is a continued request
                        is_continued_request = original_request.meta_info.get("is_continued_request", False)
                        if is_continued_request:
                            # For continued requests, get the actual original prompt length from metadata
                            original_prompt_length = original_request.meta_info.get("original_prompt_length", 1024)
                            cumulative_partial_output_length = original_request.meta_info.get("cumulative_partial_output_length", 0)
                            migration_count = original_request.meta_info.get("migration_count", 0)
                        else:
                            # For original requests, the input length is the original prompt length
                            original_prompt_length = concatenated_input_length
                    
                    # Get the newly generated output tokens from this interruption
                    output_token_ids = data.meta_info.get("output_token_ids", [])
                    newly_generated_length = 0
                    if output_token_ids and len(output_token_ids) > 0 and len(output_token_ids[0]) > 0:
                        newly_generated_length = len(output_token_ids[0])
                    
                    # Single comprehensive log entry
                    logger.info(f"Migration: BUFFERING interrupted request {request_id}: "
                               f"original_prompt_length={original_prompt_length}, "
                               f"cumulative_partial_output_length={cumulative_partial_output_length}, "
                               f"newly_generated_length={newly_generated_length}, "
                               f"migration_count={migration_count}")
                    
                    self.interrupted_query_group_buffers[prompt_id].append(data)
                    logger.info(
                        f"Migration: Added interrupted batch for prompt_id {prompt_id} to buffer {list(self.interrupted_query_group_buffers.keys())}")
                    # assert False, "can interrupt and buffer the response, but not complete it yet"
                    self.load_balance_coordinator[self.request_id_2_dp_rank[request_id]] -= 1
                    return

            assert data.meta_info["finish_status"] ==  "finished"

            # with lock
            batch = self.postprocess_output_ids(data)
            output_count = batch.batch.batch_size[0]

            with self.lock:
                self.load_balance_coordinator[self.request_id_2_dp_rank[request_id]] -= 1
                self.prompt_id_2_request_ids[prompt_id].remove(request_id)
                domain = "default"
                assert "domain" in batch.non_tensor_batch.keys(), f"{prompt_id=} {request_id=} batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())} should contain domain"

                if "domain" in batch.non_tensor_batch.keys():
                    domain = batch.non_tensor_batch["domain"][0]

                logger.info(
                    f"{request_id=} batch.non_tensor_batch: {list(batch.non_tensor_batch.keys())} self.reward_worker_iters {list(self.reward_worker_iters.keys())}")

                if domain == "default":
                    import pydevd_pycharm
                    import os
                    if os.getenv("PYCHARM", "0") == "1":
                        pydevd_pycharm.settrace('localhost', port=12332, stdoutToServer=True, stderrToServer=True,
                                            suspend=False)
                    assert False, f"batch.non_tensor_batch : {list(batch.non_tensor_batch.keys())} self.reward_worker_iters {list(self.reward_worker_iters.keys())}"

                reward_worker = next(self.reward_worker_iters[domain])

            if not self.running:
                return

            # call reward
            # reward worker得能支持单条数据计算, dynamic sampling对需要batch计算reward的需要注意...
            # 多域的时候,llm as judge, 需要单独为reward worker分配gpu
            rewards: DataProto = ray.get(reward_worker.compute_rewards.remote(batch))
            batch.union(rewards)

            response_buffers: List[DataProto] = []
            batch_expanded = [batch[[idx]] for idx in range(output_count)]

            # response_filter, 不太需要response filter
            for batch_item in batch_expanded:
                if self.response_filter_fn(batch_item, self.pipeline_config):
                    response_buffers.append(batch_item)
                else:
                    self.response_filter_count += 1

            with self.lock:
                self.response_cache[domain].extend(batch_expanded)

                if len(response_buffers) == 0:
                    if len(self.prompt_id_2_request_ids[prompt_id]) == 0:
                        self.running_prompts -= 1
                    return

                if len(self.completed_buffers[prompt_id]) > 0:
                    return

                # expand batch to response
                self.query_group_buffers[prompt_id].extend(response_buffers)

                # query_filter, query has n responses
                if len(self.query_group_buffers[prompt_id]) >= num_return_sequences:
                    if not self.query_filter_fn(self.query_group_buffers[prompt_id], self.pipeline_config):
                        self.query_filter_count += 1
                        del self.query_group_buffers[prompt_id]
                        self.abort_requests(self.prompt_id_2_request_ids[prompt_id])
                        return

                    assert len(self.query_group_buffers[prompt_id]) >= num_return_sequences, (
                        f"expect to generate {num_return_sequences} results from one prompt, "
                        f"but get {len(self.query_group_buffers[prompt_id])}."
                    )

                    self.completed_buffers[prompt_id] = self.query_group_buffers[prompt_id][:num_return_sequences]
                    self.progress_bar.update()

                    # abort uncompleted request
                    self.abort_requests(self.prompt_id_2_request_ids[prompt_id])
        except Exception as e:
            self.exception_queue.put(e)

        pydevd_pycharm.stoptrace()
