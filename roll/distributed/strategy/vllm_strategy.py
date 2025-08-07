import copy
import gc
import itertools
import os
import queue
from concurrent import futures
from typing import List, Optional, Union, Dict
import asyncio

import ray
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
from mcore_adapter.models.converter.convert_utils import RecvBucketManager
from vllm import SamplingParams, RequestOutput
from vllm.utils import random_uuid

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.vllm import LLM
from roll.third_party.vllm import AsyncLLM
from roll.utils.collective import collective
from roll.utils.functionals import concatenate_input_and_output, GenerateRequestType
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType
import threading

logger = get_logger()


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model: Union[LLM, AsyncLLM]
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.pending_size = 1
        self.recv_manager = RecvBucketManager()
        self.command_queue: Optional[queue.Queue] = None

        self.request_metas = {} # used to keep track of requests to callback
        self.group_name = "vllm_worker_default"
        self.running = False

        self.interrupted_rid_set = set()
        self.lock = threading.Lock()
        self.count_calls = {}



    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        engine_mode = vllm_config.pop("engine_mode", "sync")  # sync/async
        self.pending_size = vllm_config.pop("pending_size", 1)
        self.sleep_level = vllm_config.pop("sleep_level", 1)
        self.command_queue = queue.Queue()

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"
        vllm_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "enforce_eager": vllm_config.get("enforce_eager", False),
                "trust_remote_code": True,
                "seed": self.worker.pipeline_config.seed,
                "disable_custom_all_reduce": vllm_config.get(
                    "disable_custom_all_reduce", True
                ),  # potentially hangs in tp>1
                "enable_prefix_caching": vllm_config.get("enable_prefix_caching", False),
                "load_format": vllm_config.get("load_format", "dummy"),  # use model update passed value
            }
        )
        logger.info(f"vllm_config: {vllm_config}")
        assert not dist.is_initialized()

        # set VLLM_PORT to avoid port conflict applied by vllm
        vllm_port = self.worker.get_free_port()
        os.environ["VLLM_PORT"] = str(vllm_port)

        if engine_mode == "sync":
            self.model = LLM(resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config)
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.model = AsyncLLM(
                resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config
            )
            loop = asyncio.get_event_loop()
            self.tokenizer = loop.run_until_complete(self.model.get_tokenizer())
        additional_special_tokens = self.tokenizer.additional_special_tokens
        special_tokens = [
            add_token
            for add_token in self.tokenizer.added_tokens_decoder.values()
            if add_token.special and add_token.content not in additional_special_tokens
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}, replace_additional_special_tokens=False
        )
        logger.info(f"add {special_tokens} to additional_special_tokens: {self.tokenizer.additional_special_tokens}")

        self.worker.rank_info.dp_rank = self.worker.rank
        self.worker.rank_info.dp_size = self.worker.world_size
        collective.init_collective_group(
            world_size=self.worker.world_size,
            rank=self.worker.rank,
            group_name=self.group_name,
            master_addr=self.worker.master_addr,
            master_port=self.worker.master_port,
        )

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        # **ASSERTION**: Multi-modal data should be empty
        assert "multi_modal_data" not in batch.non_tensor_batch, f"multi_modal_data should be empty but found in batch"
        
        vllm_input_args = {
            "prompt_token_ids": gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
        }

        vllm_outputs = self.model.generate(sampling_params=sampling_params, use_tqdm=False, **vllm_input_args)

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=vllm_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )

        return output

    def process_interrupted_batch(self, request_id: str, request_complete_callback):
        # added req but not started running/waiting
        # key should exist
        assert request_id in self.request_metas, 'key should exist in request_metas'

        output_data = DataProto(meta_info=self.request_metas[request_id])
        output_data.meta_info["finish_status"] = "interrupted"
        output_data.meta_info["output_token_ids"] = []  # No output tokens for interrupted batch
        logger.info(f"process_interrupted_batch: request_id {output_data.meta_info['request_id']}")
        request_complete_callback(data=output_data)

    def handle_vllm_output(self, finished_vllm_outputs: List[RequestOutput],  interrupted_rid_set, request_complete_callback):
        finished_req_ids = [ ]
        to_callback_output = []
        # handle finshed first, then interrupted
        for request_output in finished_vllm_outputs:
            assert request_output.finished, "should be finished req"
            self.unfinished_vllm_outputs.pop(request_output.request_id, None)

            # still in request_metas not aborted, process the request output
            if request_output.request_id in self.request_metas:

                finished_req_ids.append(request_output.request_id)
                to_callback_output.append(request_output)



        if to_callback_output:
            logger.info(
            f"process_vllm_output: finished request from fetch_output and in request_metas,  request_ids {finished_req_ids} calling callback")
            # this assumes req is in self.request_metas
            self.process_vllm_output(vllm_outputs=to_callback_output,
                                 request_complete_callback=request_complete_callback)

        # pop the finished request metas after callback
        for request_id in finished_req_ids:
            self.request_metas.pop(request_id)

        for req_id in interrupted_rid_set:

            if req_id in self.request_metas:

                if req_id in self.unfinished_vllm_outputs:
                    logger.info(f'handle_vllm_output: interrupted request_id {req_id} has partial output')
                    # 🔥 ADD DETAILED PARTIAL DECODE LOGGING HERE 🔥
                    partial_request_output = self.unfinished_vllm_outputs[req_id]
                    for i, output in enumerate(partial_request_output.outputs):
                        partial_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                        logger.info(f"INTERRUPTED_PARTIAL: request_id={req_id}, \n"
                                   f"output_{i}_tokens={len(output.token_ids)}, \n"
                                   f"partial_text='{partial_text}', \n"
                                   f"finished={partial_request_output.finished}\n")
                    
 
                    self.process_vllm_output(vllm_outputs=[self.unfinished_vllm_outputs[req_id]],
                                             request_complete_callback=request_complete_callback)
                    # self.unfinished_vllm_outputs.pop(req_id)

                else:
                    logger.info(f"handle_vllm_output: interrupted request_id {req_id} no partial output yet, just process the from added_batch")
                    self.process_interrupted_batch(req_id, request_complete_callback)

                self.request_metas.pop(req_id)

            else:
                logger.warning(f"handle_vllm_output: interrupted request_id {req_id} not found in added_batch, skipping perhaps already finished or aborted")

        interrupted_rid_set.clear()



    def process_vllm_output(self, vllm_outputs: List[RequestOutput], request_complete_callback):
        # 转成response id, request_complete_callback
        for request_output in vllm_outputs:
            output_token_ids = []
            request_id = request_output.request_id
            if request_id not in self.request_metas:
                logger.warning(f"process_vllm_output: request_id {request_id} not in request_metas, skipping")
                continue
            for completion_output in request_output.outputs:
                output_token_ids.append(completion_output.token_ids)
            output_data = DataProto(meta_info=self.request_metas[request_id])
            output_data.meta_info["output_token_ids"] = output_token_ids

            if request_output.finished:
                output_data.meta_info["finish_status"] = "finished"
                # not interrupted yet, otherwise will be remove from added batch
                # if request_id in self.added_batch:
                #     logger.info(f"process_vllm_output: finished request_id {request_id}")

            else:
                output_data.meta_info["finish_status"] = "interrupted"

            logger.info(
                f"VLLM RAW OUTPUT: request_id={request_output.request_id}, "
                f"output_token_ids={request_output.outputs[0].token_ids}"
            )

            request_complete_callback(data=output_data)

    def start_server(self, data: DataProto, request_complete_callback):
        collective.barrier(group_name=self.group_name)
        self.running = True
        self.unfinished_vllm_outputs = {}
        interrupted_rid_set = set()
        while True:
            while not self.command_queue.empty():
                command, batch = self.command_queue.get_nowait()
                if command == GenerateRequestType.ADD:
                    input_ids = batch.batch["input_ids"]
                    attention_mask = batch.batch["attention_mask"]
                    request_id = batch.meta_info["request_id"]
                    
                    # Debug: Log raw tensor info
                    logger.info(f"RAW_TENSOR_DEBUG: request_id={request_id}, input_ids_id={id(input_ids)}, input_ids_device={input_ids.device}, input_ids_dtype={input_ids.dtype}")
                    logger.info(f"RAW_TENSOR_DEBUG: request_id={request_id}, input_ids_shape={input_ids.shape}, input_ids_first_10={input_ids[0][:10].tolist()}")
                    
                    # Debug: Check if the tensor is a view/shared memory
                    logger.info(f"RAW_TENSOR_DEBUG: request_id={request_id}, input_ids_is_contiguous={input_ids.is_contiguous()}, input_ids_stride={input_ids.stride()}")
                    
                    self.request_metas[request_id] = batch.meta_info
                    generation_config = batch.meta_info.get("generation_config")
                    max_new_tokens = batch.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
                    max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
                    sampling_params = create_sampling_params_for_vllm(
                        gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens}
                    )
                    
                    # Debug: Check if there's any text in non_tensor_batch
                    if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch:
                        logger.info(f"RAW_TENSOR_DEBUG: request_id={request_id}, non_tensor_batch_keys={list(batch.non_tensor_batch.keys())}")
                        
                        # Check if there's any text we can compare
                        for key in ['prompt', 'text', 'messages', 'ground_truth']:
                            if key in batch.non_tensor_batch:
                                text_data = batch.non_tensor_batch[key]
                                if hasattr(text_data, '__len__') and len(text_data) > 0:
                                    sample_text = str(text_data[0])[:100] if text_data[0] else "None"
                                    logger.info(f"RAW_TENSOR_DEBUG: request_id={request_id}, {key}_sample='{sample_text}'")
                    
                    # **ASSERTION**: Multi-modal data should be empty
                    assert "multi_modal_data" not in batch.non_tensor_batch, f"request_id={request_id}: multi_modal_data should be empty but found in batch"
                    multi_modal_data = None
                    
                    # Check if this is a continuation request (interrupted request being resumed)
                    is_continued_request = batch.meta_info.get("is_continued_request", False)
                    continuation_mode = batch.meta_info.get("continuation_mode", False)
                    
                    if continuation_mode and "partial_output_tokens" in batch.meta_info:
                        logger.info(f"PROCESSING_PATH: request_id={request_id}, using CONTINUATION mode")
                        
                        # **ASSERTIONS for continued/interrupted requests**
                        assert is_continued_request, f"request_id={request_id}: continuation_mode=True but is_continued_request=False"
                        assert "original_prompt_length" in batch.meta_info, f"request_id={request_id}: continuation_mode requires original_prompt_length in meta_info"
                        assert "cumulative_partial_output_length" in batch.meta_info, f"request_id={request_id}: continuation_mode requires cumulative_partial_output_length in meta_info"
                        assert "migration_count" in batch.meta_info, f"request_id={request_id}: continuation_mode requires migration_count in meta_info"
                        
                        # Handle continuation mode - concatenate original prompt + partial output
                        partial_output_tokens = batch.meta_info["partial_output_tokens"]
                        
                        # **ATTENTION MASK DEBUG**: Log attention mask details before gather_unpadded_input_ids
                        logger.info(f"ATTENTION_MASK_DEBUG: request_id={request_id}, input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}")
                        logger.info(f"ATTENTION_MASK_DEBUG: request_id={request_id}, attention_mask.sum()={attention_mask.sum().item()}, attention_mask={attention_mask.tolist()}")
                        
                        logger.info(f"Continuation mode for request {request_id}: before gather_unpadded_input_ids: request_id={request_id}, input_ids.shape={input_ids.shape}, input_ids_first_10={input_ids[0][:10].tolist()}")
                        original_prompt_tokens = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
                        logger.info(f"Continuation mode for request {request_id}: after gather_unpadded_input_ids: request_id={request_id},  len(original_prompt_tokens[0])={ len(original_prompt_tokens[0])} original_prompt_tokens_first_10={original_prompt_tokens[0][:10] if original_prompt_tokens and len(original_prompt_tokens[0]) > 0 else 'empty'}")
                        
                        # **CRITICAL ASSERTION**: Verify gathered tokens match attention mask
                        assert len(original_prompt_tokens[0]) == attention_mask.sum().item(), f"request_id={request_id}: gathered tokens length {len(original_prompt_tokens[0])} != attention_mask sum {attention_mask.sum().item()}"
                        
                        # **ASSERTION**: Verify the input_ids contains the expected continued input
                        original_prompt_length = batch.meta_info["original_prompt_length"]
                        cumulative_partial_output_length = batch.meta_info["cumulative_partial_output_length"]
                        expected_input_length = original_prompt_length + cumulative_partial_output_length
                        actual_input_length = len(original_prompt_tokens[0])
                        
                        # **CRITICAL LENGTH VERIFICATION**
                        logger.info(f"LENGTH_VERIFICATION: request_id={request_id}, original_prompt_length={original_prompt_length}, cumulative_partial_output_length={cumulative_partial_output_length}, expected_input_length={expected_input_length}, actual_input_length={actual_input_length}")
                        logger.info(f"LENGTH_VERIFICATION: request_id={request_id}, attention_mask.sum()={attention_mask.sum().item()}, input_ids.shape[1]={input_ids.shape[1]}")
                        
                        # **ASSERTION**: Verify the attention mask fix worked correctly
                        assert input_ids.shape[1] == attention_mask.shape[1], f"request_id={request_id}: input_ids length {input_ids.shape[1]} != attention_mask length {attention_mask.shape[1]}"
                        assert actual_input_length <= input_ids.shape[1], f"request_id={request_id}: gathered length {actual_input_length} > input_ids length {input_ids.shape[1]}"
                        
                        # **FIXED ASSERTION**: Now that we track effective lengths, this should match
                        assert actual_input_length == expected_input_length, f"request_id={request_id}: effective input_ids length mismatch. Expected {expected_input_length} (original_effective={original_prompt_length} + cumulative_partial={cumulative_partial_output_length}), got {actual_input_length}. Note: input_ids.shape[1]={input_ids.shape[1]}, attention_mask.sum()={attention_mask.sum().item()}"
                        
                        # **CRITICAL FIX**: Don't concatenate again! The input_ids already contains the continued input
                        # The scheduler has already concatenated: original_prompt + all_previous_partial_outputs
                        # We just need to use the input_ids as-is, not concatenate partial_output_tokens again
                        prompt_token_ids = original_prompt_tokens
                        
                        logger.info(f"Continuation mode for request {request_id}: "
                                   f"original_prompt_length={original_prompt_length}, "
                                   f"cumulative_partial_output_length={cumulative_partial_output_length}, "
                                   f"current_partial_output_length={len(partial_output_tokens)}, "
                                   f"input_ids_length={actual_input_length}")
                    else:
                        # Normal mode
                        logger.info(f"PROCESSING_PATH: request_id={request_id}, using NORMAL mode")
                        
                        # **ATTENTION MASK DEBUG**: Log attention mask details for normal mode too
                        logger.info(f"ATTENTION_MASK_DEBUG: request_id={request_id}, input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}")
                        logger.info(f"ATTENTION_MASK_DEBUG: request_id={request_id}, attention_mask.sum()={attention_mask.sum().item()}, attention_mask={attention_mask.tolist()}")
                        
                        logger.info(f"Before gather_unpadded_input_ids: request_id={request_id}, input_ids.shape={input_ids.shape}, input_ids_first_10={input_ids[0][:10].tolist()}")
                        prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
                        logger.info(f"After gather_unpadded_input_ids: request_id={request_id},  len(prompt_token_ids[0])={ len(prompt_token_ids[0])} prompt_token_ids_first_10={prompt_token_ids[0][:10] if prompt_token_ids and len(prompt_token_ids[0]) > 0 else 'empty'}")
                        
                        # **ASSERTION**: Verify gathered tokens match attention mask for normal mode
                        assert len(prompt_token_ids[0]) == attention_mask.sum().item(), f"request_id={request_id}: gathered tokens length {len(prompt_token_ids[0])} != attention_mask sum {attention_mask.sum().item()}"
                    
                    # Debug logging - all in one line
                    decoded_prompt = self.tokenizer.decode(prompt_token_ids[0], skip_special_tokens=True) if prompt_token_ids else ""
                    
                    # Debug: Compare original input_ids vs processed prompt_token_ids
                    if prompt_token_ids and len(prompt_token_ids[0]) > 0:
                        original_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        processed_decoded = self.tokenizer.decode(prompt_token_ids[0], skip_special_tokens=True)
                        
                        # Check if they're different
                        if original_decoded != processed_decoded:
                            logger.info(f"DECODE_MISMATCH: request_id={request_id}")
                            logger.info(f"DECODE_MISMATCH: original_decoded='{original_decoded[:200]}'")
                            logger.info(f"DECODE_MISMATCH: processed_decoded='{processed_decoded[:200]}'")
                            logger.info(f"DECODE_MISMATCH: original_input_ids={input_ids[0].tolist()}")
                            logger.info(f"DECODE_MISMATCH: processed_prompt_token_ids={prompt_token_ids[0]}")
                        else:
                            logger.info(f"DECODE_MATCH: request_id={request_id}, decoded prompts are identical")
                    
                    logger.info(f"ADD_REQUEST: request_id={request_id} | input_ids.shape={input_ids.shape} | attention_mask.shape={attention_mask.shape} | "
                               f"input_ids={input_ids.tolist()} | attention_mask sum ={sum(attention_mask)} | "
                               f"processed_prompt_token_ids={prompt_token_ids} | \ndecoded_prompt='{decoded_prompt}'")
                    
                    self.model.add_requests(request_ids=[request_id],
                                            prompt_token_ids=prompt_token_ids,
                                            sampling_params=sampling_params,
                                            multi_modal_data=multi_modal_data)
                    logger.info(f"request {request_id} added")

                elif command == GenerateRequestType.ABORT:
                    request_id = batch.meta_info["request_id"]
                    assert request_id in self.request_metas
                    logger.info(f"{request_id=} abort command sent to backend engine, remove from request_metas")
                    self.model.abort_request(request_id=request_id)
                    self.request_metas.pop(request_id)

                elif command == GenerateRequestType.INTERRUPT:
                    request_id = batch.meta_info.get("request_id", None)
                    target_leftover_cnt = batch.meta_info.get("target_leftover_cnt", None)
                    assert (request_id is None) ^ (target_leftover_cnt is None), f"they are exclusive but got {request_id=} {target_leftover_cnt=}"
                    if request_id:
                        assert request_id in self.request_metas, f"request_id {request_id} not in request_metas {self.request_metas.keys()}"
                        logger.info(f"interrupt request command sent to backend engine")
                        self.model.abort_request(request_id=request_id)
                        interrupted_rid_set.add(request_id)


                    if target_leftover_cnt:
                        # Check if we have the v1 engine
                        if hasattr(self.model.llm_engine, 'engine_core'):
                            # Use v1 engine's collective_rpc to call abort_to_target_requests_cnt
                            logger.info(f"Using v1 engine abort_to_target_requests_cnt with target={target_leftover_cnt}")

                            # Use collective_rpc to call the method on the engine core
                            results = self.model.llm_engine.collective_rpc(
                                method='abort_to_target_requests_cnt',
                                args=(target_leftover_cnt,)
                            )
                            # collective_rpc returns a list of results of interrupted requet id or None if no request was interrupted
                            if results is not None:
                                assert isinstance(results, list), f"result from collective_rpc should be a list, got {type(results)}"
                                for interrupted_rids in results:
                                    assert interrupted_rids in self.request_metas, f"interrupted_rids {interrupted_rids} not in request_metas {self.request_metas.keys()}"
                                    interrupted_rid_set.update(interrupted_rids)
                            logger.info(f"V1 engine interrupted {len(interrupted_rid_set)} requests  {interrupted_rid_set}" )
                            # Note: We can't track which specific requests were interrupted in v1
                            logger.info(f"V1 engine processed interruption to target count {target_leftover_cnt}")

                        else:
                            # Fallback for v0 engine - use original implementation
                            # interrupt requests up to the point of target_leftover_cnt,
                            # sort request by the overhead of migration and find the easiest to interrupt
                            
                            # Get current request counts from vLLM engine stats
                            stats = self.model.llm_engine._get_stats(scheduler_outputs=None)
                            
                            # Count requests in different queues
                            waiting_count = stats.num_waiting_sys
                            running_count = stats.num_running_sys
                            swapped_count = stats.num_swapped_sys
                            total_count = waiting_count + running_count + swapped_count
                            
                            # Calculate how many requests to interrupt
                            interrupt_count = total_count - target_leftover_cnt
                            
                            if interrupt_count <= 0:
                                logger.info(f"No interruption needed: total_count={total_count}, target_leftover={target_leftover_cnt}")
                                continue
                                
                            logger.info(f"Dynamic load balance: need to interrupt {interrupt_count} requests "
                                       f"(waiting={waiting_count}, running={running_count}, swapped={swapped_count}, "
                                       f"target_leftover={target_leftover_cnt})")
                            
                            # Get the scheduler to access request queues
                            scheduler = self.model.llm_engine.scheduler[0]
                            
                            # Get all requests from queues
                            waiting_requests = list(scheduler.waiting)
                            swapped_requests = list(scheduler.swapped)
                            running_requests = list(scheduler.running)
                            
                            # Merge swapped and running requests and sort by total length (shortest first)
                            swapped_and_running = []
                            
                            # Add swapped requests
                            for seq_group in swapped_requests:
                                request_id = seq_group.request_id
                                assert request_id in self.request_metas, f"request_id {request_id} not found in request_metas buffer"
                                
                                # Calculate total sequence length (prompt + generated tokens)
                                total_length = 0
                                for seq in seq_group.get_seqs():
                                    total_length += seq.get_len()
                                
                                swapped_and_running.append((request_id, total_length, 'swapped', seq_group))
                            
                            # Add running requests
                            for seq_group in running_requests:
                                request_id = seq_group.request_id
                                assert request_id in self.request_metas, f"request_id {request_id} not found in request_metas buffer"
                                
                                # Calculate total sequence length (prompt + generated tokens)
                                total_length = 0
                                for seq in seq_group.get_seqs():
                                    total_length += seq.get_len()
                                
                                swapped_and_running.append((request_id, total_length, 'running', seq_group))
                            
                            # Sort by total length (ascending) - interrupt shortest sequences first
                            swapped_and_running.sort(key=lambda x: x[1])
                            
                            # Concatenate all requests in priority order: waiting -> (swapped+running sorted by length)
                            all_requests_ordered = []
                            all_requests_ordered.extend([(sg.request_id, 'waiting', sg) for sg in waiting_requests])
                            all_requests_ordered.extend([(rid, status, sg) for rid, _, status, sg in swapped_and_running])
                            
                            # Select requests to interrupt using while loop
                            requests_to_interrupt = []
                            idx = 0
                            
                            while len(requests_to_interrupt) < interrupt_count and idx < len(all_requests_ordered):
                                request_id, status, seq_group = all_requests_ordered[idx]
                                requests_to_interrupt.append(request_id)
                                
                                if status == 'running':
                                    total_length = sum(seq.get_len() for seq in seq_group.get_seqs())
                                    logger.info(f"Selected {status} request {request_id} for interruption (total_length={total_length})")
                                else:
                                    logger.info(f"Selected {status} request {request_id} for interruption")
                                
                                idx += 1
                            
                            # Step 3: Abort the selected requests
                            for request_id in requests_to_interrupt:
                                logger.info(f"Interrupting request {request_id}")
                                self.model.abort_request(request_id=request_id)
                                interrupted_rid_set.add(request_id)


                elif command == GenerateRequestType.STOP:
                    self.model.abort_request(request_id=list(self.request_metas.keys()))
                    self.request_metas.clear()
                    while not self.command_queue.empty():
                        self.command_queue.get_nowait()
                    # Run llm_engine again to consume all out standing requests and
                    # stop model execute loop, otherwise collective_rpc will stuck by
                    # model execute loop or there will be garbage output at next step.
                    self.model.clear_unfinished_requests()
                    self.running = False
                    self.unfinished_vllm_outputs.clear()
                    return

            finished_vllm_outputs, unfinished_vllm_outputs = self.model.fetch_output()
            # add or update the buffer of unfinished request output
            for request_output in unfinished_vllm_outputs:

                # if request in added_batch/not aborted, update the buffered the partial request output
                if request_output.request_id in self.request_metas:
                    self.unfinished_vllm_outputs[request_output.request_id] = request_output

            # Log finished outputs for debugging
            for f in finished_vllm_outputs:
                # Collect all prompt information
                token_prompt = ""
                text_prompt = ""
                token_ids = []
                
                if hasattr(f, 'prompt_token_ids') and f.prompt_token_ids:
                    token_prompt = self.tokenizer.decode(f.prompt_token_ids, skip_special_tokens=True)
                    token_ids = f.prompt_token_ids
                
                if hasattr(f, 'prompt') and f.prompt:
                    text_prompt = f.prompt
                
                # Collect all outputs
                outputs_info = []
                for i, output in enumerate(f.outputs):
                    output_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                    outputs_info.append(f"Output_{i}[tokens={len(output.token_ids)}]: '{output_text}'")
                
                # Log everything in a single line
                logger.info(f"FINISHED_OUTPUT: request_id={f.request_id}, finished={f.finished} | \n"
                           f"TOKEN_PROMPT: '{token_prompt}' \n| TOKEN_IDS: {token_ids} | \n"
                           f"TEXT_PROMPT: '{text_prompt}' \n| OUTPUTS: {' | '.join(outputs_info)}")
                
            self.handle_vllm_output(finished_vllm_outputs, interrupted_rid_set, request_complete_callback)




            #
            # for request_output in finished_vllm_outputs:
            #     # still in added_batch not aborted, process the request output
            #     if request_output.request_id in self.added_batch:
            #         logger.info(
            #             f"process_vllm_output: finished request from fetch_output and in added_batch,  request_id {request_output.request_id}")
            #         self.process_vllm_output(vllm_outputs=request_output,
            #                                  request_complete_callback=request_complete_callback)
            #         self.unfinished_vllm_outputs.pop(request_output.request_id, None)
            #         self.added_batch.pop(request_output.request_id, None)
            #
            # # add or update the buffer of unfinished request output
            # for request_output in unfinished_vllm_outputs:
            #
            #     # if request in added_batch/not aborted, update the buffered the partial request output
            #     if request_output.request_id in self.added_batch:
            #         self.unfinished_vllm_outputs[request_output.request_id] = request_output
            #
            # self.handle_interrupted_requests(request_complete_callback)


    def add_request(self, command, data: DataProto):
        self.command_queue.put((command, data))

    async def async_generate(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        # TODO: refactor async_generate interface. not supported now!
        raise NotImplementedError()
        from vllm.inputs.data import TokensPrompt

        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask
        assert input_ids.size(0) == 1, f"async_generate: batch['input_ids'] must have exactly one batch dimension"

        # TODO meaningful request id?
        #   async_generate如何实现abort_request
        request_id = random_uuid()
        prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
        result_generator = self.model.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids[0]),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        vllm_output: Optional[RequestOutput] = None
        async for request_output in result_generator:
            vllm_output = request_output
        assert vllm_output is not None

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=[vllm_output], pad_token_id=self.tokenizer.pad_token_id, device=input_ids.device
        )
        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )
        return output

    # offload/reload 接口
    def load_states(self, *args, **kwargs):
        self.model.load_states()

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            self.model.offload_states(self.sleep_level)
        self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # 参数同步相关接口
    def setup_collective_group(self, comm_plan, backend="nccl"):
        self.model.setup_collective_group(comm_plan=comm_plan, backend=backend, rank_in_cluster=self.worker.rank)

    def broadcast_parameter(self, src_pp_rank, dtype, shape, parameter_name):
        self.model.broadcast_parameter(src_pp_rank, dtype, shape, parameter_name)

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        self.model.broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter(self, parameter_name, weight, ranks_in_worker):
        self.model.update_parameter(parameter_name, weight, ranks_in_worker)

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        self.model.update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)


def gather_unpadded_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    # Debug: Log input details
    logger.info(f"GATHER_DEBUG: input_ids_shape={input_ids.shape}, attention_mask_shape={attention_mask.shape}")
    logger.info(f"GATHER_DEBUG: input_ids={input_ids.tolist()}")
    logger.info(f"GATHER_DEBUG: attention_mask sum ={sum(attention_mask)}")
    
    gathered_input_ids = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
    
    # Debug: Log output details
    logger.info(f"GATHER_DEBUG: gathered_input_ids={gathered_input_ids}")
    
    return gathered_input_ids


def gather_outputs_to_pad_tensor(request_outputs: List["RequestOutput"], pad_token_id, device="cuda") -> torch.Tensor:
    token_ids_list_of_lists = [
        torch.tensor(completion_output.token_ids, device=device)
        for request_output in request_outputs
        for completion_output in request_output.outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_vllm(gen_kwargs):
    if gen_kwargs["num_beams"] > 1:
        return SamplingParams(
            max_tokens=gen_kwargs["max_new_tokens"],
            stop_token_ids=gen_kwargs["eos_token_id"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            n=gen_kwargs["num_return_sequences"],
            best_of=gen_kwargs["num_beams"],
            use_beam_search=True,
            logprobs=0,
        )
    return SamplingParams(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        logprobs=0,
    )


def compare_sampling_params(params1: SamplingParams, params2: SamplingParams) -> bool:
    # 只比较采样参数的配置
    param_attrs = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "n",
        "stop_token_ids", 
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "min_p",
        "best_of",
        "stop",
        "ignore_eos",
        "use_beam_search",
        "best_of",
        "use_beam_search",
    ]

    # 比较每个采样参数
    for attr in param_attrs:
        if hasattr(params1, attr) and hasattr(params2, attr):
            val1 = getattr(params1, attr)
            val2 = getattr(params2, attr)
            if val1 != val2:
                print(f"采样参数 {attr} 不同: {val1} != {val2}")
                return False
    return True
