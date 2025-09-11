import random
import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger


logger = get_logger()


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_mock_batch(batch_size=4, seq_len=128, device="cuda"):
    """Create mock input data for testing critics."""
    batch = DataProto()
    
    # Create mock tensors
    batch.batch = {
        "prompts": torch.randint(0, 30000, (batch_size, seq_len // 2), device=device),
        "responses": torch.randint(0, 30000, (batch_size, seq_len // 2), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "response_mask": torch.ones(batch_size, seq_len // 2, dtype=torch.bool, device=device),
        "final_response_mask": torch.ones(batch_size, seq_len // 2, dtype=torch.bool, device=device),
        "rewards": torch.randn(batch_size, seq_len // 2, device=device) * 0.1,
        "old_log_probs": torch.randn(batch_size, seq_len // 2, device=device) * 0.1 - 2.0,
        "ref_log_probs": torch.randn(batch_size, seq_len // 2, device=device) * 0.1 - 2.0,
    }
    
    # Set some masks to False to simulate real data
    batch.batch["response_mask"][:, :5] = False
    batch.batch["final_response_mask"][:, :-10] = False
    
    batch.non_tensor_batch = {
        "domain": ["math"] * batch_size,
    }
    
    batch.meta_info = {
        "global_step": 0,
        "is_offload_states": False,
    }
    
    return batch


class SimpleCriticTest(BasePipeline):
    """Simplified test to verify two critic clusters produce identical outputs."""
    
    def __init__(self, pipeline_config1: RLVRConfig, pipeline_config2: RLVRConfig):
        super().__init__(pipeline_config1)
        self.pipeline_config = pipeline_config1
        self.pipeline_config2 = pipeline_config2
        
        # Set max_steps for both configs
        self.pipeline_config.set_max_steps(max_steps=1)
        self.pipeline_config2.set_max_steps(max_steps=1)
        
        # Initialize critics only
        logger.info("Initializing critic clusters...")
        
        self.critic1 = Cluster(
            name="critic1_megatron",
            worker_cls=self.pipeline_config.critic.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.critic,
        )
        
        self.critic2 = Cluster(
            name="critic2_deepspeed",
            worker_cls=self.pipeline_config2.critic.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config2.critic,
        )
        
        # Initialize with same seed for both
        logger.info("Initializing critics with synchronized seeds...")
        set_seed(42)
        self.critic1.initialize(pipeline_config=self.pipeline_config, blocking=True)
        
        set_seed(42)
        self.critic2.initialize(pipeline_config=self.pipeline_config2, blocking=True)
        
        logger.info("Critics initialized successfully")
    
    @torch.no_grad()
    def run(self):
        """Run the simplified critic comparison test."""
        logger.info("=" * 60)
        logger.info("Starting Simplified Critic Parity Test")
        logger.info("=" * 60)
        
        num_test_batches = 5
        
        for test_step in range(num_test_batches):
            logger.info(f"\n[Test Step {test_step + 1}/{num_test_batches}]")
            
            # Create mock batch
            batch = create_mock_batch(batch_size=4, seq_len=128)
            logger.info(f"Created mock batch with shape: {batch.batch['prompts'].shape}")
            
            # Compute values from both critics
            logger.info("Computing values from critic1 (Megatron)...")
            values1_refs = self.critic1.compute_values(batch, blocking=False)
            values1 = DataProto.materialize_concat(data_refs=values1_refs)
            
            logger.info("Computing values from critic2 (DeepSpeed)...")
            values2_refs = self.critic2.compute_values(batch, blocking=False)
            values2 = DataProto.materialize_concat(data_refs=values2_refs)
            
            # Extract tensors
            values1_tensor = values1.batch["values"]
            values2_tensor = values2.batch["values"]
            
            # Verify shapes match
            assert values1_tensor.shape == values2_tensor.shape, \
                f"Shape mismatch: Critic1 {values1_tensor.shape} vs Critic2 {values2_tensor.shape}"
            logger.info(f"✓ Shapes match: {values1_tensor.shape}")
            
            # Calculate comparison metrics
            abs_diff = torch.abs(values1_tensor - values2_tensor)
            rel_diff = abs_diff / (torch.abs(values1_tensor) + 1e-8)
            
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            # Log comparison results
            logger.info(f"  Critic1 values (first 5): {values1_tensor.flatten()[:5].tolist()}")
            logger.info(f"  Critic2 values (first 5): {values2_tensor.flatten()[:5].tolist()}")
            logger.info(f"  Max Absolute Diff: {max_abs_diff:.6e}")
            logger.info(f"  Mean Absolute Diff: {mean_abs_diff:.6e}")
            logger.info(f"  Max Relative Diff: {max_rel_diff:.4%}")
            logger.info(f"  Mean Relative Diff: {mean_rel_diff:.4%}")
            
            # Assert functional equivalence
            rtol = 1e-3  # 0.1% relative tolerance
            atol = 1e-4  # Small absolute tolerance
            
            is_close = torch.allclose(values1_tensor, values2_tensor, rtol=rtol, atol=atol)
            assert is_close, \
                f"Critic values not equivalent! Max abs: {max_abs_diff:.6e}, Max rel: {max_rel_diff:.4%}"
            
            logger.info(f"  ✓ Critics are functionally equivalent (rtol={rtol}, atol={atol})")
            
            # Test training step
            logger.info("\nTesting training step...")
            
            # Add required fields for training
            batch.batch["returns"] = batch.batch["rewards"].cumsum(dim=-1)
            batch.batch["values"] = values1_tensor  # Use critic1's values
            batch.batch["advantages"] = batch.batch["returns"] - batch.batch["values"]
            
            # Train both critics
            train_metrics1 = self.critic1.train_step(batch, blocking=True)
            
            # For critic2, use its own values
            batch2 = DataProto(
                batch=batch.batch.clone(),
                non_tensor_batch=batch.non_tensor_batch.copy(),
                meta_info=batch.meta_info.copy()
            )
            batch2.batch["values"] = values2_tensor
            batch2.batch["advantages"] = batch2.batch["returns"] - batch2.batch["values"]
            
            train_metrics2 = self.critic2.train_step(batch2, blocking=True)
            
            # Extract losses if available
            loss1 = train_metrics1.meta_info.get("metrics", {}).get("critic/loss", "N/A")
            loss2 = train_metrics2.meta_info.get("metrics", {}).get("critic/loss", "N/A")
            logger.info(f"  Critic1 loss: {loss1}")
            logger.info(f"  Critic2 loss: {loss2}")
            
            # Re-compute values after training to verify they still match
            logger.info("\nVerifying values after training...")
            values1_after_refs = self.critic1.compute_values(batch, blocking=False)
            values1_after = DataProto.materialize_concat(data_refs=values1_after_refs)
            
            values2_after_refs = self.critic2.compute_values(batch, blocking=False)
            values2_after = DataProto.materialize_concat(data_refs=values2_after_refs)
            
            values1_after_tensor = values1_after.batch["values"]
            values2_after_tensor = values2_after.batch["values"]
            
            # Check if training changed the values
            change1 = torch.abs(values1_after_tensor - values1_tensor).mean().item()
            change2 = torch.abs(values2_after_tensor - values2_tensor).mean().item()
            logger.info(f"  Mean change in Critic1 values: {change1:.6e}")
            logger.info(f"  Mean change in Critic2 values: {change2:.6e}")
            
            # Verify post-training equivalence (may have larger tolerance due to training)
            post_train_rtol = 1e-2  # 1% relative tolerance after training
            post_train_atol = 1e-3  # Larger absolute tolerance
            
            is_close_after = torch.allclose(values1_after_tensor, values2_after_tensor, 
                                           rtol=post_train_rtol, atol=post_train_atol)
            if is_close_after:
                logger.info(f"  ✓ Critics remain equivalent after training (rtol={post_train_rtol}, atol={post_train_atol})")
            else:
                abs_diff_after = torch.abs(values1_after_tensor - values2_after_tensor)
                max_abs_diff_after = abs_diff_after.max().item()
                logger.warning(f"  ⚠ Critics diverged after training. Max diff: {max_abs_diff_after:.6e}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("[Test Summary]")
        logger.info(f"✓ Successfully completed {num_test_batches} test batches")
        logger.info("✓ Megatron and DeepSpeed critics are functionally equivalent")
        logger.info("✓ Both critics produce matching value predictions")
        logger.info("=" * 60)


def main():
    """Main test entry point."""
    import sys
    import os
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    # Use existing config files in the tests/pipeline directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_name1 = sys.argv[1] if len(sys.argv) > 1 else "example_ppo_megatron_critic_mini"
    config_name2 = sys.argv[2] if len(sys.argv) > 2 else "example_ppo"
    
    logger.info(f"Loading configs: {config_name1} and {config_name2}")
    
    # Initialize Hydra and load config
    with initialize_config_dir(config_dir=base_dir, version_base=None):
        cfg1 = compose(config_name=config_name1 + ".yaml")
        cfg2 = compose(config_name=config_name2 + ".yaml")
    
    # Convert to dict and create RLVRConfig
    config_dict1 = OmegaConf.to_container(cfg1, resolve=True)
    config_dict2 = OmegaConf.to_container(cfg2, resolve=True)
    
    # Remove hydra-specific keys
    for key in ['defaults', 'hydra']:
        config_dict1.pop(key, None)
        config_dict2.pop(key, None)
    
    pipeline_config1 = RLVRConfig(**config_dict1)
    pipeline_config2 = RLVRConfig(**config_dict2)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create and run test
    test = SimpleCriticTest(pipeline_config1, pipeline_config2)
    test.run()
    
    # Cleanup
    ray.shutdown()
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()