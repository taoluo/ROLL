"""Critic Parity Testing Module

This module contains tests for comparing the parity between DeepSpeed and Megatron
critic implementations in reinforcement learning pipelines.
"""

from .critic_parity_pipeline import CriticParityPipeline

__all__ = ["CriticParityPipeline"]