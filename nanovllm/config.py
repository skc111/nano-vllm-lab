import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduling_policy: str = "prefill_first"
    kv_allocation: str = "full"
    kv_cache_blocks: int | None = None  # optional fixed pool for capacity experiments
    observe_kv: bool = False

    def __post_init__(self):
        if self.kv_allocation not in ("full", "on_demand"):
            raise ValueError("kv_allocation must be 'full' or 'on_demand'")
        if self.kv_allocation == "on_demand" and (self.scheduling_policy != "mixed" or self.tensor_parallel_size != 1):
            raise ValueError("on_demand currently requires mixed scheduling on one GPU")
        if self.kv_cache_blocks is not None and (type(self.kv_cache_blocks) is not int or self.kv_cache_blocks <= 0):
            raise ValueError("kv_cache_blocks must be a positive integer")
        if self.scheduling_policy not in ("prefill_first", "interleave", "mixed"):
            raise ValueError("scheduling_policy must be 'prefill_first', 'interleave' or 'mixed'")
        if self.scheduling_policy == "mixed" and self.tensor_parallel_size != 1:
            raise ValueError("mixed scheduling currently supports a single GPU only")
        if self.max_num_seqs <= 0 or self.max_num_batched_tokens <= 0:
            raise ValueError("request and token budgets must be positive")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
