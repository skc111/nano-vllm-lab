# nano-vllm-lab：单卡调度与KV管理实验

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 `bb823b3` 开展实验，
研究长请求进入时的输出停顿，以及KV分配、准入和重计算的取舍，不声称首创这些机制。

本分支增加了阶段交错、单次forward混合batch、按需KV分配及配套测试/实验工具。
分页缓存、前缀复用、Tensor Parallel和decode CUDA Graph等原有能力归属于上游。

- **当前实验入口：** [持续到达实验说明](benchmarks/ARRIVAL_EXPERIMENT.md)，对比三种调度策略，固定原KV分配。
- **研究原则：** 同条件对照，保留负结果；更少预留块不等于更快，吞吐与延迟可能相互取舍。
- 私人学习笔记、模型和原始结果目录不随仓库提交；实验会保存版本、配置、日志与逐请求原始记录。

---

## 上游说明（保留来源）

以下为上游介绍与示例；其中性能数字、徽章及“从零实现”的表述属于上游，**不是本分支的优化成果**。
下面的安装命令安装的是上游；运行本分支实验请使用自己的克隆及已验收环境，见上面的实验说明。

<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
