# nano-vllm-lab：单卡调度与KV管理实验

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 `bb823b3` 开展实验，
研究长请求进入时的输出停顿，以及KV分配、准入和重计算的取舍，不声称首创这些机制。

本分支增加了阶段交错、单次forward混合batch、按需KV分配及配套测试/实验工具。
分页缓存、前缀复用、Tensor Parallel和decode CUDA Graph等原有能力归属于上游。

- **项目报告：** [设计、结果与边界](benchmarks/RESULTS.md)；[逐轮数值来源](benchmarks/evidence/provenance.json)；[面试讲述提纲](benchmarks/INTERVIEW_GUIDE.md)。
- **实验入口：** [持续到达实验说明](benchmarks/ARRIVAL_EXPERIMENT.md)，对比三种调度策略，固定原KV分配。
- **研究原则：** 同条件对照，保留负结果；更少预留块不等于更快，吞吐与延迟可能相互取舍。
- 私人学习笔记、模型和完整原始结果不随仓库提交；公开报告附逐轮数值摘录与来源校验信息，不把摘录当成完整原始数据。

**当前结果概览（仅限报告中的受控负载）：** 4090/Qwen3-8B的32请求、4请求/秒混合负载中，mixed相对原调度吞吐提高约8.9%，
短请求最大输出间隔的样本p95下降，但短请求TTFT p95增加约37%。按需KV减少未写入块预留，尚无明显吞吐收益，并保留一个延迟退化案例。
1请求/秒组存在未定位的执行异常，暂不用于加速结论。未与生产vLLM做公平性能对照。

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
