from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.scheduling_policy = config.scheduling_policy
        self.kv_allocation = getattr(config, "kv_allocation", "full")
        self.max_model_len = getattr(config, "max_model_len", None)
        self.observe_kv = getattr(config, "observe_kv", False)
        self.preemptions = 0
        self.evicted_cached_tokens = 0
        self.recomputed_tokens = 0
        self._computed_high_water = {}
        self._last_was_prefill = False
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if self.kv_allocation == "on_demand":
            # Conservative admission: even without sharing, one request must
            # fit through its final forward. The final sampled token has no KV.
            final_tokens = seq.num_prompt_tokens + seq.max_tokens - 1
            if seq.max_tokens <= 0 or final_tokens > len(self.block_manager.blocks) * self.block_size:
                raise ValueError("request cannot finish within the KV pool at its output limit")
            if self.max_model_len is not None and final_tokens + 1 > self.max_model_len:
                raise ValueError("prompt plus output limit exceeds max_model_len")
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.kv_allocation == "on_demand":
            return self.schedule_on_demand()
        if self.scheduling_policy == "mixed":
            return self.schedule_mixed()
        scheduled_seqs = []
        num_batched_tokens = 0

        # Interleave whole steps; this is NOT a mixed prefill/decode forward.
        # Require the first decode to make progress without preemption. Otherwise
        # a partial prefill holding blocks could be stranded behind a requeued
        # decode request when a forced decode produces an empty batch.
        yield_to_decode = (
            self.scheduling_policy == "interleave"
            and self._last_was_prefill
            and bool(self.running)
            and self.block_manager.can_append(self.running[0])
        )

        # prefill
        while self.waiting and not yield_to_decode and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self._last_was_prefill = True
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        self._last_was_prefill = False
        return scheduled_seqs, False

    def schedule_mixed(self) -> tuple[list[Sequence], bool]:
        """Decode first, then chunk prefill in the same varlen forward.

        The bool selects the attention execution path (varlen vs decode), NOT
        the phase of every request. Per-request phase is seq.is_prefill.
        Full prompt block allocation is intentionally unchanged.
        """
        decodes, prefills = [], []
        remaining = self.max_num_batched_tokens
        # Under a full decode budget, give waiting prefill one slot/token on
        # alternate rounds. Rotate decode requests so the same tail cannot starve.
        reserve = int(bool(self.waiting and self.running and not self._last_was_prefill))

        def take_decode(slots, token_reserve):
            nonlocal remaining
            while self.running and len(decodes) < slots and remaining > token_reserve:
                if (self.waiting and self.waiting[0].block_table
                        and not self.block_manager.can_append(self.running[0])):
                    # Let an already allocated partial prefill advance rather
                    # than put a preempted decode in front of it with no room.
                    break
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                    decodes.append(seq)
                    remaining -= 1

        take_decode(self.max_num_seqs - reserve, reserve)
        while self.waiting and remaining and len(decodes) + len(prefills) < self.max_num_seqs:
            seq = self.waiting[0]
            if not seq.block_table:
                cached_blocks = self.block_manager.can_allocate(seq)
                if cached_blocks < 0:
                    break
                self.block_manager.allocate(seq, cached_blocks)
            seq.is_prefill = True
            seq.num_scheduled_tokens = min(len(seq) - seq.num_cached_tokens, remaining)
            assert seq.num_scheduled_tokens > 0
            remaining -= seq.num_scheduled_tokens
            prefills.append(seq)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                # Add only AFTER decode selection, so a new prefill is not
                # accidentally selected again in the fallback below.
            else:
                break
        if not prefills and reserve:
            take_decode(self.max_num_seqs, 0)
        self.running.extend(decodes)
        self.running.extend(s for s in prefills if s.status == SequenceStatus.RUNNING)
        selected = decodes + prefills
        if not selected:
            raise RuntimeError("mixed scheduler cannot make progress with the current KV capacity")
        self._last_was_prefill = bool(prefills)
        return selected, bool(prefills)

    def schedule_on_demand(self) -> tuple[list[Sequence], bool]:
        """Mixed scheduling with incremental blocks and bounded partial admission.

        At most one unfinished prefill holds blocks. Ordinary capacity failures
        defer that request; they never evict a selected request. If no query can
        run, reclaim younger owners and run the oldest request alone. That oldest
        request cannot be evicted by recovery, ensuring finite-work progress
        for admitted requests (not a latency/fairness guarantee under overload).
        """
        selected, prefills = [], []
        remaining = self.max_num_batched_tokens
        reserve = int(bool(self.waiting and self.running and not self._last_was_prefill))
        manager = self.block_manager

        def decodes(slots, reserved_tokens):
            nonlocal remaining
            for seq in list(self.running):
                if len(selected) >= slots or remaining <= reserved_tokens:
                    break
                if seq in selected:
                    continue
                if not manager.allocate_chunk(seq, 1):
                    continue
                seq.is_prefill = False
                seq.num_scheduled_tokens = 1
                selected.append(seq)
                remaining -= 1

        decodes(self.max_num_seqs - reserve, reserve)
        while self.waiting and remaining and len(selected) < self.max_num_seqs:
            seq = self.waiting[0]
            query = manager.allocate_chunk(seq, remaining)
            if not query:
                break
            seq.is_prefill = True
            seq.num_scheduled_tokens = query
            remaining -= query
            selected.append(seq); prefills.append(seq)
            if seq.num_cached_tokens + query == len(seq):
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
            else:
                break
        if not prefills and reserve:
            decodes(self.max_num_seqs, 0)
        if not selected:
            owners = list(self.running) + list(self.waiting)
            if not owners:
                raise RuntimeError("cannot schedule an empty queue")
            oldest = min(owners, key=lambda s: s.seq_id)
            oldest.is_prefill = oldest.status == SequenceStatus.WAITING
            budget = self.max_num_batched_tokens if oldest.is_prefill else 1
            query = manager.allocate_chunk(oldest, budget)
            for victim in sorted(owners, key=lambda s: s.seq_id, reverse=True):
                if query:
                    break
                if victim is oldest or not victim.block_table:
                    continue
                if victim in self.running:
                    self.running.remove(victim)
                else:
                    self.waiting.remove(victim)
                self.preempt(victim)
                query = manager.allocate_chunk(oldest, budget)
            if not query:
                raise RuntimeError("KV recovery failed despite single-request capacity validation")
            oldest.num_scheduled_tokens = query
            selected = [oldest]
            if oldest.is_prefill:
                prefills = [oldest]
                self.waiting.remove(oldest)
                if oldest.num_cached_tokens + query == len(oldest):
                    oldest.status = SequenceStatus.RUNNING
                else:
                    self.waiting.appendleft(oldest)
        # Rotate only selected running requests. New completed prefills were
        # never eligible for decode in this step.
        for seq in selected:
            if seq in self.running:
                self.running.remove(seq)
            if seq.status == SequenceStatus.RUNNING:
                self.running.append(seq)
        self._last_was_prefill = bool(prefills)
        return selected, bool(prefills)

    def kv_snapshot(self):
        """Physical pool occupancy before forward, not nvidia-smi allocation."""
        written = set()
        for seq in list(self.running) + list(self.waiting):
            written.update(seq.block_table[:(seq.num_cached_tokens + self.block_size - 1) // self.block_size])
        return {
            "allocated_blocks_before_forward": len(self.block_manager.used_block_ids),
            "unwritten_blocks_before_forward": len(self.block_manager.used_block_ids - written),
            "preemptions_total": self.preemptions,
            "evicted_cached_tokens_total": self.evicted_cached_tokens,
            "recomputed_tokens_total": self.recomputed_tokens,
        }

    def preempt(self, seq: Sequence):
        self.preemptions += 1
        self.evicted_cached_tokens += seq.num_cached_tokens
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.kv_allocation == "on_demand":
            self.waiting.append(seq)  # do not strand an existing partial prefill
        else:
            self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int | None], is_prefill: bool):
        if len(seqs) != len(token_ids):
            raise ValueError("runner output must preserve one entry per scheduled request")
        for seq, token_id in zip(seqs, token_ids):
            high = self._computed_high_water.get(seq.seq_id, 0)
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            self.recomputed_tokens += max(0, min(high, end) - seq.num_cached_tokens)
            self._computed_high_water[seq.seq_id] = max(high, end)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if token_id is None:
                raise ValueError("missing sampled token for a completed query")
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self._computed_high_water.pop(seq.seq_id, None)
