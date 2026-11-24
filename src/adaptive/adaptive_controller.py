import os
from collections import deque
from threading import Lock


class AdaptiveCacheController:

    def __init__(
        self,
        expected_llm_latency_ms=None,
        expected_llm_latency_prefix_hit_ms=None,
        expected_prefix_hit_rate=None,
        min_samples=20,
        window_size=100,
        exploration_interval=10,
    ):
        self.expected_llm_latency_ms = (
            expected_llm_latency_ms
            if expected_llm_latency_ms is not None
            else float(
                os.getenv("EXPECTED_LLM_LATENCY_MS", "2000")
            )
        )

        self.expected_llm_latency_prefix_hit_ms = (
            expected_llm_latency_prefix_hit_ms
            if expected_llm_latency_prefix_hit_ms is not None
            else float(
                os.getenv(
                    "EXPECTED_LLM_LATENCY_PREFIX_HIT_MS",
                    "800",
                )
            )
        )

        if expected_prefix_hit_rate is None:
            expected_prefix_hit_rate = float(
                os.getenv("EXPECTED_PREFIX_HIT_RATE", "0")
            )

        self.expected_prefix_hit_rate = min(1.0,max(0.0, float(expected_prefix_hit_rate)),)

        self.min_samples = min_samples
        self.exploration_interval = exploration_interval

        # Each entry is (was_semantic_hit, lookup_latency_ms).
        self.observations = deque(maxlen=window_size)

        self.decisions = 0
        self.lock = Lock()

    def effective_llm_latency_ms(self):
        #Expected inference latency based on the configured probability of prefix-cache reuse.
        
        p = self.expected_prefix_hit_rate

        return ( (1 - p) * self.expected_llm_latency_ms + (p * self.expected_llm_latency_prefix_hit_ms))

    def should_try_semantic(self):
        with self.lock:
            self.decisions += 1

            # Initially explore to collect observations.
            if len(self.observations) < self.min_samples:
                return True

            # Periodically explore even when lookup is not
            # currently predicted to be worthwhile.
            if self.decisions % self.exploration_interval == 0:
                return True

            hit_rate = sum( was_hit for was_hit, _ in self.observations)/len(self.observations)
            avg_lookup_ms = sum( latency for _, latency in self.observations)/len(self.observations)
            expected_savings_ms = ( hit_rate * self.effective_llm_latency_ms())

            return expected_savings_ms > avg_lookup_ms

    def record(self, was_hit, lookup_latency_ms):
        if lookup_latency_ms < 0:
            raise ValueError("Lookup latency cannot be negative.")

        with self.lock:
            self.observations.append(
                (bool(was_hit), float(lookup_latency_ms))
            )

    def get_stats(self):
        with self.lock:
            count = len(self.observations)
            effective_latency = self.effective_llm_latency_ms()

            if count == 0:
                hit_rate = None
                avg_lookup_ms = None
                recommended = None
            else:
                hit_rate = sum( was_hit for was_hit,_ in self.observations)/count
                avg_lookup_ms = sum( latency for _,latency in self.observations)/count
                recommended = ( hit_rate * effective_latency > avg_lookup_ms)

            return {
                "observations": count,
                "semantic_hit_rate": hit_rate,
                "average_semantic_lookup_ms": avg_lookup_ms,
                "expected_llm_latency_without_prefix_reuse_ms":
                    self.expected_llm_latency_ms,
                "expected_llm_latency_with_prefix_reuse_ms":
                    self.expected_llm_latency_prefix_hit_ms,
                "expected_prefix_hit_rate":
                    self.expected_prefix_hit_rate,
                "effective_expected_llm_latency_ms":
                    effective_latency,
                "semantic_lookup_recommended": recommended,
            }
