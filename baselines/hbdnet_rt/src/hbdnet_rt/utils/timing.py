"""计时工具，用于各模块延迟统计。"""

import time
from contextlib import contextmanager
from collections import defaultdict


class Timer:
    """轻量计时器。统计各阶段的平均耗时。"""

    def __init__(self):
        self._records: dict = defaultdict(list)

    @contextmanager
    def measure(self, name: str):
        """上下文管理器，自动记录耗时。用法: with timer.measure("dwa"): ..."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000
        self._records[name].append(elapsed)

    def stats(self) -> dict:
        """返回各阶段耗时统计 (ms)。"""
        return {
            name: {
                "mean_ms": sum(vals) / len(vals),
                "min_ms": min(vals),
                "max_ms": max(vals),
                "count": len(vals),
            }
            for name, vals in self._records.items()
        }

    def total_mean_ms(self) -> float:
        """总平均耗时。"""
        return sum(s["mean_ms"] for s in self.stats().values())

    def reset(self):
        self._records.clear()
