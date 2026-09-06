"""Goal labels for CLI output."""

from __future__ import annotations

from ocq.types import OptimizationGoal


def goal_label(goal: OptimizationGoal) -> str:
    return {
        OptimizationGoal.BALANCED: "Balanced (llmfit composite score when in catalog)",
        OptimizationGoal.SPEED: "Maximum throughput",
        OptimizationGoal.QUALITY: "Highest quality that fits",
        OptimizationGoal.MEMORY: "Minimum memory footprint",
    }[goal]
