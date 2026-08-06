from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CampaignError(RuntimeError):
    code: str
    message: str

    def __post_init__(self) -> None:
        self.code = str(self.code)
        self.message = str(self.message)
        RuntimeError.__init__(self, f"{self.code}: {self.message}")


class TaskLockedError(CampaignError):
    def __init__(self, task_id: str) -> None:
        super().__init__("E_CAMPAIGN_TASK_LOCKED", f"task is already locked: {task_id}")
