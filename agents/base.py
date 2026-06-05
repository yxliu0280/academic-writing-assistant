from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.schemas import AppState, Issue


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, state: AppState) -> List[Issue]:
        raise NotImplementedError
