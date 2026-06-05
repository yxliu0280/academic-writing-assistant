from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from core.schemas import ToolFact
from tools.providers import VLMProvider, build_vlm_provider


class ImageQATool:
    """Minimal image-routing skeleton for ordinary grounded chat.

    V1 only extracts lightweight numeric/chart facts through the shared multimodal
    provider and returns normalized tool facts for Router/Role consumption.
    """

    def __init__(self, provider_name: str = "none") -> None:
        self.provider: VLMProvider = build_vlm_provider(provider_name)

    def enabled(self) -> bool:
        return self.provider.enabled()

    def analyze(self, image_path: Path, user_request: str = "") -> List[ToolFact]:
        if not image_path.exists() or not image_path.is_file():
            return []
        try:
            image_bytes = image_path.read_bytes()
        except OSError:
            return []
        facts = self.provider.extract_numeric_facts(
            image_bytes=image_bytes,
            meta={
                "path": str(image_path),
                "name": image_path.name,
                "ext": image_path.suffix.lstrip("."),
                "user_request": user_request,
            },
        )
        if not facts:
            return []
        return [
            ToolFact(
                kind="image_numeric_fact",
                source=str(image_path.name),
                summary=(
                    f"Figure fact: figure_id={fact.figure_id or 'unknown'}, value={fact.value}, "
                    f"evidence_type={fact.evidence_type}, confidence={fact.confidence:.2f}"
                ),
                data={
                    "figure_id": fact.figure_id,
                    "value": fact.value,
                    "evidence_type": fact.evidence_type,
                    "confidence": fact.confidence,
                    "source": fact.source,
                },
            )
            for fact in facts
        ]

    @staticmethod
    def summarize_available_images(image_entries: List[Dict[str, str]]) -> List[ToolFact]:
        return [
            ToolFact(
                kind="image_resource",
                source=str(item.get("path", "")),
                summary=f"Available image resource: {item.get('path', '')}",
                data=dict(item),
            )
            for item in image_entries
        ]
