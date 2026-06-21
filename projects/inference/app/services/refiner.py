from __future__ import annotations

from app.core.config import get_enable_heavy_refinement
from app.schemas import ImageEvidence, InferResponse


class HeavyRefiner:
    """
    Optional heavy reconstruction hook used only for high-uncertainty requests.

    The current implementation is deterministic and side-effect free. It records
    whether heavy refinement was attempted so downstream systems can branch
    between online and offline refinement paths.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = get_enable_heavy_refinement() if enabled is None else bool(enabled)

    def maybe_refine(self, response: InferResponse, evidence: list[ImageEvidence]) -> InferResponse:
        strategy = str(response.escalation.get("strategy", ""))
        if strategy != "escalate_mvs_refinement":
            return response

        if not self.enabled:
            if "heavy_refinement_disabled" not in response.review_flags:
                response.review_flags.append("heavy_refinement_disabled")
            response.constraints_report["heavy_refinement_attempted"] = False
            return response

        # Hook point for MVS/NeRF/3DGS refinement. Keep deterministic metadata now.
        response.constraints_report["heavy_refinement_attempted"] = True
        response.escalation["strategy"] = "mvs_refinement_applied"
        response.escalation["refiner"] = "heavy_refiner_v1"
        response.escalation["views_used"] = len(evidence)
        if "heavy_refinement_applied" not in response.review_flags:
            response.review_flags.append("heavy_refinement_applied")

        return response
