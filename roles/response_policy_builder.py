from __future__ import annotations

from core.schemas import CheckerResultPayload, RoleResponsePolicy, RoleType


class RoleResponsePolicyBuilder:
    def build(
        self,
        *,
        role: RoleType,
        mode: str,
        route: str = "",
        patch_stage: str = "idle",
        has_patch_candidate: bool = False,
        checker_results: CheckerResultPayload | None = None,
    ) -> RoleResponsePolicy:
        if mode == "grounded_chat":
            return self._grounded_chat_policy(role, route)
        if mode == "check_response":
            return self._check_response_policy(role, patch_stage, has_patch_candidate, checker_results)
        return self._system_exception_policy(role)

    def _grounded_chat_policy(self, role: RoleType, route: str) -> RoleResponsePolicy:
        is_tool_related = route in {"image_related", "table_related"}
        if role == "Reviewer":
            return RoleResponsePolicy(
                role=role,
                mode="grounded_chat",
                focus_points=(
                    ["evidence_fact", "evidence_risk", "interpretation_risk"]
                    if is_tool_related
                    else ["observation", "risk", "impact"]
                ),
                framing_guidance=(
                    "Start with the concrete evidence-based observation, then explain the main risk, then explain its impact. Do not rewrite."
                    if is_tool_related
                    else "Start with the concrete observation, then explain the main risk, then explain its impact. Do not rewrite."
                ),
                forbid_rewrite=True,
                required_closing="",
            )
        if role == "Advisor":
            return RoleResponsePolicy(
                role=role,
                mode="grounded_chat",
                focus_points=(
                    ["evidence_guidance", "clarification", "next_step"]
                    if is_tool_related
                    else ["guidance", "rationale", "next_step"]
                ),
                framing_guidance=(
                    "Start with the best guidance, then explain the rationale from the evidence, then give the next step. Keep any example short and clearly labeled."
                ),
                forbid_rewrite=True,
                allow_example=True,
                example_max_count=1,
                example_max_words=30,
                example_max_chars=180,
                required_closing="",
            )
        if route == "editor_rewrite":
            closing = "If you want, I can prepare a patch for the confirmed selection after your consent."
        elif is_tool_related:
            closing = "If you want direct editing later, I can prepare a patch for the confirmed selection after your consent."
        else:
            closing = "If you want direct editing, I can prepare a patch for the confirmed selection after your consent."
        return RoleResponsePolicy(
            role=role,
            mode="grounded_chat",
            focus_points=(
                ["edit_analysis", "evidence_support", "next_patch_step"]
                if is_tool_related
                else ["edit_analysis", "change_rationale", "next_patch_step"]
            ),
            framing_guidance=(
                "Start with the edit analysis, then explain whether the evidence supports a safe edit target, then state the next patch step or consent boundary."
            ),
            forbid_rewrite=True,
            editor_phase="analysis_only" if route != "editor_rewrite" else "invite_patch",
            required_closing=closing,
        )

    def _check_response_policy(
        self,
        role: RoleType,
        patch_stage: str,
        has_patch_candidate: bool,
        checker_results: CheckerResultPayload | None,
    ) -> RoleResponsePolicy:
        if role == "Reviewer":
            return RoleResponsePolicy(
                role=role,
                mode="check_response",
                focus_points=["observation", "risk", "impact"],
                framing_guidance="Start with the concrete finding, then explain the risk, then explain its impact. Do not rewrite.",
                forbid_rewrite=True,
            )
        if role == "Advisor":
            return RoleResponsePolicy(
                role=role,
                mode="check_response",
                focus_points=["guidance", "rationale", "next_step"],
                framing_guidance="Start with the most useful guidance, then explain why it follows from the finding, then give the next step. Keep any example short and clearly labeled.",
                forbid_rewrite=True,
                allow_example=True,
                example_max_count=1,
                example_max_words=30,
                example_max_chars=180,
                required_closing=(
                    "Would you like targeted revision suggestions for these issues?"
                    if checker_results and checker_results.issues
                    else ""
                ),
            )
        if patch_stage == "awaiting_apply_consent" and has_patch_candidate:
            return RoleResponsePolicy(
                role=role,
                mode="check_response",
                focus_points=["edit_analysis", "patch_suitability", "next_patch_step"],
                framing_guidance="Start with the edit analysis, then state that the patch is ready, then ask explicitly whether to apply it now.",
                forbid_rewrite=True,
                editor_phase="awaiting_apply",
                required_closing="Do you want me to apply this patch to the confirmed selection now?",
            )
        if patch_stage == "awaiting_preview_consent" and has_patch_candidate:
            return RoleResponsePolicy(
                role=role,
                mode="check_response",
                focus_points=["edit_analysis", "patch_suitability", "next_patch_step"],
                framing_guidance="Start with the edit analysis, then explain that a patch candidate is available, then ask explicitly whether to show and prepare the diff.",
                forbid_rewrite=True,
                editor_phase="invite_patch",
                required_closing="Do you want me to show and prepare a patch diff now?",
            )
        return RoleResponsePolicy(
            role=role,
            mode="check_response",
            focus_points=["edit_analysis", "patch_suitability", "next_patch_step"],
            framing_guidance="Start with the edit analysis, then explain whether a safe patch target exists, then state the next patch step or consent boundary.",
            forbid_rewrite=True,
            editor_phase="analysis_only",
            required_closing="No edit will be applied unless you explicitly ask for a patch and confirm it.",
        )

    def _system_exception_policy(self, role: RoleType) -> RoleResponsePolicy:
        focus_points = {
            "Reviewer": ["missing_dependency", "verification_block"],
            "Advisor": ["missing_dependency", "next_step"],
            "Editor": ["missing_dependency", "editing_block"],
        }
        return RoleResponsePolicy(
            role=role,
            mode="system_exception",
            focus_points=focus_points.get(role, ["missing_dependency"]),
            framing_guidance="State the blocking dependency first, then explain the consequence, then state the next step.",
            forbid_rewrite=True,
        )
