"""Privileged physical consistency and P/U/N projection for Pipeline v3.

The information score remains the canonical

    D_t = gamma * phi_v(t+1) - phi_v(t) + beta * psi_a(t).

This module never adds privileged geometry to ``D_t``.  Instead, physical
evidence is used categorically to validate, contradict, or abstain from the
soft information judgment before producing Positive / Unclear / Negative
supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

from mi_reward.data.libero_privileged import classify_physical_transition


P_LABEL = "Positive"
U_LABEL = "Unclear"
N_LABEL = "Negative"


@dataclass(frozen=True)
class PhysicalConsistencyEvidence:
    """Raw categorical evidence for one transition.

    ``relation_direction`` is the privileged task-stage direction used by the
    full teacher.  The event fields are kept separately so experiments can
    evaluate a weaker gate that does not consume displacement direction.
    """

    relation_direction: int
    entered_success: bool
    acquired_grasp: bool
    lost_grasp: bool
    gained_goal_contact: bool
    lost_goal_contact: bool
    terminal_failure: bool = False

    @classmethod
    def from_frames(
        cls,
        left: dict[str, object],
        right: dict[str, object],
    ) -> "PhysicalConsistencyEvidence":
        left_success = bool(left.get("environment_success", False))
        right_success = bool(right.get("environment_success", False))
        left_grasp = bool(left.get("grasped", False))
        right_grasp = bool(right.get("grasped", False))
        left_contact = bool(left.get("object_goal_contact", False))
        right_contact = bool(right.get("object_goal_contact", False))
        right_done = bool(right.get("recorded_done", False))
        return cls(
            relation_direction=int(classify_physical_transition(left, right)),
            entered_success=bool(right_success and not left_success),
            acquired_grasp=bool(right_grasp and not left_grasp),
            lost_grasp=bool(left_grasp and not right_grasp),
            gained_goal_contact=bool(right_contact and not left_contact),
            lost_goal_contact=bool(left_contact and not right_contact and not right_success),
            terminal_failure=bool(right_done and not right_success),
        )


@dataclass(frozen=True)
class TeacherConsistencyDecision:
    label: str
    consistency: str
    information_direction: int
    physical_direction: int
    confidence: float
    authoritative_override: bool


def information_direction(score: float, *, deadband: float) -> int:
    if score > float(deadband):
        return 1
    if score < -float(deadband):
        return -1
    return 0


def event_direction(evidence: PhysicalConsistencyEvidence) -> int:
    """Return strong event direction without relation-distance evidence."""

    if evidence.entered_success:
        return 1
    if evidence.terminal_failure or evidence.lost_grasp or evidence.lost_goal_contact:
        return -1
    if evidence.acquired_grasp or evidence.gained_goal_contact:
        return 1
    return 0


def physical_direction(
    evidence: PhysicalConsistencyEvidence,
    *,
    use_relation_direction: bool,
) -> int:
    event = event_direction(evidence)
    if event != 0:
        return event
    if use_relation_direction:
        return int(evidence.relation_direction)
    return 0


def _confidence(
    score: float,
    deadband: float,
    *,
    consistency: str,
    authoritative: bool,
) -> float:
    if authoritative:
        return 1.0
    scale = max(float(deadband), 1e-6)
    information_strength = min(abs(float(score)) / (3.0 * scale), 1.0)
    if consistency == "consistent":
        return 0.5 + 0.5 * information_strength
    if consistency == "insufficient":
        return 0.25 * information_strength
    return 0.0


def project_teacher_label(
    score: float,
    evidence: PhysicalConsistencyEvidence,
    *,
    deadband: float,
    use_relation_direction: bool = True,
) -> TeacherConsistencyDecision:
    """Project information + privileged consistency into P/U/N.

    The ordering follows Pipeline v3 rather than a weighted reward sum:

    * measured success is authoritative Positive;
    * grasp/contact loss and full-gate physical regression are authoritative
      Negative evidence;
    * positive information requires supporting physical progress;
    * conflicting or insufficient evidence abstains to Unclear;
    * sufficiently negative information remains Negative unless contradicted
      by positive physical evidence.
    """

    info = information_direction(float(score), deadband=float(deadband))
    physical = physical_direction(evidence, use_relation_direction=use_relation_direction)

    if evidence.entered_success:
        return TeacherConsistencyDecision(P_LABEL, "consistent", info, 1, 1.0, True)

    strong_negative = bool(
        evidence.terminal_failure or evidence.lost_grasp or evidence.lost_goal_contact
    )
    relation_negative = bool(use_relation_direction and evidence.relation_direction < 0)
    if strong_negative or relation_negative:
        return TeacherConsistencyDecision(N_LABEL, "consistent" if info <= 0 else "contradicted", info, -1, 1.0, True)

    if info < 0:
        if physical > 0:
            consistency = "contradicted"
            label = U_LABEL
        else:
            consistency = "consistent" if physical < 0 else "insufficient"
            label = N_LABEL
        return TeacherConsistencyDecision(
            label,
            consistency,
            info,
            physical,
            _confidence(score, deadband, consistency=consistency, authoritative=False),
            False,
        )

    if info > 0:
        if physical > 0:
            consistency = "consistent"
            label = P_LABEL
        elif physical < 0:
            consistency = "contradicted"
            label = N_LABEL if use_relation_direction else U_LABEL
        else:
            consistency = "insufficient"
            label = U_LABEL
        return TeacherConsistencyDecision(
            label,
            consistency,
            info,
            physical,
            _confidence(score, deadband, consistency=consistency, authoritative=False),
            False,
        )

    # Information lies in the calibrated dead band.  Positive physical events
    # are not enough to invent progress except for measured success above;
    # negative authoritative relation/event evidence was handled above.
    consistency = "insufficient" if physical == 0 else "contradicted"
    return TeacherConsistencyDecision(
        U_LABEL,
        consistency,
        0,
        physical,
        _confidence(score, deadband, consistency=consistency, authoritative=False),
        False,
    )


def project_teacher_label_veto(
    score: float,
    evidence: PhysicalConsistencyEvidence,
    *,
    deadband: float,
    use_relation_direction: bool = False,
) -> TeacherConsistencyDecision:
    """Less conservative consistency gate for coverage-sensitive supervision.

    Unlike :func:`project_teacher_label`, absence of a positive physical event
    is *not* treated as evidence insufficiency.  Strong information judgments
    are retained unless privileged evidence contradicts them.  This is useful
    for measuring the specific value of physical vetoes without conflating
    that value with a large abstention-rate increase.

    The gate still respects Pipeline-v3 authority rules:

    * measured success overrides to Positive;
    * grasp/contact loss overrides to Negative;
    * optional relation regression overrides to Negative;
    * positive physical evidence conflicting with a Negative information score
      abstains to Unclear rather than inventing Positive progress.
    """

    info = information_direction(float(score), deadband=float(deadband))
    physical = physical_direction(evidence, use_relation_direction=use_relation_direction)

    if evidence.entered_success:
        return TeacherConsistencyDecision(P_LABEL, "consistent", info, 1, 1.0, True)

    authoritative_negative = bool(
        evidence.terminal_failure
        or evidence.lost_grasp
        or evidence.lost_goal_contact
        or (use_relation_direction and evidence.relation_direction < 0)
    )
    if authoritative_negative:
        consistency = "consistent" if info <= 0 else "contradicted"
        return TeacherConsistencyDecision(N_LABEL, consistency, info, -1, 1.0, True)

    if info > 0:
        consistency = "consistent" if physical > 0 else "uncontradicted"
        confidence = _confidence(
            score,
            deadband,
            consistency="consistent" if physical > 0 else "insufficient",
            authoritative=False,
        )
        # A lack of event evidence should not collapse confidence to near zero
        # in this veto-style ablation.  Preserve information strength while
        # still recording that physical support was absent.
        if physical == 0:
            scale = max(float(deadband), 1e-6)
            confidence = min(max(abs(float(score)) / (3.0 * scale), 0.5), 1.0)
        return TeacherConsistencyDecision(P_LABEL, consistency, info, physical, confidence, False)

    if info < 0:
        if physical > 0:
            return TeacherConsistencyDecision(U_LABEL, "contradicted", info, physical, 0.0, False)
        consistency = "consistent" if physical < 0 else "uncontradicted"
        scale = max(float(deadband), 1e-6)
        confidence = min(max(abs(float(score)) / (3.0 * scale), 0.5), 1.0)
        return TeacherConsistencyDecision(N_LABEL, consistency, info, physical, confidence, False)

    return TeacherConsistencyDecision(U_LABEL, "insufficient", 0, physical, 0.0, False)

