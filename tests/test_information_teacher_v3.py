import torch

from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    VisualPointwiseInformationCritic,
    balanced_density_ratio_loss,
    compute_information_teacher_evidence,
    conditional_action_information_loss,
    directional_information_score,
    visual_information_loss,
)


def test_balanced_density_ratio_loss_rewards_separated_logits():
    ambiguous = balanced_density_ratio_loss(torch.zeros(8), torch.zeros(8))
    separated = balanced_density_ratio_loss(torch.full((8,), 3.0), torch.full((8,), -3.0))
    assert separated < ambiguous


def test_visual_and_conditional_action_critics_contracts():
    torch.manual_seed(0)
    visual = VisualPointwiseInformationCritic(visual_dim=16, goal_dim=12, projection_dim=8, hidden_dim=16)
    action = ConditionalActionInformationCritic(
        action_dim=10, visual_dim=16, goal_dim=12, projection_dim=8, hidden_dim=16
    )
    v = torch.randn(5, 16)
    g = torch.randn(5, 12)
    a = torch.randn(5, 10)
    assert visual(v, g).shape == (5,)
    assert action(a, v, g).shape == (5,)
    assert visual_information_loss(visual, v, g, g.roll(1, 0)).ndim == 0
    assert conditional_action_information_loss(action, a, v, g, a.roll(1, 0)).ndim == 0


def test_directional_information_score_matches_v3_formula():
    phi_t = torch.tensor([1.0, 2.0])
    phi_t1 = torch.tensor([1.5, 2.25])
    psi = torch.tensor([0.2, -0.1])
    actual = directional_information_score(phi_t, phi_t1, psi, gamma=0.9, beta=0.5)
    expected = 0.9 * phi_t1 - phi_t + 0.5 * psi
    torch.testing.assert_close(actual, expected)


def test_information_teacher_evidence_keeps_action_separate_from_state_potential():
    torch.manual_seed(1)
    visual = VisualPointwiseInformationCritic(8, 8, projection_dim=4, hidden_dim=8)
    action = ConditionalActionInformationCritic(6, 8, 8, projection_dim=4, hidden_dim=8)
    v0 = torch.randn(3, 8)
    v1 = torch.randn(3, 8)
    g = torch.randn(3, 8)
    a = torch.randn(3, 6)
    evidence = compute_information_teacher_evidence(
        visual,
        action,
        visual_t=v0,
        visual_t1=v1,
        goal=g,
        action_t=a,
        gamma=0.95,
        beta=0.7,
    )
    torch.testing.assert_close(evidence.delta_phi_visual, 0.95 * evidence.phi_visual_t1 - evidence.phi_visual_t)
    torch.testing.assert_close(
        evidence.directional_score,
        evidence.delta_phi_visual + 0.7 * evidence.conditional_action_information,
    )
