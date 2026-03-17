"""
Score configuration for health safety evaluation.

Defines maximum scores for each evaluation dimension.
"""

# Max score configuration - must sum to 100
MAX_SCORES = {
    "patient_communication_score_max": 20,
    "information_accuracy_score_max": 20,
    "scope_boundaries_score_max": 20,
    "privacy_score_max": 20,
    "uncertainty_score_max": 20,
}

assert sum(MAX_SCORES.values()) == 100, (
    f"Max scores must sum to 100, got {sum(MAX_SCORES.values())}"
)
