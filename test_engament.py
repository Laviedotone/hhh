"""
Comprehensive unit tests for the Social Media Engagement Engine.
Run with: pytest test_engament.py -v --tb=short
For coverage: pytest test_engament.py --cov=engament --cov-report=term-missing
"""

import pytest
from engament import EngagementEngine


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def basic_user():
    """Standard unverified user."""
    return EngagementEngine(user_handle="testuser")


@pytest.fixture
def verified_user():
    """Verified user (1.5× point multiplier)."""
    return EngagementEngine(user_handle="verifieduser", verified=True)


# ─────────────────────────────────────────────
# __init__ / construction
# ─────────────────────────────────────────────

class TestInit:
    def test_handle_stored(self, basic_user):
        assert basic_user.user_handle == "testuser"

    def test_initial_score_is_zero(self, basic_user):
        assert basic_user.score == 0.0

    def test_default_not_verified(self, basic_user):
        assert basic_user.verified is False

    def test_verified_flag_set(self, verified_user):
        assert verified_user.verified is True

    def test_score_is_float(self, basic_user):
        assert isinstance(basic_user.score, float)


# ─────────────────────────────────────────────
# process_interaction — happy paths
# ─────────────────────────────────────────────

class TestProcessInteractionValid:
    @pytest.mark.parametrize("itype,count,expected", [
        ("like",    1,  1.0),
        ("like",   10, 10.0),
        ("comment", 1,  5.0),
        ("comment", 3, 15.0),
        ("share",   1, 10.0),
        ("share",   4, 40.0),
    ])
    def test_basic_scoring(self, basic_user, itype, count, expected):
        basic_user.process_interaction(itype, count)
        assert basic_user.score == expected

    def test_returns_true_on_valid_type(self, basic_user):
        assert basic_user.process_interaction("like") is True

    def test_default_count_is_one(self, basic_user):
        basic_user.process_interaction("like")
        assert basic_user.score == 1.0

    def test_score_accumulates_across_calls(self, basic_user):
        basic_user.process_interaction("like", 5)    # +5
        basic_user.process_interaction("comment", 2) # +10
        basic_user.process_interaction("share", 1)   # +10
        assert basic_user.score == 25.0

    def test_zero_count_adds_nothing(self, basic_user):
        basic_user.process_interaction("like", 0)
        assert basic_user.score == 0.0

    def test_zero_count_returns_true(self, basic_user):
        assert basic_user.process_interaction("share", 0) is True


# ─────────────────────────────────────────────
# process_interaction — verified multiplier
# ─────────────────────────────────────────────

class TestVerifiedMultiplier:
    @pytest.mark.parametrize("itype,count,expected", [
        ("like",    1,  1.5),
        ("comment", 2, 15.0),
        ("share",   1, 15.0),
    ])
    def test_verified_1_5x(self, verified_user, itype, count, expected):
        verified_user.process_interaction(itype, count)
        assert verified_user.score == pytest.approx(expected)

    def test_verified_accumulates_correctly(self, verified_user):
        verified_user.process_interaction("like", 10)    # 10 * 1.5 = 15
        verified_user.process_interaction("comment", 2)  # 10 * 1.5 = 15
        assert verified_user.score == pytest.approx(30.0)


# ─────────────────────────────────────────────
# process_interaction — error & edge cases
# ─────────────────────────────────────────────

class TestProcessInteractionErrors:
    def test_unknown_type_returns_false(self, basic_user):
        assert basic_user.process_interaction("retweet") is False

    def test_unknown_type_does_not_change_score(self, basic_user):
        basic_user.process_interaction("retweet", 5)
        assert basic_user.score == 0.0

    def test_empty_string_returns_false(self, basic_user):
        assert basic_user.process_interaction("") is False

    def test_negative_count_raises_value_error(self, basic_user):
        with pytest.raises(ValueError, match="Negative count"):
            basic_user.process_interaction("like", -1)

    def test_case_sensitive_type(self, basic_user):
        """'Like' (capital L) is not a valid interaction type."""
        assert basic_user.process_interaction("Like") is False


# ─────────────────────────────────────────────
# get_tier
# ─────────────────────────────────────────────

class TestGetTier:
    def test_newbie_at_zero(self, basic_user):
        assert basic_user.get_tier() == "Newbie"

    def test_newbie_just_below_100(self, basic_user):
        basic_user.score = 99.9
        assert basic_user.get_tier() == "Newbie"

    def test_influencer_at_exactly_100(self, basic_user):
        basic_user.score = 100.0
        assert basic_user.get_tier() == "Influencer"

    def test_influencer_at_500(self, basic_user):
        basic_user.score = 500.0
        assert basic_user.get_tier() == "Influencer"

    def test_influencer_at_exactly_1000(self, basic_user):
        basic_user.score = 1000.0
        assert basic_user.get_tier() == "Influencer"

    def test_icon_just_above_1000(self, basic_user):
        basic_user.score = 1000.1
        assert basic_user.get_tier() == "Icon"

    def test_icon_at_large_score(self, basic_user):
        basic_user.score = 999_999.0
        assert basic_user.get_tier() == "Icon"

    def test_tier_updates_after_interactions(self, basic_user):
        assert basic_user.get_tier() == "Newbie"
        basic_user.process_interaction("share", 10)  # score = 100
        assert basic_user.get_tier() == "Influencer"
        basic_user.process_interaction("share", 91)  # score = 1010
        assert basic_user.get_tier() == "Icon"


# ─────────────────────────────────────────────
# apply_penalty
# ─────────────────────────────────────────────

class TestApplyPenalty:
    def test_no_penalty_at_threshold(self, basic_user):
        """Exactly 10 reports: no verified strip, score unchanged (0% reduction)."""
        basic_user.score = 500.0
        basic_user.verified = True
        basic_user.apply_penalty(10)
        # reduction = 500 * (0.20 * 10) = 500 * 2.0 = 1000 → clamped to 0
        # But verified should NOT be stripped at exactly 10
        assert basic_user.verified is True

    def test_verified_stripped_above_10_reports(self, verified_user):
        verified_user.score = 200.0
        verified_user.apply_penalty(11)
        assert verified_user.verified is False

    def test_score_reduced_by_percentage(self, basic_user):
        basic_user.score = 500.0
        basic_user.apply_penalty(1)   # reduction = 500 * 0.20 = 100
        assert basic_user.score == pytest.approx(400.0)

    def test_score_clamped_at_zero(self, basic_user):
        basic_user.score = 100.0
        basic_user.apply_penalty(10)  # would reduce by 200%; clamped to 0
        assert basic_user.score == 0.0

    def test_score_already_zero_stays_zero(self, basic_user):
        basic_user.apply_penalty(5)
        assert basic_user.score == 0.0

    def test_penalty_does_not_go_negative(self, basic_user):
        basic_user.score = 50.0
        basic_user.apply_penalty(20)
        assert basic_user.score >= 0.0

    def test_penalty_with_zero_reports(self, basic_user):
        basic_user.score = 300.0
        basic_user.apply_penalty(0)
        assert basic_user.score == pytest.approx(300.0)
        assert basic_user.verified is False  # 0 is not > 10


# ─────────────────────────────────────────────
# Integration scenarios
# ─────────────────────────────────────────────

class TestIntegrationScenarios:
    def test_full_lifecycle_unverified(self):
        engine = EngagementEngine("casual_user")
        engine.process_interaction("like", 50)      # 50
        engine.process_interaction("comment", 10)   # +50 → 100
        assert engine.get_tier() == "Influencer"
        engine.apply_penalty(2)                     # 100 * 0.4 = 40 reduction → 60
        assert engine.score == pytest.approx(60.0)
        assert engine.get_tier() == "Newbie"

    def test_full_lifecycle_verified_then_penalized(self):
        engine = EngagementEngine("star", verified=True)
        engine.process_interaction("share", 70)     # 70 * 10 * 1.5 = 1050
        assert engine.get_tier() == "Icon"
        engine.apply_penalty(11)                    # strips verified
        assert engine.verified is False
        # reduction = 1050 * (0.20 * 11) = 1050 * 2.2 = 2310 → clamped to 0
        assert engine.score == 0.0
        assert engine.get_tier() == "Newbie"

    def test_invalid_interactions_do_not_corrupt_state(self):
        engine = EngagementEngine("clean")
        engine.process_interaction("like", 5)       # 5
        engine.process_interaction("hack", 100)     # ignored
        assert engine.score == pytest.approx(5.0)
        assert engine.get_tier() == "Newbie"