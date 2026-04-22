"""
Quiz: Social Media Engagement Engine Testing

Objective: Write a comprehensive suite of unit tests for an engagement tracking system and automate their execution using a Continuous Integration (CI) pipeline.
UPLOAD THE COVERAGE REPORT
USE MY CODE FOR WORKFLOW YAML (IF YOU HAVE TO CHANGE IT A BIT PLEASE DO IT)
"""


class EngagementEngine:
    def __init__(self, user_handle, verified=False):
        self.user_handle = user_handle
        self.score = 0.0
        self.verified = verified

    def process_interaction(self, itype, count=1):
        if count < 0: raise ValueError("Negative count")
        weights = {"like": 1, "comment": 5, "share": 10}
        if itype not in weights: return False
        
        points = weights[itype] * count
        if self.verified: points *= 1.5
        self.score += points
        return True

    def get_tier(self):
        if self.score < 100: return "Newbie"
        if self.score <= 1000: return "Influencer"
        return "Icon"

    def apply_penalty(self, report_count):
        if report_count > 10: self.verified = False
        reduction = self.score * (0.20 * report_count)
        self.score = max(0, self.score - reduction)