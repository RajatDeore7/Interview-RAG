from datetime import datetime


class InterviewManager:
    def __init__(self, interview_timing: int):
        self.start_time = datetime.now()
        self.total_time = interview_timing
        self.phase = "resume"

        # Use your original ratios
        self.ratios = {
            "resume": 0.25,
            "role based technical": 0.45,
            "behavioral": 0.20,
            "wrapup": 0.10,
        }

        # Compute cutoffs based on ratios
        cutoff_resume = self.total_time * self.ratios["resume"]
        cutoff_technical = cutoff_resume + self.total_time * self.ratios["role based technical"]
        cutoff_behavioral = cutoff_technical + self.total_time * self.ratios["behavioral"]
        cutoff_wrapup = cutoff_behavioral + self.total_time * self.ratios["wrapup"]

        self.cutoffs = {
            "resume": cutoff_resume,
            "role based technical": cutoff_technical,
            "behavioral": cutoff_behavioral,
            "wrapup": cutoff_wrapup
        }

    def get_elapsed_minutes(self):
        return (datetime.now() - self.start_time).seconds / 60.0

    def update_phase(self):
        minutes = self.get_elapsed_minutes()
        if minutes < self.cutoffs["resume"]:
            self.phase = "resume"
        elif minutes < self.cutoffs["role based technical"]:
            self.phase = "role based technical"
        elif minutes < self.cutoffs["behavioral"]:
            self.phase = "behavioral"
        elif minutes < self.cutoffs["wrapup"]:
            self.phase = "wrapup"
        else:
            self.phase = "closed"

    def is_interview_over(self):
        return self.get_elapsed_minutes() >= (self.total_time - 0.0333)

    def get_current_phase(self):
        self.update_phase()
        return self.phase