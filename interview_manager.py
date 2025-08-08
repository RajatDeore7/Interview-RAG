from datetime import datetime


class InterviewManager:
    def __init__(self):
        self.start_time = datetime.now()
        self.phase = "resume"

    def get_elapsed_minutes(self):
        return (datetime.now() - self.start_time).seconds // 60

    def update_phase(self):
        minutes = self.get_elapsed_minutes()
        if minutes < 1:
            self.phase = "resume"
        elif minutes < 2:
            self.phase = "role based technical"
        elif minutes < 4:
            self.phase = "behavioral"
        elif minutes < 4.55:
            self.phase = "wrapup"
        else:
            self.phase = "closed"

    def is_interview_over(self):
        return self.get_elapsed_minutes() > 5  # or any limit you want

    def get_current_phase(self):
        self.update_phase()
        return self.phase
