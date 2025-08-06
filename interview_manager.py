from datetime import datetime


class InterviewManager:
    def __init__(self):
        self.start_time = datetime.now()
        self.phase = "resume"

    def get_elapsed_minutes(self):
        return (datetime.now() - self.start_time).seconds // 60

    def update_phase(self):
        minutes = self.get_elapsed_minutes()
        if minutes < 3:
            self.phase = "resume"
        elif minutes < 6:
            self.phase = "role based technical"
        elif minutes < 9:
            self.phase = "behavioral"
        else:
            self.phase = "wrapup"

    def is_interview_over(self):
        return self.get_elapsed_minutes() > 10  # or any limit you want

    def get_current_phase(self):
        self.update_phase()
        return self.phase
