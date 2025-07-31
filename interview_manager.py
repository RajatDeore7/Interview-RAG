from datetime import datetime


class InterviewManager:
    def __init__(self):
        self.start_time = datetime.now()
        self.phase = "resume"

    def get_elapsed_minutes(self):
        return (datetime.now() - self.start_time).seconds // 60

    def update_phase(self):
        minutes = self.get_elapsed_minutes()
        if minutes < 2:
            self.phase = "resume"
        elif minutes < 10:
            self.phase = "role based technical"
        elif minutes < 15:
            self.phase = "behavioral"
        else:
            self.phase = "wrapup"

    def get_current_phase(self):
        self.update_phase()
        return self.phase
