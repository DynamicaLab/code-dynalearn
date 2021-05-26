from datetime import datetime

from .logger import Logger


class TimeLogger(Logger):
    def __init__(self):
        Logger.__init__(self)
        self.begin = None
        self.update = None
        self.end = None

    def on_task_begin(self):
        self.begin = datetime.now()
        self.update = self.begin
        self.log["begin"] = self.begin.strftime("%Y-%m-%d %H:%M:%S")

    def on_task_end(self):
        self.end = datetime.now()
        self.log["end"] = self.begin.strftime("%Y-%m-%d %H:%M:%S")
        days, hours, mins, secs = self.format_diff(self.begin, self.end)
        self.log["time"] = f"{days:0=2d}-{hours:0=2d}:{mins:0=2d}:{secs:0=2d}"
        self.log["total"] = self.format_diff(self.begin, self.end, to_sec=True)

    def on_task_update(self, stepname=None):
        stepname = stepname or "update"
        now = datetime.now()
        dt = self.format_diff(self.update, now, to_sec=True)
        if f"time-{stepname}" in self.log:
            self.log[f"time-{stepname}"].append(dt)
        else:
            self.log[f"time-{stepname}"] = [dt]
        self.update = now

    def format_diff(self, t0, t1, to_sec=False):
        dt = t1 - t0
        days = dt.days
        hours, r = divmod(dt.seconds, 60 * 60)
        mins, r = divmod(r, 60)
        secs = r
        if to_sec:
            return ((days * 24 + hours) * 60 + mins) * 60 + secs
        else:
            return days, hours, mins, secs
