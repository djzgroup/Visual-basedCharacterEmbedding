import time


class Timer:
    def __init__(self):
        self._start_time = 0
        self._end_time = 0

    def tick(self):
        self._start_time = time.time()

    def tock(self):
        self._end_time = time.time()

    def last_time(self):
        return self._end_time - self._end_time