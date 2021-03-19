import select
import sys
import threading

INPUT_TIME_OUT = 1


class EnvControls(threading.Thread):
    def __init__(self, env=None):
        super(EnvControls, self).__init__()
        self._stop_event = threading.Event()
        self.env = env
        return

    def run(self):
        print("Press 'r' and confirm with 'Enter' to turn on/off rendering. Default=" + str(self.env.headless))

        while not self.stopped():
            i, o, e = select.select([sys.stdin], [], [], INPUT_TIME_OUT)
            if i:
                value = sys.stdin.readline().strip()
                if value == 'r':
                    self.env.headless = not self.env.headless

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        is_set = self._stop_event.is_set()
        return is_set
