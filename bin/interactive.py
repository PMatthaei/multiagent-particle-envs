import threading


class EnvInputListener(threading.Thread):
    def __init__(self, args=(), kwargs=None, ):
        super(EnvInputListener, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.env = self.kwargs['env']
        return

    def run(self):

        print("Press 'r' and confirm with 'Enter' to turn on/off rendering. Default=" + str(self.env.headless))
        while True:
            value = input()
            if value == 'r':
                self.env.headless = not self.env.headless
