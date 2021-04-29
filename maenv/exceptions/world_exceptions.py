class NoTeamFoundError(Exception):
    def __init__(self, tid):
        super().__init__("Not team was found with the given id: {}".format(tid))
