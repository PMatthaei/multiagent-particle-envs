class TooManyWinners(Exception):
    def __init__(self, winner):
        super().__init__("Only one team can win the game. Teams with ids {0} seem to have won.".format(winner))


class ActionCountMismatch(Exception):
    def __init__(self, expected, served):
        super().__init__(f"The environment expected {expected} instead of {served} action ids.")
