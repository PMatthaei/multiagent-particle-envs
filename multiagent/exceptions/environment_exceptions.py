class TooManyWinners(Exception):
    def __init__(self, winner):
        super().__init__("Only one team can win the game. Teams with ids {0} seem to have won.".format(winner))


class MissingActions(Exception):
    def __init__(self):
        super().__init__("The environment expected more action ids than were served into step()")
