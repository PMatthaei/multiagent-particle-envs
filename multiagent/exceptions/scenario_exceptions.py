class SymmetricScenarioSetupError(Exception):
    def __init__(self, n_agents, n_teams):
        super().__init__("Cannot symmetric teams with {} agents and {} teams. "
                         "Consider the asymmetric scenario or build teams with equal size.".format(n_agents, n_teams))
