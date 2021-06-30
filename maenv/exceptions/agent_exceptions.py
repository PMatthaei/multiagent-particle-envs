from core import Agent


class NoTargetFoundError(Exception):
    def __init__(self):
        super().__init__("Received attack action without target set. Check correctness")


class IllegalTargetError(Exception):
    def __init__(self, agent: Agent):
        super().__init__(
            "The chosen target with id {0} can not be attacked/healed by agent with id {1}."
                .format(agent.target_id, agent.id))


class OverhealError(Exception):
    def __init__(self, agent: Agent):
        super().__init__(
            "The chosen target with id {0} can not be overhealed by agent with id {1}."
                .format(agent.target_id, agent.id))
