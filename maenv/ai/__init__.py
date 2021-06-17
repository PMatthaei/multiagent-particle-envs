from .basic_ai import BasicScriptedAI
from .role_focus_ai import FocusScriptedAI

REGISTRY = {
    "basic": BasicScriptedAI,
    "focus": FocusScriptedAI
}
