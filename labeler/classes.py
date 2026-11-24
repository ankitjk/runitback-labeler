"""The scene classes and the rule that turns two player states into one class.

The rule is the one in CLASSES.md ("How a frame is classified"). It lives in
code so it is applied the same way every time and can be tested.
"""

CLASSES = (
    "non_gameplay",
    "both_grounded",
    "one_airborne",
    "both_airborne",
    "one_offstage",
    "both_offstage",
    "one_on_ledge",
    "both_on_ledge",
    "one_knocked_down",
    "both_knocked_down",
    "respawning",
)

# Highest rank first.
PLAYER_STATES = ("respawning", "offstage", "on_ledge", "knocked_down", "airborne", "grounded")

SOURCES = ("model", "llm", "human")


def scene_from_states(gameplay: bool, state_a: str | None = None, state_b: str | None = None) -> str:
    """Return the scene class for one frame, given what each player is doing."""
    if not gameplay:
        return "non_gameplay"

    for state in (state_a, state_b):
        if state not in PLAYER_STATES:
            raise ValueError(f"unknown player state {state!r}; expected one of {PLAYER_STATES}")

    top = min(state_a, state_b, key=PLAYER_STATES.index)
    if top == "respawning":
        return "respawning"
    # "grounded" is the lowest rank, so it can only be top when both are grounded.
    return f"both_{top}" if state_a == state_b else f"one_{top}"
