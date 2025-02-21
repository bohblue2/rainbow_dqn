from enum import IntEnum

class Transition(IntEnum):
    STATE = 0
    ACTION = 1
    REWARD = 2
    NEXT_STATE = 3
    DONE = 4

