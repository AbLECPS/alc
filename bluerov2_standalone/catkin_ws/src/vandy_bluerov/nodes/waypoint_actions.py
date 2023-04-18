#!/usr/bin/env python

from enum import IntEnum

class WaypointAction(IntEnum):
    PASS = 0
    LOINTER_N = 1

class WaypointParams(IntEnum):
    X = 0
    Y = 1
    Z = 2
    SPEED = 3
    ACTION = 4
    P0 = 5 # N for LOITER_N
    P1 = 6 # Radius for LOITER_N