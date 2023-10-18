import math

import numpy as np


def speed_to_angle(speed_kmh: float, duration_s: int = 60) -> float:
    """Convert speed in km/h to angle (to horizon) in a square image

    Args:
        speed_kmh (float): speed in km/h

    Returns:
        float: angle with horizontal in degree
    """
    DISTANCE_M = 800
    speed_ms = speed_kmh * 1000 / 3600
    time_s = DISTANCE_M / speed_ms
    angle_radian = np.arctan(time_s / duration_s)
    angle_degree = math.degrees(angle_radian)
    return angle_degree


def time_to_speed(time_s: float) -> float:
    """Convert time to finish the distance to speed in km/h

    Args:
        time_s (float): Time to finish the length, in seconds

    Returns:
        float: speed in km/h
    """
    DISTANCE_M = 800
    speed_ms = DISTANCE_M / time_s
    speed_kmh = speed_ms / 1000 * 3600
    return speed_kmh
