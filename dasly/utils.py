"""Provides convenient utility functions for other modules"""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-08-23'


import re
import uuid
from datetime import datetime, timedelta
from typing import Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import NullPool


def extract_dt(path: str) -> str:
    """Extract the date and time from the input path with format ending with
    /YYYYMMDD/dphi/HHMMSS.hdf5. Output format: 'YYYYMMDD HHMMSS'

    Args:
        path (str): Input path.

    Returns:
        str: Date and time extracted from the input path.
    """
    # Define the regular expression pattern
    pattern = r"/(\d{8})/dphi/(\d{6})\.hdf5$"

    # Search for the pattern in the input string
    match = re.search(pattern, path)

    # Extract the matched groups
    date_part = match.group(1)
    time_part = match.group(2)

    # Combine date and time parts
    date_time = f"{date_part} {time_part}"
    return date_time


def add_subtract_dt(dt: str, x: int, format: str = '%Y%m%d %H%M%S') -> str:
    """Add or subtract x seconds to time t having format 'YYYYMMDD HHMMSS'.

    Args:
        dt (str): Input datetime string.
        x (int): Number of seconds to add or subtract.
        format (str, optional): Format of the input datetime string.
            Defaults to '%Y%m%d %H%M%S'.

    Returns:
        str: New datetime string after adding or subtracting x seconds.
    """
    # Convert input string to datetime object
    input_dt = datetime.strptime(dt, format)
    # Add x seconds to the datetime object
    new_dt = input_dt + timedelta(seconds=x)
    # Format the new datetime object back into the same string format
    new_dt_str = new_dt.strftime(format)
    return new_dt_str


def create_connection_string(
    endpoint: str,
    database: str,
    db_username: str,
    db_password: str,
    database_type: str = 'postgresql',
    dbapi: str = 'psycopg2',
    port: int = 5432
) -> str:
    """Create the connection string for the SQL database.

    Args:
        endpoint (str): Database endpoint.
        database (str): Database name.
        db_username (str): Database username.
        db_password (str): Database password.
        database_type (str, optional): Database type. Defaults to 'postgresql'.
        dbapi (str, optional): Database API. Defaults to 'psycopg2'.
        port (int, optional): Database port. Defaults to 5432.

    Returns:
        str: Connection string for the SQL database.
    """
    connection_string = (
        f'{database_type}+{dbapi}://{db_username}:{db_password}'
        + f'@{endpoint}:{port}/{database}'
    )
    return connection_string


def read_sql(query: str, connection_string: str) -> pd.DataFrame:
    """Load the data from SQL database.

    Args:
        query (str): SQL query to execute.
        connection_string (str): Connection string to the database.

    Returns:
        pd.DataFrame: Data loaded from the SQL database.
    """
    # Create an engine
    engine = create_engine(connection_string, poolclass=NullPool)
    # Execute a SQL query
    df = pd.read_sql(query, engine)
    return df


def write_sql(
    df: pd.DataFrame,
    database_table: str,
    connection_string: str,
    created_at: bool = True,
) -> None:
    """Write the data frame to SQL database.

    Args:
        df (pd.DataFrame): Data to write to the database.
        database_table (str): Table name in the database.
        connection_string (str): Connection string to the database.
        created_at (bool, optional): Add a 'created_at' column to the data.
            UTC timezone. Defaults to True.
    """
    # Create an engine
    engine = create_engine(connection_string, poolclass=NullPool)
    # Execute a SQL query
    if created_at:
        df.insert(0, 'created_at', pd.Timestamp.now(tz='UTC'))
    df.to_sql(database_table, engine, if_exists='append', index=False)


def check_table_exists(
    database_table: str,
    connection_string: str
) -> bool:
    """Check if the table exists in the database.

    Args:
        database_table (str): Table name to check.
        connection_string (str): Connection string to the database.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    engine = create_engine(connection_string, poolclass=NullPool)
    inspector = inspect(engine)
    return inspector.has_table(database_table)


def assign_id(
    df: pd.DataFrame,
    id_col: str = 'id',
) -> pd.DataFrame:
    """Assign unique ID to each row in the data frame.

    Args:
        df (pd.DataFrame): Input data frame.
        id_col (str): Name of the ID column. Default is 'id'.

    Returns:
        pd.DataFrame: Data frame with unique ID assigned.
    """
    uuid_list = [str(uuid.uuid4()) for _ in range(len(df))]
    df.insert(0, id_col, uuid_list)
    return df


def reassign_id(
    previous_lines_id: np.ndarray[Union[str, int]],
    current_lines_id: np.ndarray[Union[str, int]],
    dist_mat: np.ndarray[float],
    threshold: float
) -> np.ndarray[Union[str, int]]:
    """Reassign the line ID based on the minimum distance between the previous
    and new lines. If the distance is below the threshold, the new line is
    assigned the ID of the previous nearest line. If the distance is above the
    threshold, the new line ID is kept.

    Args:
        previous_lines_id (np.ndarray[Union[str, int]],): Previous line IDs.
            Shape (N,).
        current_lines_id (np.ndarray[Union[str, int]],): Current line IDs.
            Shape (M,).
        dist_mat (np.ndarray[float]): Distance matrix between the previous
            lines and new lines. Shape (N, M)
        threshold (float): Threshold to consider the distance as a match.

    Returns:
        np.ndarray[Union[str, int]],: New line IDs.
    """
    # Find the indices of the minimum distances
    min_indices = np.argmin(dist_mat, axis=0)
    # Find the minimum distances for each point in b
    min_distances = np.min(dist_mat, axis=0)
    # Create a mask for distances below the threshold
    mask = min_distances < threshold
    # Create an array to hold the new names
    lines_id_resigned = np.where(
        mask, previous_lines_id[min_indices], current_lines_id)
    return lines_id_resigned


def interpolate_timestamps(
    s1: int,
    t1: Union[pd.Timestamp, datetime],
    s2: int,
    t2: Union[pd.Timestamp, datetime],
) -> np.ndarray[np.datetime64]:
    """Get the timestamps for a segment of a line segment defined by (s1, t1)
    and (s2, t2).

    Args:
        s1 (int): Start index of the line segment.
        t1 (Union[pd.Timestamp, datetime]): Start timestamp of the line seg.
        s2 (int): End index of the line segment.
        t2 (Union[pd.Timestamp, datetime]): End timestamp of the line segment.

    Returns:
        np.ndarray[np.datetime64]: Timestamps for the line segment between s1
            and s2.
    """
    # Calculate the number of space indices
    num_space_indices = np.abs(s2 - s1) + 1

    # Create an array of timestamps linearly spaced from t11 to t12
    timestamps = pd.date_range(start=t1, end=t2, periods=num_space_indices)

    return timestamps.to_numpy()


def calculate_speed(
    lines: np.ndarray[Union[float, pd.Timestamp, datetime]]
) -> Union[float, np.ndarray[float]]:
    """Calculate the speed of line segments defined by (s1, t1, s2, t2). This
    is for spatio-temporal data where the x-axis is space and the y-axis
    is time.

    Args:
        lines (np.ndarray[Union[float, pd.Timestamp, datetime]]): Array of line
            segments. Shape (N, 4) where N is the number of line segments, or
            (4,) for a single line. Each line segment is defined by:
            - s1 (float): Start index of the line segment.
            - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (float): End index of the line segment.
            - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        Union[float, np.ndarray[float]]: If the input is a single segment,
            returns a float. If the input is an array of segments, returns an
            array of slopes.
    """
    single_line = False
    # If input is a single line segment with shape (4,)
    if lines.ndim == 1 and lines.shape[0] == 4:
        lines = lines.reshape(1, 4)
        single_line = True

    # Extracting s1, t1, s2, t2
    s1 = lines[:, 0]
    t1 = lines[:, 1]
    s2 = lines[:, 2]
    t2 = lines[:, 3]

    # Calculate the time difference between t2 and t1
    time_diff = (t2 - t1)
    if isinstance(time_diff[0], timedelta):
        time_diff = time_diff.astype('timedelta64[s]').astype(float)

    # Calculate the space difference between s2 and s1
    space_diff = s2 - s1

    # Calculate the speed of the line segment
    speed = np.divide(
        space_diff,
        time_diff,
        where=time_diff != 0,
        out=np.full_like(space_diff, np.inf)
    )
    if single_line:
        return speed[0]

    return speed.astype(float)


def calculate_slope(
    lines: np.ndarray[Union[float, pd.Timestamp, datetime]]
) -> Union[float, np.ndarray[float]]:
    """Calculate the slope of line segments defined by (s1, t1, s2, t2). This
    is for spatio-temporal data where the x-axis is space and the y-axis
    is time.

    Args:
        lines (np.ndarray[Union[float, pd.Timestamp, datetime]]): Array of line
            segments. Shape (N, 4) where N is the number of line segments, or
            (4,) for a single line. Each line segment is defined by:
            - s1 (float): Start index of the line segment.
            - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (float): End index of the line segment.
            - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        Union[float, np.ndarray[float]]: If the input is a single segment,
            returns a float. If the input is an array of segments, returns an
            array of slopes.
    """
    speed = calculate_speed(lines)
    if isinstance(speed, float):
        if speed == 0:
            return np.inf
        return 1 / speed
    with np.errstate(divide='ignore'):
        return np.reciprocal(speed)


def reorder_coordinates(
    lines: np.ndarray[Union[float, pd.Timestamp, datetime]]
) -> np.ndarray[Union[float, pd.Timestamp, datetime]]:
    """Reorder the 2 endpoints of lines segments so that y1 <= y2. If y1 == y2,
    ensure x1 <= x2.

    Args:
        lines (np.ndarray[Union[float, pd.Timestamp, datetime]]): Array of line
            segments. Shape (N, 4) where N is the number of line segments, or
            (4,) for a single line. Each line segment is defined by:
            - s1 (float): Start index of the line segment.
            - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (float): End index of the line segment.
            - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray[Union[float, pd.Timestamp, datetime]]: Reordered array of
            line segments.
    """
    lines_cp = lines.copy()

    # If input is a 1D array of shape (4,), reshape it to (1, 4)
    single_line = False
    if lines.ndim == 1 and lines.shape[0] == 4:
        lines_cp = lines_cp.reshape(1, 4)
        single_line = True

    # Extract the coordinates
    x1 = lines_cp[:, 0]
    y1 = lines_cp[:, 1]
    x2 = lines_cp[:, 2]
    y2 = lines_cp[:, 3]

    # Create a mask where y1 > y2
    mask_y = y1 > y2

    # Swap coordinates where the mask is True (y1 > y2)
    x1[mask_y], x2[mask_y] = x2[mask_y], x1[mask_y]
    y1[mask_y], y2[mask_y] = y2[mask_y], y1[mask_y]

    # Create a mask where y1 == y2 and x1 > x2
    mask_x = (y1 == y2) & (x1 > x2)

    # Swap coordinates where the mask is True (y1 == y2 and x1 > x2)
    x1[mask_x], x2[mask_x] = x2[mask_x], x1[mask_x]
    y1[mask_x], y2[mask_x] = y2[mask_x], y1[mask_x]

    # Combine the coordinates back into the lines array
    lines_cp = np.stack((x1, y1, x2, y2), axis=1)

    # If it was a single line segment, reshape the output back to (4,)
    if single_line:
        return lines_cp.reshape(4,)

    return lines_cp


def intersect_space(
    lines1: np.ndarray,  # Shape (N, 4) or (4,)
    lines2: np.ndarray   # Shape (M, 4) or (4,)
) -> np.ndarray:
    """Calculate the space intersection limits of line segments.

    Args:
        lines1 (np.ndarray): Line segments 1. Shape (N, 4) or (4,), where each
            row is defined by:
            - s1 (float): Start index of the line segment.
            - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (float): End index of the line segment.
            - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.
        lines2 (np.ndarray): Line segments 2. Shape (M, 4) or (4,), where each
            row is defined by:
            - s1 (float): Start index of the line segment.
            - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (float): End index of the line segment.
            - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray: Space intersection limits. Shape (N, M, 2) or (2,) if input
            shapes were (4,).
    """
    # Ensure line1 and line2 are at least 2D (shape (N, 4) and (M, 4))
    lines1 = np.atleast_2d(lines1)
    lines2 = np.atleast_2d(lines2)

    # Extract the relevant parts of the line segments
    s11, s12 = lines1[:, 0], lines1[:, 2]
    s21, s22 = lines2[:, 0], lines2[:, 2]

    # Reshape arrays for broadcasting if needed
    s11 = s11[:, np.newaxis]  # Shape (N, 1)
    s12 = s12[:, np.newaxis]  # Shape (N, 1)
    s21 = s21[np.newaxis, :]  # Shape (1, M)
    s22 = s22[np.newaxis, :]  # Shape (1, M)

    # Calculate the min and max for the ranges
    range1_min = np.minimum(s11, s12)  # Shape (N, 1)
    range1_max = np.maximum(s11, s12)  # Shape (N, 1)
    range2_min = np.minimum(s21, s22)  # Shape (1, M)
    range2_max = np.maximum(s21, s22)  # Shape (1, M)

    # Calculate the intersection of the two ranges
    lim1 = np.maximum(range1_min, range2_min)  # Shape (N, M)
    lim2 = np.minimum(range1_max, range2_max)  # Shape (N, M)

    # Stack the results along the last dimension to get the final shape (N, M, 2)
    result = np.stack([lim1, lim2], axis=-1)  # Shape (N, M, 2)

    # If original inputs were 1D, return a 1D result
    if lines1.shape[0] == 1 and lines2.shape[0] == 1:
        result = result.squeeze(axis=(0, 1))  # Shape (2,)

    return result


# def calculate_distance(
#     lines1: np.ndarray,  # Shape (N, 4) or (4,)
#     lines2: np.ndarray   # Shape (M, 4) or (4,)
# ) -> np.ndarray:
#     """Calculate time gap among line segments.

#     Args:
#         lines1 (np.ndarray): Line segments 1. Shape (N, 4) or (4,), where each
#             row is defined by:
#             - s1 (float): Start index of the line segment.
#             - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
#             - s2 (float): End index of the line segment.
#             - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.
#         lines2 (np.ndarray): Line segments 2. Shape (M, 4) or (4,), where each
#             row is defined by:
#             - s1 (float): Start index of the line segment.
#             - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
#             - s2 (float): End index of the line segment.
#             - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.

#     Returns:
#         np.ndarray: Time gap between line segments. Shape (N, M).
#     """
#     space_intersection = intersect_space(lines1, lines2)
#     if len(space_intersection) == 0:
#         return float('inf')

#     slope1 = calculate_slope(lines1)
#     slope2 = calculate_slope(lines2)

#     s11, t11, _s12, _t12 = lines1
#     s21, t21, _s22, _t22 = lines2

#     t1_interp = (
#         t11 + pd.to_timedelta((space_intersection - s11) * slope1, unit='s'))
#     t2_interp = (
#         t21 + pd.to_timedelta((space_intersection - s21) * slope2, unit='s'))

#     time_diff = (t1_interp - t2_interp).total_seconds()
#     avg_abs_time_diff = np.mean(np.abs(time_diff))

#     return avg_abs_time_diff


def calculate_distance(
    lines1: np.ndarray,  # Shape (N, 4) or (4,)
    lines2: np.ndarray   # Shape (M, 4) or (4,)
) -> np.ndarray:
    """Calculate time gap among line segments.

    Args:
        lines1 (np.ndarray): Line segments 1. Shape (N, 4) or (4,), where each
            row is defined by:
            - s1 (float): Start index of the line segment.
            - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (float): End index of the line segment.
            - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.
        lines2 (np.ndarray): Line segments 2. Shape (M, 4) or (4,), where each
            row is defined by:
            - s1 (float): Start index of the line segment.
            - t1 (Union[float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (float): End index of the line segment.
            - t2 (Union[float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray: Time gap between line segments. Shape (N, M).
    """
    # Ensure lines1 and lines2 are at least 2D (shape (N, 4) and (M, 4))
    lines1 = np.atleast_2d(lines1)
    lines2 = np.atleast_2d(lines2)

    # Calculate the space intersection limits for each pair of lines
    space_intersection = intersect_space(lines1, lines2)

    if len(space_intersection) == 0:
        return np.full((lines1.shape[0], lines2.shape[0]), np.inf)

    # Calculate the slope for each line segment
    slope1 = calculate_slope(lines1)[:, np.newaxis]  # Shape (N, 1)
    slope2 = calculate_slope(lines2)[np.newaxis, :]  # Shape (1, M)

    # Extract the start points and times
    s11, t11 = lines1[:, 0][:, np.newaxis], lines1[:, 1][:, np.newaxis]
    s21, t21 = lines2[:, 0][np.newaxis, :], lines2[:, 1][np.newaxis, :]

    # Calculate interpolated times at the intersection points
    if isinstance(t11.flat[0], (pd.Timestamp, datetime)):
        # Convert the time difference to timedelta
        t1_interp = t11 + pd.to_timedelta((space_intersection[..., 0] - s11) * slope1, unit='s')
        t2_interp = t21 + pd.to_timedelta((space_intersection[..., 0] - s21) * slope2, unit='s')
    else:
        t1_interp = t11 + (space_intersection[..., 0] - s11) * slope1
        t2_interp = t21 + (space_intersection[..., 0] - s21) * slope2

    # Calculate the absolute time differences
    time_diff = np.abs(t1_interp - t2_interp)
    
    if isinstance(t11.flat[0], (pd.Timestamp, datetime)):
        time_diff = time_diff.total_seconds()  # Convert to seconds if needed

    avg_abs_time_diff = np.mean(time_diff, axis=-1)  # Shape (N, M)

    return avg_abs_time_diff
