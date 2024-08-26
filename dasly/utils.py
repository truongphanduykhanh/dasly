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


def calculate_speed(lines: np.ndarray) -> Union[float, np.ndarray[float]]:
    """Calculate the speed of line segments defined by (s1, t1, s2, t2). This
    is for spatio-temporal data where the x-axis is space and the y-axis
    is time.

    Args:
        lines (np.ndarray[Union[float, pd.Timestamp, datetime]]): Array of line
            segments. Shape (N, 4) where N is the number of line segments, or
            (4,) for a single line. Each line segment is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        Union[float, np.ndarray[float]]: If the input is a single segment,
            returns a float. If the input is an array of segments, returns an
            array of slopes.
    """
    single_line = False
    # If input is a single line segment, reshape it to (1, 4)
    if lines.ndim == 1:
        lines = np.atleast_2d(lines)
        single_line = True

    # Convert timestamps to numeric values
    lines = convert_to_numeric(lines)  # Shape (N, 4)

    # Extracting s1, t1, s2, t2
    s1 = lines[:, 0]
    t1 = lines[:, 1]
    s2 = lines[:, 2]
    t2 = lines[:, 3]

    # Calculate the time difference between t2 and t1
    time_diff = (t2 - t1).astype(float)

    # Calculate the space difference between s2 and s1
    space_diff = (s2 - s1).astype(float)

    # Calculate the speed of the line segment
    speed = np.divide(
        space_diff,
        time_diff,
        where=time_diff != 0,
        out=np.full_like(space_diff, np.inf)
    )

    # If it was a single line segment, return a float
    if single_line:
        return speed[0]

    return speed


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
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

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
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray[Union[float, pd.Timestamp, datetime]]: Reordered array of
            line segments.
    """
    # Create a copy of the original array to avoid modifying it
    lines_copy = lines.copy()

    # If input is 1D array of shape (4,), reshape it to (1, 4)
    single_line = False
    if lines.ndim == 1:
        lines = np.atleast_2d(lines)
        single_line = True

    # Convert timestamps to numeric values
    lines = convert_to_numeric(lines)  # Shape (N, 4)

    # Extract the coordinates
    x1 = lines_copy[:, 0]
    y1 = lines_copy[:, 1]
    x2 = lines_copy[:, 2]
    y2 = lines_copy[:, 3]

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
    lines_copy = np.stack((x1, y1, x2, y2), axis=1)

    # If it was a single line segment, reshape the output back to (4,)
    if single_line:
        return lines_copy.reshape(4,)

    return lines_copy


def overlap_space_idx(
    lines1: np.ndarray,  # Shape (N, 4) or (4,)
    lines2: np.ndarray   # Shape (M, 4) or (4,)
) -> np.ndarray:
    """Calculate the space overlaping indices limits of line segments.

    Args:
        lines1 (np.ndarray): Line segments 1. Shape (N, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.
        lines2 (np.ndarray): Line segments 2. Shape (M, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray: Space overlaping indices limits. Shape (N, M, 2) or (2,) if
            input shapes were (4,).
    """
    # If inputs are 1D arrays of shape (4,), reshape them to (1, 4)
    single_lines = False
    if lines1.ndim == 1 and lines2.ndim == 1:
        lines1 = np.atleast_2d(lines1)
        lines2 = np.atleast_2d(lines2)
        single_lines = True

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

    # Stack the results along the last dimension to get shape (N, M, 2)
    result = np.stack([lim1, lim2], axis=-1)  # Shape (N, M, 2)

    # If original inputs were 1D, return a 1D result
    if single_lines:
        result = result.squeeze(axis=(0, 1))  # Shape (2,)

    return result.astype(float)


def calculate_line_gap(
    lines1: np.ndarray,  # Shape (N, 4) or (4,)
    lines2: np.ndarray   # Shape (M, 4) or (4,)
) -> Union[float, np.ndarray[float]]:
    """Calculate the average time gap between line segments.

    Args:
        lines1 (np.ndarray): Line segments 1. Shape (N, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.
        lines2 (np.ndarray): Line segments 2. Shape (M, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        Union[float, np.ndarray[float]]: If the input is a single segment,
            returns a float. If the input is an array of segments, returns an
            array of average time gaps.
    """
    # If inputs are 1D arrays of shape (4,), reshape them to (1, 4)
    single_lines = False
    if lines1.ndim == 1 and lines2.ndim == 1:
        lines1 = np.atleast_2d(lines1)  # Shape (1, 4)
        lines2 = np.atleast_2d(lines2)  # Shape (1, 4)
        single_lines = True

    # Convert timestamps to numeric values
    lines1 = convert_to_numeric(lines1)  # Shape (N, 4)
    lines2 = convert_to_numeric(lines2)  # Shape (M, 4)

    # Calculate slopes m for each line segment (N, )
    slope1 = calculate_slope(lines1)  # Shape (N, )
    slope2 = calculate_slope(lines2)  # Shape (M, )

    # Calculate the overlapping space indices limits
    overlap_lim = overlap_space_idx(lines1, lines2)  # Shape (N, M, 2)

    # Extract coordinates for lines1
    x1_1 = lines1[:, 0].reshape(-1, 1, 1)  # Shape (N, 1, 1)
    y1_1 = lines1[:, 1].reshape(-1, 1, 1)  # Shape (N, 1, 1)

    # Extract coordinates for lines2
    x1_2 = lines2[:, 0].reshape(1, -1, 1)  # Shape (1, M, 1)
    y1_2 = lines2[:, 1].reshape(1, -1, 1)  # Shape (1, M, 1)

    # Calculate the y values for each line segment at the overlapping space idx
    with np.errstate(invalid='ignore'):  # Ignore division by zero
        # Calculate y values for lines1 at each x position
        y_values_line1 = (  # Shape (N, M, L)
            slope1[:, np.newaxis, np.newaxis] * (overlap_lim - x1_1) + y1_1)
        y_values_line2 = (  # Shape (N, M, L)
            slope2[np.newaxis, :, np.newaxis] * (overlap_lim - x1_2) + y1_2)

    # Compute the absolute difference
    abs_diff = np.abs(y_values_line1 - y_values_line2)  # Shape (N, M, L)

    # Compute the average along the L dimension
    avg_abs_diff = np.mean(abs_diff, axis=-1)  # Shape (N, M)

    # Check the invalid_lim for overlap_lim and update the avg_abs_diff
    # This means 2 line segments do not overlap in space
    invalid_lim = overlap_lim[:, :, 0] > overlap_lim[:, :, -1]
    avg_abs_diff[invalid_lim] = np.inf

    # If original inputs were 1D, return a float
    if single_lines:
        return avg_abs_diff[0, 0]

    return avg_abs_diff


def convert_to_numeric(lines: np.ndarray) -> np.ndarray:
    """Convert the timestamps in the line segments to numeric values.

    Args:
        lines (np.ndarray): Array of line segments. Shape (N, 4) where N is the
            number of line segments. Each line segment is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray: Line segments with timestamps converted.
    """
    # Create a copy of the original array to avoid modifying it
    lines_copy = lines.copy()
    for i in range(lines_copy.shape[1]):
        if isinstance(lines_copy[:, i][0], (pd.Timestamp, datetime)):
            lines_copy[:, i] = (
                lines_copy[:, i].astype('datetime64[ns]').astype(int) / 10**9)
    return lines_copy.astype(float)


def convert_to_datetime(lines: np.ndarray) -> np.ndarray:
    """Convert the numeric values in the line segments to timestamps.

    Args:
        lines (np.ndarray): Array of line segments. Shape (N, 4) where N is the
            number of line segments. Each line segment is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray: Line segments with timestamps converted.
    """
    # Create a copy of the original array to avoid modifying it
    lines_copy = lines.copy()
    for i in [1, 3]:
        if isinstance(lines_copy[:, i][0], (int, float)):
            lines_copy[:, i] = pd.to_datetime(lines_copy[:, i], unit='s')
    return lines_copy
