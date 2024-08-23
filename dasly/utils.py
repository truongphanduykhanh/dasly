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
    t2: Union[pd.Timestamp, datetime]
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


def calculate_slope(
    s1: int,
    t1: Union[pd.Timestamp, datetime],
    s2: int,
    t2: Union[pd.Timestamp, datetime]
) -> float:
    """Calculate the slope of a line segment defined by (s1, t1) and (s2, t2).
    This is for spatio-temporal data where the x-axis is space and the y-axis
    is time. This slope is the inverse of the speed m/s.

    Args:
        s1 (int): Start index of the line segment.
        t1 (Union[pd.Timestamp, datetime]): Start timestamp of the line seg.
        s2 (int): End index of the line segment.
        t2 (Union[pd.Timestamp, datetime]): End timestamp of the line segment.

    Returns:
        float: Slope of the line segment.
    """
    # Calculate the time difference between t2 and t1
    time_diff = (t2 - t1).total_seconds()
    # Calculate the space difference between s2 and s1
    space_diff = s2 - s1
    # Calculate the slope of the line segment
    slope = time_diff / space_diff
    return slope


def calculate_speed(
    s1: int,
    t1: Union[pd.Timestamp, datetime],
    s2: int,
    t2: Union[pd.Timestamp, datetime]
) -> float:
    """Calculate the speed of a line segment defined by (s1, t1) and (s2, t2).
    This is for spatio-temporal data where the x-axis is space and the y-axis
    is time. The speed is in m/s.

    Args:
        s1 (int): Start index of the line segment.
        t1 (Union[pd.Timestamp, datetime]): Start timestamp of the line seg.
        s2 (int): End index of the line segment.
        t2 (Union[pd.Timestamp, datetime]): End timestamp of the line segment.

    Returns:
        float: Speed of the line segment.
    """
    # Calculate the time difference between t2 and t1
    time_diff = (t2 - t1).total_seconds()
    # Calculate the space difference between s2 and s1
    space_diff = s2 - s1
    # Calculate the slope of the line segment
    speed = space_diff / time_diff
    return speed


def calculate_distance(
    s11: int,
    t11: Union[pd.Timestamp, datetime],
    s12: int,
    t12: Union[pd.Timestamp, datetime],
    s21: int,
    t21: Union[pd.Timestamp, datetime],
    s22: int,
    t22: Union[pd.Timestamp, datetime]
) -> float:
    """Calculate the average time gap between two line segments defined by
    (s11, t11), (s12, t12) and (s21, t21), (s22, t22). This is for
    spatio-temporal data where the x-axis is space and the y-axis is time. The
    gap is in seconds.

    Args:
        s11 (int): Start index of the first line segment.
        t11 (Union[pd.Timestamp, datetime]): Start timestamp of first line seg.
        s12 (int): End index of the first line segment.
        t12 (Union[pd.Timestamp, datetime]): End timestamp of first line seg.
        s21 (int): Start index of the second line segment.
        t21 (Union[pd.Timestamp, datetime]): Start timestamp of second line sg.
        s22 (int): End index of the second line segment.
        t22 (Union[pd.Timestamp, datetime]): End timestamp of second line seg.

    Returns:
        float: Average time gap between the two line segments.
    """
    slope1 = calculate_slope(s11, t11, s12, t12)
    slope2 = calculate_slope(s21, t21, s22, t22)

    timestamps1 = interpolate_timestamps(s11, t11, s12, t12)
    timestamps2 = interpolate_timestamps(s21, t21, s22, t22)

    # Check slope of the two lines to determine the limits
    if slope1 >= 0 and slope2 >= 0:
        lim1 = max(s11, s21)
        lim2 = min(s12, s22)
        if lim1 > lim2:
            return 0
    elif slope1 < 0 and slope2 < 0:
        lim1 = min(s11, s21)
        lim2 = max(s12, s22)
        if lim1 > lim2:
            return 0
    elif slope1 * slope2 < 0:
        lim1 = max(s11, s22)
        lim2 = min(s12, s21)
        if lim1 > lim2:
            return 0

    # Calculate the average time gap between the two line segments
    timestamps1_lim = timestamps1[lim1 - s11:lim2 - s11 + 1]
    timestamps2_lim = timestamps2[lim1 - s21:lim2 - s21 + 1]
    time_diff = np.abs(timestamps1_lim - timestamps2_lim)
    avg_time_diff = np.mean(time_diff)
    return avg_time_diff.total_seconds()
