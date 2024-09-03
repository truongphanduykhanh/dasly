"""Provides convenient utility functions for other modules"""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-08-23'


import os
import time
import re
import uuid
from datetime import datetime, timedelta
from typing import Union, Callable, Tuple
import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
from watchdog.events import FileSystemEventHandler

from dasly.simpledas import simpleDASreader

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_connection_string(
    endpoint: str,
    database: str,
    db_username: str,
    db_password: str,
    type: str = 'postgresql',
    dbapi: str = 'psycopg2',
    port: int = 5432
) -> str:
    """Create the connection string for the SQL database.

    Args:
        endpoint (str): Database endpoint.
        database (str): Database name.
        db_username (str): Database username.
        db_password (str): Database password.
        type (str, optional): Database type. Defaults to 'postgresql'.
        dbapi (str, optional): Database API. Defaults to 'psycopg2'.
        port (int, optional): Database port. Defaults to 5432.

    Returns:
        str: Connection string for the SQL database.
    """
    connection_string = (
        f'{type}+{dbapi}://{db_username}:{db_password}'
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
) -> None:
    """Write the data frame to SQL database.

    Args:
        df (pd.DataFrame): Data to write to the database.
        database_table (str): Table name in the database.
        connection_string (str): Connection string to the database.
    """
    # Create an engine
    engine = create_engine(connection_string, poolclass=NullPool)
    # Execute a SQL query
    df.to_sql(database_table, engine, if_exists='append', index=False)


def table_exists(
    table_name: str,
    connection_string: str
) -> bool:
    """Check if the table exists in the database.

    Args:
        table_name (str): Table name to check.
        connection_string (str): Connection string to the database.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    engine = create_engine(connection_string, poolclass=NullPool)
    inspector = inspect(engine)
    return inspector.has_table(table_name)


def gen_id(n: int) -> list[str]:
    """Generate a list of n unique IDs.

    Args:
        n (int): Number of unique IDs to generate.

    Returns:
        list[str]: List of n unique IDs.
    """
    uuid_list = [str(uuid.uuid4()) for _ in range(n)]
    return uuid_list


def match_line_id(
    current_lines_id: np.ndarray[Union[str, int]],
    previous_lines_id: np.ndarray[Union[str, int]],
    dist_mat: np.ndarray[float],
    threshold: float
) -> np.ndarray[Union[str, int]]:
    """Match the current line IDs to the previous line IDs based on the
    distance matrix. If the distance between the new line and the previous line
    is below the threshold, assign the previous line ID to the new line,
    otherwise maintain the current line ID.

    Args:
        current_lines_id (np.ndarray[Union[str, int]]): Current line IDs. Shape
            (N,).
        previous_lines_id (np.ndarray[Union[str, int]]): Previous line IDs.
            Shape (M,).
        dist_mat (np.ndarray[float]): Distance matrix between the new lines and
            the previous lines and. Shape (N, M).
        threshold (float): Threshold distance for assigning the line IDs.

    Returns:
        np.ndarray[Union[str, int]]: New line IDs.
    """
    # Replace NaN values with infinity
    # A vertical line will have nan distance with all other lines (including
    # horizontal lines, vertical lines, one point, etc.)
    # So we assume the distance between a vertical line and any other line is
    # infinity.
    dist_mat = np.nan_to_num(dist_mat, nan=np.inf)

    # Find the indices of the minimum distances
    min_indices = np.nanargmin(dist_mat, axis=1)

    # Find the minimum distances for each point
    min_distances = np.nanmin(dist_mat, axis=1)

    # Create a mask for distances below the threshold
    mask = min_distances <= threshold

    # Assign the previous line IDs to the new lines, where the distance is
    # below the threshold, otherwise assign a new unique ID
    new_lines_id = np.where(
        mask, previous_lines_id[min_indices], current_lines_id)

    return new_lines_id


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
        out=np.where(
            time_diff == 0,
            np.where(
                space_diff == 0,
                np.nan,
                np.where(space_diff > 0, np.inf, -np.inf)
            ),
            np.zeros_like(space_diff)
        )
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
            if np.signbit(speed):  # Check if speed is -0.0
                return -np.inf
            return np.inf
        return 1 / speed

    with np.errstate(divide='ignore'):
        reciprocal_speed = np.reciprocal(speed)
        # Handle the case for arrays with -0.0
        reciprocal_speed[np.isclose(speed, 0) & np.signbit(speed)] = -np.inf
        return reciprocal_speed


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


def assign_id_df(
    current_lines: pd.DataFrame,
    previous_lines: pd.DataFrame,
    id_col: str = 'id',
    line_id_col: str = 'line_id',
    s1_col: str = 's1',
    t1_col: str = 't1',
    s2_col: str = 's2',
    t2_col: str = 't2',
    threshold: float = 3,
) -> pd.DataFrame:
    """Assign unique ID and line ID to each line in the data frame lines.

    Args:
        current_lines (pd.DataFrame): Current lines. Must have columns s1_col,
            t1_col, s2_col and t2_col.
        previous_lines (pd.DataFrame): Previous lines. Must have columns
            s1_col, t1_col, s2_col, t2_col and line_id_col.
        id_col (str): Name of the ID column in previous_lines and to be added
            to current_lines. Default is 'id'.
        line_id_col (str): Name of the line ID column in previous_lines and to
            be added to current_lines. Default is 'line_id'.
        s1_col (str): Name of the s1 column in lines and previous_lines.
            Default is 's1'.
        t1_col (str): Name of the t1 column in lines and previous_lines.
            Default is 't1'.
        s2_col (str): Name of the s2 column in lines and previous_lines.
            Default is 's2'.
        t2_col (str): Name of the t2 column in lines and previous_lines.
            Default is 't2'.
        threshold (float): Maximum gap between the current line and the
            previous lines to be considered as the same line. Default is 3.

    Returns:
        pd.DataFrame: new columns id_col and line_id_col are added to the lines
            data frame.
    """
    # Extract the coordinates of the lines
    lines_coords = current_lines[[s1_col, t1_col, s2_col, t2_col]].to_numpy()
    previous_lines_coords = (
        previous_lines[[s1_col, t1_col, s2_col, t2_col]].to_numpy())

    # Calculate pairwise gap between line and the previous lines
    lines_gaps = calculate_line_gap(lines_coords, previous_lines_coords)

    # Assign unique ID and line ID to each line
    current_lines.insert(0, id_col, gen_id(len(current_lines)))
    lines_id = match_line_id(
        current_lines_id=current_lines[id_col].to_numpy(),
        previous_lines_id=previous_lines[line_id_col].to_numpy(),
        dist_mat=lines_gaps,
        threshold=threshold,
    )
    current_lines.insert(0, line_id_col, lines_id)

    return current_lines


def save_lines_csv(
    lines_df: pd.DataFrame,
    output_dir: str,
    file_name: str
) -> None:
    """Save the lines data frame to a CSV file.

    Args:
        lines_df (pd.DataFrame): Data frame containing the lines.
        output_dir (str): Path to the output directory.
        file_name (str): Name of the output CSV file.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the lines data frame to a CSV file
    lines_df.to_csv(os.path.join(output_dir, file_name), index=False)


# Define the event handler class
class HDF5EventHandler(FileSystemEventHandler):

    def __init__(
        self,
        event_thresh: int,
        dasly_fn: Callable[[str], None]
    ):
        """Initialize the event handler with an event threshold.

        Args:
            event_thresh (int): Number of events to wait before running dasly.
            dasly_fn (Callable[[str], None]): Function to run dasly. It takes
                the path to the hdf5 file as input.
        """
        super().__init__()
        self.event_thresh = event_thresh  # Set the event thresholh
        self.event_count = 0  # Initialize the event count
        self.last_created = None
        self.dasly_fn = dasly_fn

    def on_created(self, event):
        """Event handler for file creation (copying or moving). This is for
        testing only. Comment out when deploying.
        """
        if (
            # Check if the event is a hdf5 file
            event.src_path.endswith('.hdf5') and
            # Ensure the file is not a duplicate (sometimes watchdog triggers
            # the created event twice for the same file. This should be a bug
            # and we need to work around it by storing the last created file)
            event.src_path != self.last_created
        ):
            time.sleep(3)  # Wait for the file to be completely written
            logger.info(f'New hdf5: {event.src_path}')
            self.last_created = event.src_path  # Update the last created
            # In case we set the batch more than 10 seconds (i.e. wait for
            # more than one file to be created before running dasly), we need
            # to count the number of events and run dasly only when the event
            # count reaches the threshold
            self.event_count += 1
            if self.event_count >= self.event_thresh:
                logger.info('Runing dasly...')
                self.dasly_fn(event.src_path)
                self.event_count = 0  # Reset the event count

    def on_moved(self, event):
        """Event handler for file moving. In integrator, when a hdf5 file is
        completely written, it is moved from hdf5.tmp to hdf5.
        """
        if (
            # Check if the event is a hdf5 file
            event.dest_path.endswith('.hdf5') and
            # Ensure the file is not a duplicate (sometimes watchdog triggers
            # the created event twice for the same file. This should be a bug
            # and we need to work around it by storing the last created file)
            event.dest_path != self.last_created
        ):
            time.sleep(1)  # Wait for the file to be completely written
            logger.info(f'New hdf5: {event.dest_path}')
            self.last_created = event.dest_path  # Update the last created
            # In case we set the batch more than 10 seconds (i.e. wait for
            # more than one file to be created before running dasly), we need
            # to count the number of events and run dasly only when the event
            # count reaches the threshold
            self.event_count += 1
            if self.event_count >= self.event_thresh:
                logger.info('Runing dasly...')
                self.dasly_fn(event.dest_path)
                self.event_count = 0  # Reset the event count


def flatten_dict(
        d: dict,
        parent_key: str = '',
        sep: str = '_') -> dict:
    """Flatten a nested dictionary.

    Args:
        d (dict): Nested dictionary to flatten.
        parent_key (str, optional): Parent key. Defaults to ''.
        sep (str, optional): Separator. Defaults to '_'.

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_exp_dir(file_path: str) -> str:
    """Get the experiment directory from the HDF5 file path. The experiment
    directory is the parent directory of date directory. For example, file_path
    '/raid1/fsi/exps/Aastfjordbrua/20231005/dphi/082354.hdf5' would return
    '/raid1/fsi/exps/Aastfjordbrua'.

    Args:
        file_path (str): The path to the file. The file path must end with
            '/YYYYMMDD/dphi/HHMMSS.hdf5'.

    Returns:
        str: The experiment directory.
    """
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

    return exp_dir


def get_date_time(file_path: str) -> Tuple[str, str]:
    """Extract the date and time from the HDF5 file path.

    Args:
        file_path (str): The path to the HDF5 file. The file path must end with
            '/YYYYMMDD/dphi/HHMMSS.hdf5'.

    Returns:
        Tuple[str, str]: Date and time extracted from the file
    """
    # Define the regular expression pattern
    pattern = r"/(\d{8})/dphi/(\d{6})\.hdf5$"

    # Search for the pattern in the input string
    match = re.search(pattern, file_path)

    # Extract the matched groups
    date_str = match.group(1)
    time_str = match.group(2)

    return date_str, time_str


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


def get_file_paths_deploy(
    start_file: str = None,
    end_file: str = None,
    num_files: int = None,
    file_duration: int = 10
) -> list[str]:
    """Get the list of file paths for deployment.

    Args:
        start_file (str, optional): The start file path, inclusive. Defaults to
            None.
        end_file (str, optional): The end file path, inclusive. Defaults to
            None.
        num_files (int, optional): Number of files to get. Defaults to None.
        file_duration (int, optional): Duration of each file in seconds.
            Defaults to 10.

    Returns:
        list[str]: List of file paths. This could be empty if no files are
            found.
    """
    # Check if two and only two out of three are inputted
    if (start_file is None) + (num_files is None) + (end_file is None) != 1:
        raise ValueError('The function accepts two and only two out of '
                         + 'three (start_file, end_file, num_files)')

    if end_file is None:
        exp_dir = get_exp_dir(start_file)
        start_date, start_time = get_date_time(start_file)
        start_date_time = f'{start_date} {start_time}'
        end_date_time = add_subtract_dt(
            start_date_time,
            (num_files - 1) * file_duration
        )
        end_date, end_time = end_date_time.split(' ')
        end_file = os.path.join(exp_dir, end_date, 'dphi', f'{end_time}.hdf5')

    elif start_file is None:
        exp_dir = get_exp_dir(end_file)
        end_date, end_time = get_date_time(end_file)
        end_date_time = f'{end_date} {end_time}'
        start_date_time = add_subtract_dt(
            end_date_time,
            - (num_files - 1) * file_duration
        )
        start_date, start_time = start_date_time.split(' ')
        start_file = os.path.join(
            exp_dir, start_date, 'dphi', f'{start_time}.hdf5'
        )

    else:  # number_files is None
        exp_dir = get_exp_dir(start_file)
        start_date, start_time = get_date_time(start_file)
        start_date_time = f'{start_date} {start_time}'
        end_date, end_time = get_date_time(end_file)
        end_date_time = f'{end_date} {end_time}'
        num_files = int(
            (datetime.strptime(end_date_time, '%Y%m%d %H%M%S')
             - datetime.strptime(start_date_time, '%Y%m%d %H%M%S'))
            .total_seconds() / file_duration
        )

    file_paths_gen = gen_file_paths(start_file, end_file, file_duration)

    file_paths_exist, _, _ = simpleDASreader.find_DAS_files(
        experiment_path=exp_dir,
        start=start_date_time,
        duration=(num_files + 1) * file_duration,
        show_header_info=False
    )

    file_paths = list(set(file_paths_gen) & set(file_paths_exist))
    file_paths.sort()

    return file_paths


def gen_file_paths(
    start_file: str, end_file: str, file_duration: str = 10
) -> list[str]:
    """Generate the list of file paths between the start and end files.

    Args:
        start_file (str): Start file path.
        end_file (str): End file path.
        file_duration (int, optional): Duration of each file in seconds.
            Defaults to 10.

    Returns:
        list[str]: List of file paths.
    """
    # Extract the common prefix from the file paths
    exp_dir = get_exp_dir(start_file)

    # Extract the date and time from the start and end paths
    start_date, start_time = get_date_time(start_file)
    start_date_time = f'{start_date} {start_time}'

    end_date, end_time = get_date_time(end_file)
    end_date_time = f'{end_date} {end_time}'

    # Convert the extracted strings into datetime objects
    start_datetime = datetime.strptime(start_date_time, '%Y%m%d %H%M%S')
    end_datetime = datetime.strptime(end_date_time, '%Y%m%d %H%M%S')

    # Generate the list of datetime objects with 10-second intervals
    current_datetime = start_datetime
    datetime_list = []

    while current_datetime <= end_datetime:
        datetime_list.append(current_datetime)
        current_datetime += timedelta(seconds=file_duration)

    # Convert the datetime objects back into file paths
    file_paths = [
        os.path.join(
            exp_dir,
            dt.strftime('%Y%m%d'),
            'dphi',
            f'{dt.strftime("%H%M%S")}.hdf5')
        for dt in datetime_list
    ]
    return file_paths


def extract_elements(
    lst: list[Union[str, int, float]],
    num: int,
    last_value: Union[str, int, float]
) -> list[Union[str, int, float]]:
    """Extract num elements from the list lst that is up to the last_value
    (inclusive).

    Args:
        lst (list[Union[str, int, float]]): List of elements.
        num (int): Number of elements to extract.
        last_value (Union[str, int, float]): Value up to in the extracted
            elements.

    Returns:
        list[Union[str, int, float]]: Extracted elements
    """
    idx = lst.index(last_value)
    start_idx = max(0, idx - num + 1)
    ext_elems = lst[start_idx:idx + 1]
    return ext_elems


def drop_table(table_name: str, connection_string: str) -> None:
    """Drop a table from the database.

    Args:
        table_name (str): Name of the table to drop.
        connection_string (str): Connection string to the database.
    """
    engine = create_engine(connection_string, poolclass=NullPool)
    Base = declarative_base()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        Base.metadata.drop_all(engine, [table], checkfirst=True)
