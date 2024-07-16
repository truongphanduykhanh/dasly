"""Provides convenient methods to deploy in a DAS system."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-07-10'

import os
import time
from datetime import datetime, timedelta
import uuid
import re
import logging

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from dasly.dasly import Dasly

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Parameters from the environment variables and YAML file
###############################################################################
###############################################################################
###############################################################################

# Access environment variables to get the database credentials
# need to set the environment variables in the terminal in advance:
# export POSTGRESQL_USERNAME='your_username'
# export POSTGRESQL_PASSWORD='your_password'

db_username = os.getenv('POSTGRESQL_USERNAME')
db_password = os.getenv('POSTGRESQL_PASSWORD')

# Define the path to the YAML file
yaml_path = 'config.yml'

# Open and read the YAML file
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)


# Access parameters from the YAML file
input_dir = params['input_dir']

database_type = params['database']['database_type']
dbapi = params['database']['dbapi']
endpoint = params['database']['endpoint']
port = params['database']['port']
database = params['database']['database']
table = params['database']['table']

batch_hdf5 = params['batch_hdf5']
batch = params['dasly']['batch']
batch_gap = params['dasly']['batch_gap']

lowpass_filter_freq = params['lowpass_filter_freq']
decimate_t_rate = params['decimate_t_rate']

gaussian_smooth_s1 = params['gaussian_smooth']['s1']
gaussian_smooth_s2 = params['gaussian_smooth']['s2']
gaussian_smooth_std_s = params['gaussian_smooth']['std_s']

binary_threshold = params['binary_threshold']

hough_speed_res = params['hough_transform']['speed_res']
hough_length_meters = params['hough_transform']['length_meters']

dbscan_eps_seconds = params['dbscan_eps_seconds']


# Helper functions
###############################################################################
###############################################################################
###############################################################################

def extract_dt(path: str) -> str:
    """Extract the date and time from the input path with format ending with
    /YYYYMMDD/dphi/HHMMSS.hdf5. Output format: 'YYYYMMDD HHMMSS'
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
    """Add or subtract x seconds to time t having format 'YYYYMMDD HHMMSS'
    """
    # Convert input string to datetime object
    input_dt = datetime.strptime(dt, format)
    # Add x seconds to the datetime object
    new_dt = input_dt + timedelta(seconds=x)
    # Format the new datetime object back into the same string format
    new_dt_str = new_dt.strftime(format)
    return new_dt_str


def assign_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assign unique ID to each row in the data frame."""
    uuid_list = [str(uuid.uuid4()) for _ in range(len(df))]
    df.insert(0, 'id', uuid_list)
    return df


def read_sql(query: str) -> pd.DataFrame:
    """Load the data from SQL database."""
    # Create the connection string
    connection_string = (
        f'{database_type}+{dbapi}://{db_username}:{db_password}'
        + f'@{endpoint}:{port}/{database}'
    )
    # Create an engine
    engine = create_engine(connection_string)
    # Execute a SQL query
    df = pd.read_sql(query, engine)
    return df


def write_sql(df: pd.DataFrame, add_created_at: bool = True) -> None:
    """Write the data to SQL database."""
    # Create the connection string
    connection_string = (
        f'{database_type}+{dbapi}://{db_username}:{db_password}'
        + f'@{endpoint}:{port}/{database}'
    )
    # Create an engine
    engine = create_engine(connection_string)
    # Execute a SQL query
    if add_created_at:
        df['created_at'] = pd.Timestamp.now()
    df.to_sql(table, engine, if_exists='append', index=False)


def reassign_lines_id(
    previous_lines_id: np.ndarray,
    new_lines_id: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Reassign the line ID based on the minimum distance between the previous
    and new lines. If the distance is below the threshold, the new line is
    assigned the ID of the previous nearest line. If the distance is above the
    threshold, the new line ID is kept.

    Args:
        previous_lines_id (np.ndarray): Previous line IDs.
        new_lines_id (np.ndarray): New line IDs.
        dist_mat (np.ndarray): Distance matrix between the previous lines and
            new lines.
        threshold (float): Threshold to consider the distance as a match.

    Returns:
        np.ndarray: New line IDs.
    """
    # Find the indices of the minimum distances
    min_indices = np.argmin(dist_mat, axis=0)
    # Find the minimum distances for each point in b
    min_distances = np.min(dist_mat, axis=0)
    # Create a mask for distances below the threshold
    mask = min_distances < threshold
    # Create an array to hold the new names
    new_lines_id = np.where(mask, previous_lines_id[min_indices], new_lines_id)
    return new_lines_id


# Run dasly
###############################################################################
###############################################################################
###############################################################################
def run_dasly(file_path: str) -> None:
    """Run the dasly algorithm on the input directory and save the output to
    PostgreSQL.
    """
    # Get the first file date and time to load in dasly
    ###########################################################################
    file_dt = extract_dt(file_path)
    first_file_dt = add_subtract_dt(file_dt, batch_hdf5 - batch)

    # Load the data
    ###########################################################################
    lines = np.empty((0, 4))  # create an empty array to store the lines
    das = Dasly()
    das.load_data(
        folder_path=input_dir,
        start=first_file_dt,
        duration=batch,
        integrate=False  # do not integrate the data (strain rate unit)
    )

    # forward Gaussian smoothing
    ###########################################################################
    das.lowpass_filter(cutoff=lowpass_filter_freq)
    das.decimate(t_rate=decimate_t_rate)
    das.gaussian_smooth(
        s1=gaussian_smooth_s1,
        s2=gaussian_smooth_s2,
        std_s=gaussian_smooth_std_s)
    das.sobel_filter()
    das.binary_transform(threshold=binary_threshold)
    das.hough_transform(
        target_speed=(gaussian_smooth_s1 + gaussian_smooth_s2) / 2,
        speed_res=hough_speed_res,
        length_meters=hough_length_meters)
    if das.lines is not None:
        # store forward lines
        mask = das.lines_df['speed_kmh'].to_numpy() >= 0
        lines = np.concatenate((lines, das.lines[mask]), axis=0)

    # backward Gaussian smoothing
    ###########################################################################
    das.reset()
    das.lowpass_filter(cutoff=lowpass_filter_freq)
    das.decimate(t_rate=decimate_t_rate)
    das.gaussian_smooth(
        s1=-gaussian_smooth_s2,
        s2=-gaussian_smooth_s1,
        std_s=gaussian_smooth_std_s)
    das.sobel_filter()
    das.binary_transform(threshold=binary_threshold)
    das.hough_transform(
        target_speed=(gaussian_smooth_s1 + gaussian_smooth_s2) / 2,
        speed_res=hough_speed_res,
        length_meters=hough_length_meters)
    if das.lines is not None:
        # store forward lines
        mask = das.lines_df['speed_kmh'].to_numpy() < 0
        lines = np.concatenate((lines, das.lines[mask]), axis=0)

    # dbscan forward and backward lines
    ###########################################################################
    das.lines = lines  # assign the new lines to the das object
    das.dbscan(eps_seconds=dbscan_eps_seconds)  # group similar lines together
    lines = das.lines
    lines_df = assign_id(das.lines_df)  # assign unique id to each line

    # load previous data
    ###########################################################################
    previous_bacth_id = add_subtract_dt(file_dt, -batch_gap)
    previous_bacth_id = (
        pd.Timestamp(previous_bacth_id)
        .strftime('%Y-%m-%d %H:%M:%S')
    )
    query = f'SELECT * FROM vehicles WHERE batch_id = {previous_bacth_id};'
    previous_lines_df = read_sql(query)
    previous_lines = previous_lines_df[['x1, y1, x2, y2']].to_numpy()
    # shift previous lines up (add y values by gap)
    previous_lines[:, [1, 3]] += batch_gap * das.t_rate

    # Group the new lines with the previous lines
    ###########################################################################
    lines_y_vals = das._y_vals_lines(lines, np.arange(das.signal.shape[1]))
    previous_lines_y_vals = das._y_vals_lines(
        previous_lines, np.arange(das.signal.shape[1]))

    lines_y_vals_reshape = lines_y_vals[:, np.newaxis, :]
    previous_lines_y_vals_reshape = previous_lines_y_vals[np.newaxis, :, :]
    dist_mat = das._metric(lines_y_vals_reshape, previous_lines_y_vals_reshape)
    lines_id = reassign_lines_id(
        previous_lines_id=previous_lines_df['id'].to_numpy(),
        new_lines_id=lines['id'].to_numpy(),
        dist_mat=dist_mat,
        threshold=1
    )
    lines_df['id'] = lines_id

    # Append DataFrame to table in PostgreSQL
    ###########################################################################
    # assign batch id (batch id is taken from the file path)
    lines_df['batch_id'] = pd.Timestamp(file_dt)
    write_sql(lines_df, add_created_at=True)


# Define the event handler class
# To run function dasly() whenever a new hdf5 file is created in the input dir
###############################################################################
###############################################################################
###############################################################################
class MyHandler(FileSystemEventHandler):

    def __init__(self, event_thresh):
        """Initialize the event handler with an event threshold
        """
        super().__init__()
        self.event_thresh = event_thresh
        self.event_count = 0  # Initialize the event count
        self.last_created = None

    def on_any_event(self, event):
        """Event handler for any file system event
        """
        if event.is_directory:  # Skip directories
            return
        if (
            # Check if the event is a file move
            event.event_type == 'moved' and
            # Ensure the destination path is not None
            event.dest_path is not None and
            # Ensure the file is not a duplicate
            event.dest_path != self.last_created
        ):
            time.sleep(1)  # Wait for the file to be completely written
            logger.info(f'New hdf5: {event.dest_path}')
            self.last_created = event.dest_path  # Update the last created file
            # In case we set the batch more than 10 seconds (i.e. wait for more
            # than one file to be created before running dasly), we need to
            # count the number of events and run dasly only when the event
            # count reaches the threshold
            self.event_count += 1
            if self.event_count >= self.event_thresh:
                logger.info('Runing dasly...')
                run_dasly(event.dest_path)
                self.event_count = 0  # Reset the event count


# Initialize the Observer and EventHandler
###############################################################################
###############################################################################
###############################################################################
# how many files to wait before running dasly
event_thresh = batch_gap / batch_hdf5
# Initialize the Observer and EventHandler
event_handler = MyHandler(event_thresh=event_thresh)
observer = Observer()
observer.schedule(event_handler=event_handler, path=input_dir, recursive=True)

logger.info(f'Watching directory: {input_dir}')

# Start the observer
observer.start()

# Keep the script running until Ctrl+C is pressed. Without this try...except...
# the program will still terminate if using keyboard Ctrl+C, but the Observer
# will not have a chance to stop properly, which could leave resources (such as
# file handles or threads) improperly released.
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()  # Stop the observer by pressing Ctrl+C


observer.join()
