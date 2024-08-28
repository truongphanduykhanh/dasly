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
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import NullPool
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from dasly.master import Dasly

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Parameters from the environment variables and YAML file
###############################################################################
###############################################################################
###############################################################################

# Access environment variables to get the database credentials
# need to set the environment variables in the terminal in advance:
# determine if bash shell and zsh shell: echo $SHELL
# determine if bash shell with login:
# shopt -q login_shell && echo 'Login shell' || echo 'Non-login shell'
# if bash shell with non-login: nano ~/.bashrc
# if bash shell with login: nano ~/.bash_profile
# if zsh shell: nano ~/.zshrc
# add the following lines to the file:
# export POSTGRESQL_USERNAME='your_username'
# export POSTGRESQL_PASSWORD='your_password'
# save (Ctrl+O) and exit (Ctrl+X)

db_username = os.getenv('POSTGRESQL_USERNAME')
db_password = os.getenv('POSTGRESQL_PASSWORD')

# Define the path to the YAML file
yaml_path = 'config_whales.yml'

# Open and read the YAML file
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)


# Access parameters from the YAML file
input_dir = params['input_dir']
start_exact_second = params['start_exact_second']
integrate = params['integrate']

database_type = params['database']['type']
dbapi = params['database']['dbapi']
endpoint = params['database']['endpoint']
port = params['database']['port']
database = params['database']['database']
database_table = params['database']['table']

hdf5_file_length = params['hdf5_file_length']
batch = params['dasly']['batch']
batch_gap = params['dasly']['batch_gap']

bandpass_filter_low = params['bandpass_filter']['low']
bandpass_filter_high = params['bandpass_filter']['high']

sample_meters = params['sample']['meters']
sample_seconds = params['sample']['seconds']

gaussian_smooth_s1 = params['gaussian_smooth']['s1']
gaussian_smooth_s2 = params['gaussian_smooth']['s2']
gaussian_smooth_std_s = params['gaussian_smooth']['std_s']
gauusian_smooth_unit = params['gaussian_smooth']['unit']

binary_threshold = params['binary_threshold']

hough_speed_res = params['hough_transform']['speed_res']
hough_length_meters = params['hough_transform']['length_meters']
hough_threshold_percent = params['hough_transform']['threshold_percent']
hough_max_line_gap_percent = params['hough_transform']['max_line_gap_percent']
hough_speed_unit = params['hough_transform']['speed_unit']

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


def create_connection_string() -> str:
    """Create the connection string for the SQL database."""
    connection_string = (
        f'{database_type}+{dbapi}://{db_username}:{db_password}'
        + f'@{endpoint}:{port}/{database}'
    )
    return connection_string


def read_sql(query: str) -> pd.DataFrame:
    """Load the data from SQL database."""
    # Create the connection string
    connection_string = create_connection_string()
    # Create an engine
    engine = create_engine(connection_string, poolclass=NullPool)
    # Execute a SQL query
    df = pd.read_sql(query, engine)
    return df


def write_sql(
    df: pd.DataFrame,
    batch_id: str = None,
    created_at: bool = True
) -> None:
    """Write the data to SQL database."""
    # Create the connection string
    connection_string = create_connection_string()
    # Create an engine
    engine = create_engine(connection_string, poolclass=NullPool)
    # Execute a SQL query
    if batch_id:
        df.insert(0, 'batch_id', batch_id)
    if created_at:
        df.insert(0, 'created_at', pd.Timestamp.now(tz='UTC'))
    df.to_sql(database_table, engine, if_exists='append', index=False)


def check_table_exists(table_name: str) -> bool:
    """Check if the table exists in the database.
    """
    connection_string = create_connection_string()
    engine = create_engine(connection_string, poolclass=NullPool)
    inspector = inspect(engine)
    return inspector.has_table(table_name)


def assign_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assign unique ID to each row in the data frame."""
    uuid_list = [str(uuid.uuid4()) for _ in range(len(df))]
    df.insert(0, 'line_id', uuid_list)
    df.insert(0, 'id', uuid_list)
    return df


def reassign_id(
    previous_lines_id: np.ndarray,
    current_lines_id: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Reassign the line ID based on the minimum distance between the previous
    and new lines. If the distance is below the threshold, the new line is
    assigned the ID of the previous nearest line. If the distance is above the
    threshold, the new line ID is kept.

    Args:
        previous_lines_id (np.ndarray): Previous line IDs. Shape (N,).
        current_lines_id (np.ndarray): Current line IDs. Shape (M,).
        dist_mat (np.ndarray): Distance matrix between the previous lines and
            new lines. Shape (N, M)
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
    lines_id_resigned = np.where(
        mask, previous_lines_id[min_indices], current_lines_id)
    return lines_id_resigned


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
    first_file_dt = add_subtract_dt(file_dt, hdf5_file_length - batch)

    # Load the data
    ###########################################################################
    s_rate = 0.25
    das = Dasly()
    das.load_data(
        folder_path=input_dir,
        start=first_file_dt,
        duration=batch,
        start_exact_second=start_exact_second,
        integrate=integrate,
        chIndex=np.arange(round(5000 * s_rate), round(85000 * s_rate)),
        reset_channel_idx=False
    )

    das.bandpass_filter(low=bandpass_filter_low, high=bandpass_filter_high)
    das.signal = np.abs(das.signal)
    das.sample(meters=sample_meters, seconds=sample_seconds)
    das.gaussian_smooth(
        s1=gaussian_smooth_s1,
        s2=gaussian_smooth_s2,
        std_s=gaussian_smooth_std_s,
        unit=gauusian_smooth_unit
    )
    das.sobel_filter()
    das.binary_transform(threshold=binary_threshold)
    das.hough_transform(
        target_speed=(gaussian_smooth_s1 + gaussian_smooth_s2) / 2,
        speed_res=hough_speed_res,
        length_meters=hough_length_meters,
        threshold_percent=hough_threshold_percent,
        max_line_gap_percent=hough_max_line_gap_percent,
        speed_unit=hough_speed_unit
    )

    # dbscan forward and backward lines
    ###########################################################################
    if len(das.lines) == 0:  # if there are no lines, exit the function
        return
    das.dbscan(eps_seconds=dbscan_eps_seconds)  # group similar lines together
    lines = das.lines
    lines_df = assign_id(das.lines_df)  # assign unique id to each line

    # load previous data
    ###########################################################################
    # if the table does not exist yet (first time running dasly)
    if not check_table_exists(table_name=database_table):
        write_sql(
            lines_df,
            batch_id=pd.Timestamp(file_dt),
            created_at=True)
        return  # exit the function from here
    previous_bacth_id = add_subtract_dt(file_dt, -batch_gap)
    previous_bacth_id = (
        pd.Timestamp(previous_bacth_id)
        .strftime('%Y-%m-%d %H:%M:%S')
    )
    query = (
        f'SELECT * FROM {database_table}' +
        f" WHERE batch_id = '{previous_bacth_id}';"
    )
    previous_lines_df = read_sql(query)
    if len(previous_lines_df) == 0:  # if there are no previous lines
        write_sql(
            lines_df,
            batch_id=pd.Timestamp(file_dt),
            created_at=True)
        return  # exit the function from here
    previous_lines = previous_lines_df[['x1', 'y1', 'x2', 'y2']].to_numpy()
    # shift current lines up (add y values by gap)
    lines = lines.astype(np.float64)
    lines[:, [1, 3]] += batch_gap * das.t_rate

    # Group the current lines with the previous lines
    ###########################################################################
    lines_y_vals = das._y_vals_lines(lines, np.arange(das.signal.shape[1]))
    previous_lines_y_vals = das._y_vals_lines(
        previous_lines, np.arange(das.signal.shape[1]))

    dist_mat = das._metric(previous_lines_y_vals, lines_y_vals)

    lines_id = reassign_id(
        previous_lines_id=previous_lines_df['line_id'].to_numpy(),
        current_lines_id=lines_df['line_id'].to_numpy(),
        dist_mat=dist_mat,
        threshold=dbscan_eps_seconds*das.t_rate
    )
    lines_df['line_id'] = lines_id

    # Append DataFrame to table in PostgreSQL
    ###########################################################################
    # assign batch id (batch id is taken from the file path)
    write_sql(
        lines_df,
        batch_id=pd.Timestamp(file_dt),
        created_at=True)


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
            time.sleep(2)  # Wait for the file to be completely written
            logger.info(f'New hdf5: {event.src_path}')
            self.last_created = event.src_path  # Update the last created
            # In case we set the batch more than 10 seconds (i.e. wait for
            # more than one file to be created before running dasly), we need
            # to count the number of events and run dasly only when the event
            # count reaches the threshold
            self.event_count += 1
            if self.event_count >= self.event_thresh:
                logger.info('Runing dasly...')
                run_dasly(event.src_path)
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
            time.sleep(2)  # Wait for the file to be completely written
            logger.info(f'New hdf5: {event.dest_path}')
            self.last_created = event.dest_path  # Update the last created
            # In case we set the batch more than 10 seconds (i.e. wait for
            # more than one file to be created before running dasly), we need
            # to count the number of events and run dasly only when the event
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
event_thresh = batch_gap / hdf5_file_length
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
