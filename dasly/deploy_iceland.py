"""Provides the main functions to run the Dasly pipeline for the Iceland
dataset."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-08-27'


import os
import time
import logging
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from watchdog.observers import Observer

from dasly.master import Dasly
from dasly.utils import (
    assign_id_df,
    save_lines_csv,
    HDF5EventHandler,
    add_subtract_dt,
    get_date_time,
    gen_id,
    create_connection_string,
    read_sql,
    write_sql,
    table_exists
)


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load the parameters
yaml_path = 'config_iceland.yml'
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)


def dasly_core(start: str) -> pd.DataFrame:
    """Run the Dasly core algorithms.

    Args:
        start (str): The start time of the data to load.

    Returns:
        pd.DataFrame: The DataFrame containing the detected lines.
    """
    # Load the data
    ###########################################################################
    s_rate = 0.25
    das = Dasly()
    das.load_data(
        folder_path=params['input_dir'],
        start=start,
        duration=params['dasly']['batch'],
        start_exact_second=params['start_exact_second'],
        integrate=params['integrate'],
        chIndex=np.arange(round(5000 * s_rate), round(85000 * s_rate)),
        reset_channel_idx=False
    )

    # forward Gaussian smoothing
    ###########################################################################
    das.bandpass_filter(
        low=params['bandpass_filter']['low'],
        high=params['bandpass_filter']['high']
    )
    das.signal = np.abs(das.signal)
    das.sample(
        meters=params['sample']['meters'],
        seconds=params['sample']['seconds']
    )
    das.gaussian_smooth(
        s1=params['gaussian_smooth']['s1'],
        s2=params['gaussian_smooth']['s2'],
        std_s=params['gaussian_smooth']['std_s'],
        unit=params['gaussian_smooth']['unit']
    )
    das.sobel_filter()
    das.binary_transform(threshold=params['binary_threshold'])
    das.hough_transform(
        target_speed=((params['gaussian_smooth']['s1']
                       + params['gaussian_smooth']['s2']) / 2),
        speed_res=params['hough_transform']['speed_res'],
        length_meters=params['hough_transform']['length_meters'],
        threshold_percent=params['hough_transform']['threshold_percent'],
        max_line_gap_percent=params['hough_transform']['max_line_gap_percent'],
        speed_unit=params['hough_transform']['speed_unit']
    )
    # dbscan
    ###########################################################################
    if len(das.lines) == 0:  # if there are no lines, exit the function
        return
    das.dbscan(eps_seconds=params['dbscan_eps_seconds'])  # group close lines
    return das.lines_df


def dasly_pipeline_csv(file_path: str) -> None:
    """Run the Dasly pipeline and store output as csv files. This function is
    called when a new HDF5 file is detected. Refer to the `HDF5EventHandler`
    class for more information.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.
    """
    # Get the experiment directory, date, and time
    date_str, time_str = get_date_time(file_path)

    # Run the dasly_core
    first_file_dt = add_subtract_dt(
        f'{date_str} {time_str}',
        params['hdf5_file_length'] - params['dasly']['batch']
    )
    lines_df = dasly_core(first_file_dt)
    if lines_df is None:
        return

    # Load the previous lines
    previous_csv_dt = add_subtract_dt(
        f'{date_str} {time_str}', - params['dasly']['batch_gap'])
    previous_csv_date = previous_csv_dt.split(' ')[0]
    previous_csv_time = previous_csv_dt.split(' ')[1]
    previous_csv_path = os.path.join(
        params['output_dir'], previous_csv_date, f'{previous_csv_time}.csv')

    # Assign IDs to the lines
    if os.path.exists(previous_csv_path):  # Check if the previous file exists
        previous_lines_df = pd.read_csv(
            previous_csv_path,
            parse_dates=[
                't1', 't2',
                't1_edge', 't2_edge',
                't1_edge_ext', 't2_edge_ext'
            ]
        )
        # Match the IDs between the previous and current lines
        lines_df = assign_id_df(lines_df, previous_lines_df)
    else:
        # Assign new IDs to the lines
        lines_df.insert(0, 'id', gen_id(len(lines_df)))
        lines_df.insert(0, 'line_id', lines_df['id'])

    # Save the lines
    lines_df.insert(0, 'batch_id', f'{date_str} {time_str}')
    lines_df.insert(0, 'created_at', pd.Timestamp.now(tz='UTC'))
    save_lines_csv(
        lines_df=lines_df,
        output_dir=os.path.join(params['output_dir'], date_str),
        file_name=f'{time_str}.csv'
    )


def dasly_pipeline_db(file_path: str) -> None:
    """Run the Dasly pipeline and store output in a database. This function is
    called when a new HDF5 file is detected. Refer to the `HDF5EventHandler`
    class for more information.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.
    """
    # Get the experiment directory, date, and time
    date_str, time_str = get_date_time(file_path)

    # Run the dasly_core
    first_file_dt = add_subtract_dt(
        f'{date_str} {time_str}',
        params['hdf5_file_length'] - params['dasly']['batch']
    )
    lines_df = dasly_core(first_file_dt)
    if lines_df is None:
        return

    # if the table does not exist yet (first time running dasly)
    connection_string = create_connection_string(
        endpoint=params['database']['endpoint'],
        database=params['database']['database'],
        db_username=os.getenv('POSTGRESQL_USERNAME'),
        db_password=os.getenv('POSTGRESQL_PASSWORD'),
        type=params['database']['type'],
        dbapi=params['database']['dbapi'],
        port=params['database']['port']
    )
    if table_exists(
        table_name=params['database']['table'],
        connection_string=connection_string
    ):
        previous_dt = add_subtract_dt(
            f'{date_str} {time_str}', - params['dasly']['batch_gap'])
        query = (
            f'SELECT * FROM {params["database"]["table"]}' +
            f" WHERE batch_id = '{previous_dt}';"
        )
        previous_lines_df = read_sql(query, connection_string)
    else:
        previous_lines_df = pd.DataFrame()

    # Assign IDs to the lines
    if len(previous_lines_df) > 0:
        # Match the IDs between the previous and current lines
        lines_df = assign_id_df(lines_df, previous_lines_df)
    else:
        # Assign new IDs to the lines
        lines_df.insert(0, 'id', gen_id(len(lines_df)))
        lines_df.insert(0, 'line_id', lines_df['id'])

    # Save the lines
    lines_df.insert(0, 'batch_id', f'{date_str} {time_str}')
    lines_df.insert(0, 'created_at', pd.Timestamp.now(tz='UTC'))
    write_sql(
        df=lines_df,
        database_table=params['database']['table'],
        connection_string=connection_string
    )


def dasly_pipeline(file_path: str) -> Callable:
    """Run the Dasly pipeline and store output in either csv files or a
    database. This function is called when a new HDF5 file is detected. Refer
    to the `HDF5EventHandler` class for more information.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.

    Returns:
        Callable: The function to run the Dasly pipeline.
    """
    if params['output_type'] == 'db':
        dasly_pipeline_db(file_path)
    else:
        dasly_pipeline_csv(file_path)


###############################################################################
###############################################################################
###############################################################################
if __name__ == "__main__":

    # Initialize the Observer and EventHandler
    event_handler = HDF5EventHandler(
        event_thresh=params['dasly']['batch_gap'] / params['hdf5_file_length'],
        dasly_fn=dasly_pipeline,
    )
    observer = Observer()
    observer.schedule(
        event_handler=event_handler,
        path=params['input_dir'],
        recursive=True
    )
    logger.info(f'Watching directory: {params["input_dir"]}')

    # Start the observer
    observer.start()

    ###########################################################################
    # Keep the script running until Ctrl+C is pressed. Without "try . except ."
    # the program will still terminate if using keyboard Ctrl+C, but the
    # Observer will not have a chance to stop properly, which could leave
    # resources (such as file handles or threads) improperly released.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()  # Stop the observer by pressing Ctrl+C

    observer.join()
    ###########################################################################
