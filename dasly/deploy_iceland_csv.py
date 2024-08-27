"""Provides the main functions to run the Dasly pipeline for the Iceland
dataset."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-08-27'


import os
import time
import logging

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
    get_exp_dir,
    get_date_time,
    gen_id
)


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load the parameters
yaml_path = 'config_iceland.yml'
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)


# Run dasly
###############################################################################
###############################################################################
###############################################################################
def dasly_core(start: str) -> pd.DataFrame:
    """Run the Dasly core algorithms.

    Args:
        start (str): The start time of the data to load.

    Returns:
        pd.DataFrame: The DataFrame containing the detected lines.
    """
    # Load the data
    ###########################################################################
    das = Dasly()
    das.load_data(
        folder_path=params['input_dir'],
        start=start,
        duration=params['hdf5_file_length'],
        start_exact_second=params['start_exact_second'],
        integrate=params['integrate']
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
        meters=params['sample']['seconds']
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


def dasly_pipeline(file_path: str) -> None:
    """Run the Dasly pipeline. This function is called when a new HDF5 file is
    detected. Refer to the `HDF5EventHandler` class for more information.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.
    """
    # Get the experiment directory, date, and time
    exp_dir = get_exp_dir(file_path)
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

    # Check if the file exists
    if os.path.exists(previous_csv_path):
        previous_lines_df = pd.read_csv(previous_csv_path)
        # Match the IDs between the previous and current lines
        lines_df = assign_id_df(lines_df, previous_lines_df)
    else:
        # Assign new IDs to the lines
        lines_df['id'] = gen_id(lines_df)
        lines_df['line_id'] = lines_df['id']

    # Save the lines
    save_lines_csv(
        lines_df=lines_df,
        output_dir=os.path.join(exp_dir, date_str),
        file_name=f'{time_str}.csv'
    )


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
