"""Provides convenient methods to tunning the hyperparameters"""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-08-23'


import os
import logging

import numpy as np
import pandas as pd
import yaml

from dasly.master import Dasly
from dasly.utils import assign_id


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
yaml_path = 'config.yml'

# Open and read the YAML file
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)


# Access parameters from the YAML file
input_dir = params['input_dir']
start_exact_second = params['start_exact_second']
integrate = params['integrate']

database_type = params['database']['database_type']
dbapi = params['database']['dbapi']
endpoint = params['database']['endpoint']
port = params['database']['port']
database = params['database']['database']
database_table = params['database']['database_table']

hdf5_file_length = params['hdf5_file_length']
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


# Run dasly
###############################################################################
###############################################################################
###############################################################################
def run_dasly(
    folder_path: str,
    start: str,
    duration: int,
) -> pd.DataFrame:
    """Run the DASly algorithm on the input data.

    Args:

    Returns:

    """
    # Load the data
    ###########################################################################
    lines = np.empty((0, 4))  # create an empty array to store the lines
    das = Dasly()
    das.load_data(
        folder_path=folder_path,
        start=start,
        duration=batch,
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
    if len(das.lines) > 0:
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
    if len(das.lines) > 0:
        # store forward lines
        mask = das.lines_df['speed_kmh'].to_numpy() < 0
        lines = np.concatenate((lines, das.lines[mask]), axis=0)

    # dbscan forward and backward lines
    ###########################################################################
    if len(lines) == 0:  # if there are no lines, exit the function from here
        return
    das.lines = lines  # assign the new lines to the das object
    das.dbscan(eps_seconds=dbscan_eps_seconds)  # group similar lines together
    lines = das.lines
    lines_df = assign_id(das.lines_df)  # assign unique id to each line

    return lines_df


def assign_id_df(
    lines: pd.DataFrame,
    previous_lines: pd.DataFrame,
    id_col: str = 'line_id',
    s1_col: str = 's1',
    t1_col: str = 't1',
    s2_col: str = 's2',
    t2_col: str = 't2',
    dist: callable = None,
) -> pd.DataFrame:
    """Assign unique ID to each row n the data frame lines. Each line is
    defined by (s1, t1, s2, t2). Where (s1, t1) and (s2, t2) are the
    coordinates (in space and time) of the start and end points of the line,
    respectively. t1 is always smaller than t2 (t1 <= t2).

    Args:
        lines (pd.DataFrame): Current lines. Must have columns s1_col, t1_col,
            s2_col, t2_col.
        previous_lines (pd.DataFrame): Previous lines. Must have columns
            s1_col, t1_col, s2_col, t2_col and id_col.
        id_col (str): Name of the ID column in previous_lines. Default is
            'line_id'.
        s1_col (str): Name of the s1 column in lines and previous_lines.
            Default is 's1'.
        t1_col (str): Name of the t1 column in lines and previous_lines.
            Default is 't1'.
        s2_col (str): Name of the s2 column in lines and previous_lines.
            Default is 's2'.
        t2_col (str): Name of the t2 column in lines and previous_lines.
            Default is 't2'.
        dist (callable): Function to calculate the distance between two lines.

    Returns:
        pd.DataFrame: new column id_col is added to the lines data frame.
    """
    lines_coords = lines[[s1_col, t1_col, s2_col, t2_col]].to_numpy()
    previous_lines_coords = (
        previous_lines[[s1_col, t1_col, s2_col, t2_col]].to_numpy())
    
    # calculate the distance between each line and the previous lines
    

















