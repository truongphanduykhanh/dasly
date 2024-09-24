"""Provides the main functions to run the Dasly pipeline for the Aastfjordbrua
dataset."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-08-28'


import os
import logging
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from dasly.master import Dasly
from dasly.utils import (
    assign_id_df,
    save_lines_csv,
    add_subtract_dt,
    get_date_time,
    gen_id,
    create_connection_string,
    read_sql,
    write_sql,
    table_exists,
    extract_elements,
    drop_table
)


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load the parameters
yaml_path = 'config_aastfjordbrua.yml'
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)


def dasly_core(
    file_path: str,
    lowpass_filter_freq: float = 0.5,
    decimate_t_rate: int = 10,
    gaussian_smooth_s1: float = 0.5,
    gaussian_smooth_s2: float = 1,
    gaussian_smooth_std_s: float = 0.1,
    binary_threshold: float = 1e-7,
    hough_transform_speed_res: int = 0.5,
    hough_transform_length_meters: int = 500,
    dbscan_eps_seconds: int = 1
) -> pd.DataFrame:
    """Run the Dasly core algorithms.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.

    Returns:
        pd.DataFrame: The DataFrame containing the detected lines.
    """
    lines = np.empty((0, 4))  # create an empty array to store the lines

    # Load the data
    ###########################################################################

    date, time = get_date_time(file_path)
    start = add_subtract_dt(
        f'{date} {time}',
        - (params['dasly']['batch'] + params['hdf5_file_length'])
    )
    file_paths_add, _, _ = simpleDASreader.find_DAS_files(
        experiment_path=params['input_dir'],
        start=start,
        duration=params['dasly']['batch'] + 2 * params['hdf5_file_length'],
        show_header_info=False
    )
    file_paths = extract_elements(
        lst=file_paths_add,
        num=int(params['dasly']['batch'] / params['hdf5_file_length']),
        last_value=file_path
    )

    # channel
    chIndex_all = np.arange(0, 800)
    chIndex_remove1 = np.arange(0, 36)
    chIndex_remove2 = np.arange(365, 387)
    chIndex_remove3 = np.arange(751, 800)
    chIndex_remove = np.concatenate((
        chIndex_remove1, chIndex_remove2, chIndex_remove3))
    chIndex = np.setdiff1d(chIndex_all, chIndex_remove)

    das = Dasly()
    das.load_data(
        file_paths=file_paths,
        integrate=params['integrate'],
        chIndex=chIndex,
        reset_channel_idx=True
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
        speed_res=hough_transform_speed_res,
        length_meters=hough_transform_length_meters)
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
        speed_res=hough_transform_speed_res,
        length_meters=hough_transform_length_meters)
    if len(das.lines) > 0:
        # store forward lines
        mask = das.lines_df['speed_kmh'].to_numpy() < 0
        lines = np.concatenate((lines, das.lines[mask]), axis=0)

    # dbscan forward and backward lines
    ###########################################################################
    if len(lines) == 0:  # if there are no lines, exit the function from here
        return
    das.lines = lines  # assign the new lines to the das object
    das.dbscan(eps_seconds=dbscan_eps_seconds)  # group close lines
    return das.lines_df


def dasly_pipeline_csv(
    file_path: str,
    lowpass_filter_freq: float = 0.5,
    decimate_t_rate: int = 10,
    gaussian_smooth_s1: float = 0.5,
    gaussian_smooth_s2: float = 1,
    gaussian_smooth_std_s: float = 0.1,
    binary_threshold: float = 1e-7,
    hough_transform_speed_res: int = 0.5,
    hough_transform_length_meters: int = 500,
    dbscan_eps_seconds: int = 1
) -> None:
    """Run the Dasly pipeline and store output as csv files. This function is
    called when a new HDF5 file is detected. Refer to the `HDF5EventHandler`
    class for more information.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.
    """
    # Get the experiment directory, date, and time
    date_str, time_str = get_date_time(file_path)

    lines_df = dasly_core(
        file_path=file_path,
        lowpass_filter_freq=lowpass_filter_freq,
        decimate_t_rate=decimate_t_rate,
        gaussian_smooth_s1=gaussian_smooth_s1,
        gaussian_smooth_s2=gaussian_smooth_s2,
        gaussian_smooth_std_s=gaussian_smooth_std_s,
        binary_threshold=binary_threshold,
        hough_transform_speed_res=hough_transform_speed_res,
        hough_transform_length_meters=hough_transform_length_meters,
        dbscan_eps_seconds=dbscan_eps_seconds
    )
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


def dasly_pipeline_db(
    file_path: str,
    lowpass_filter_freq: float = 0.5,
    decimate_t_rate: int = 10,
    gaussian_smooth_s1: float = 0.5,
    gaussian_smooth_s2: float = 1,
    gaussian_smooth_std_s: float = 0.1,
    binary_threshold: float = 1e-7,
    hough_transform_speed_res: int = 0.5,
    hough_transform_length_meters: int = 500,
    dbscan_eps_seconds: int = 1
) -> None:
    """Run the Dasly pipeline and store output in a database. This function is
    called when a new HDF5 file is detected. Refer to the `HDF5EventHandler`
    class for more information.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.
    """
    # Get the experiment directory, date, and time
    date_str, time_str = get_date_time(file_path)

    lines_df = dasly_core(
        file_path=file_path,
        lowpass_filter_freq=lowpass_filter_freq,
        decimate_t_rate=decimate_t_rate,
        gaussian_smooth_s1=gaussian_smooth_s1,
        gaussian_smooth_s2=gaussian_smooth_s2,
        gaussian_smooth_std_s=gaussian_smooth_std_s,
        binary_threshold=binary_threshold,
        hough_transform_speed_res=hough_transform_speed_res,
        hough_transform_length_meters=hough_transform_length_meters,
        dbscan_eps_seconds=dbscan_eps_seconds
    )
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


def dasly_pipeline(
    file_path: str,
    lowpass_filter_freq: float = 0.5,
    decimate_t_rate: int = 10,
    gaussian_smooth_s1: float = 0.5,
    gaussian_smooth_s2: float = 1,
    gaussian_smooth_std_s: float = 0.1,
    binary_threshold: float = 1e-7,
    hough_transform_speed_res: int = 0.5,
    hough_transform_length_meters: int = 500,
    dbscan_eps_seconds: int = 1
) -> Callable:
    """Run the Dasly pipeline and store output in either csv files or a
    database. This function is called when a new HDF5 file is detected. Refer
    to the `HDF5EventHandler` class for more information.

    Args:
        file_path (str): The path to the HDF5 file that triggers the event.

    Returns:
        Callable: The function to run the Dasly pipeline.
    """
    if params['output_type'] == 'db':
        dasly_pipeline_db(
            file_path=file_path,
            lowpass_filter_freq=lowpass_filter_freq,
            decimate_t_rate=decimate_t_rate,
            gaussian_smooth_s1=gaussian_smooth_s1,
            gaussian_smooth_s2=gaussian_smooth_s2,
            gaussian_smooth_std_s=gaussian_smooth_std_s,
            binary_threshold=binary_threshold,
            hough_transform_speed_res=hough_transform_speed_res,
            hough_transform_length_meters=hough_transform_length_meters,
            dbscan_eps_seconds=dbscan_eps_seconds
        )
    else:
        dasly_pipeline_csv(
            file_path=file_path,
            lowpass_filter_freq=lowpass_filter_freq,
            decimate_t_rate=decimate_t_rate,
            gaussian_smooth_s1=gaussian_smooth_s1,
            gaussian_smooth_s2=gaussian_smooth_s2,
            gaussian_smooth_std_s=gaussian_smooth_std_s,
            binary_threshold=binary_threshold,
            hough_transform_speed_res=hough_transform_speed_res,
            hough_transform_length_meters=hough_transform_length_meters,
            dbscan_eps_seconds=dbscan_eps_seconds
        )


###############################################################################
###############################################################################
###############################################################################
if __name__ == "__main__":

    import optuna
    from optuna.samplers import TPESampler

    from dasly.simpledas import simpleDASreader
    from dasly.loss import timestamp_dist, loss_fn

    exp_dir = '/media/kptruong/yellow02/Aastfjordbrua/Aastfjordbrua/'
    start = '20231005 080700'
    # end = '20231005 110600'

    file_paths, _, _ = simpleDASreader.find_DAS_files(
        experiment_path=params['input_dir'],
        start=start,
        duration=3*60*60 - 1*60,  # 3 hours - 1 minute
        # duration=60,  # 3 hours - 1 minute
        show_header_info=False
    )

    def calculate_loss(
        lowpass_filter_freq: float = 0.5,
        decimate_t_rate: int = 10,
        gaussian_smooth_s1: float = 0.5,
        gaussian_smooth_s2: float = 1,
        gaussian_smooth_std_s: float = 0.1,
        binary_threshold: float = 1e-7,
        hough_transform_speed_res: int = 0.5,
        hough_transform_length_meters: int = 500,
        dbscan_eps_seconds: int = 1
    ):

        for file_path in file_paths:
            dasly_pipeline(
                file_path=file_path,
                lowpass_filter_freq=lowpass_filter_freq,
                decimate_t_rate=decimate_t_rate,
                gaussian_smooth_s1=gaussian_smooth_s1,
                gaussian_smooth_s2=gaussian_smooth_s2,
                gaussian_smooth_std_s=gaussian_smooth_std_s,
                binary_threshold=binary_threshold,
                hough_transform_speed_res=hough_transform_speed_res,
                hough_transform_length_meters=hough_transform_length_meters,
                dbscan_eps_seconds=dbscan_eps_seconds
            )

        connection_string = create_connection_string(
            endpoint=params['database']['endpoint'],
            database=params['database']['database'],
            db_username=os.getenv('POSTGRESQL_USERNAME'),
            db_password=os.getenv('POSTGRESQL_PASSWORD'),
            type=params['database']['type'],
            dbapi=params['database']['dbapi'],
            port=params['database']['port']
        )

        if not table_exists(
            table_name=params['database']['table'],
            connection_string=connection_string
        ):
            return 1e6  # return a large loss if the table does not exist

        query = f'SELECT * FROM {params["database"]["table"]};'
        lines_df = read_sql(query, connection_string)

        # drop_table(params["database"]["table"], connection_string)

        coordinates = (
            lines_df
            .loc[lambda df: df.groupby('line_id')['created_at'].idxmax()]
            .loc[lambda df: df['speed_kmh'].abs().between(40, 120)]
            .loc[:, ['s1', 't1', 's2', 't2']]
            .to_numpy()
        )

        # Create a boolean mask where column 0 is greater than column 2
        mask = coordinates[:, 0] > coordinates[:, 2]

        # Create a copy of the array to swap columns based on the mask
        swapped_array = coordinates.copy()

        # Swap columns (0, 1) with (2, 3) wherever the mask is True
        swapped_array[mask, 0:2], swapped_array[mask, 2:4] = (
            coordinates[mask, 2:4],
            coordinates[mask, 0:2]
        )

        coordinates_swapped = (
            pd.DataFrame(swapped_array, columns=['s1', 't1', 's2', 't2'])
            .sort_values('t2')
            .to_numpy()
        )
        y_pred = coordinates_swapped[:, -1]

        y_true = pd.read_excel('data/logs/vehicle-logs.xlsx')
        y_true = (
            y_true
            .loc[lambda df: df['No signal'] != 1]
            .Time
            .to_numpy()
        )

        # Compute the loss
        a = 3
        b = 3
        dist = timestamp_dist
        L = loss_fn(y_true, y_pred, a, b, dist)

        return L

    ###########################################################################
    # Optuna
    connection_string = create_connection_string(
        endpoint=params['database']['endpoint'],
        database=params['database']['database'],
        db_username=os.getenv('POSTGRESQL_USERNAME'),
        db_password=os.getenv('POSTGRESQL_PASSWORD'),
        type=params['database']['type'],
        dbapi=params['database']['dbapi'],
        port=params['database']['port']
    )

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="my_study33",
        storage=connection_string,
        load_if_exists=False,
        direction="minimize",
        sampler=sampler
    )

    def objective(trial):
        # param = {
        #     'lowpass_filter_freq': trial.suggest_categorical(
        #         'lowpass_filter_freq', [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        #     ),
        #     'decimate_t_rate': trial.suggest_categorical(
        #         'decimate_t_rate', [1, 2, 4, 5, 8, 10, 20, 25]
        #     ),
        #     'gaussian_smooth_s1': params['gaussian_smooth']['s1'],
        #     'gaussian_smooth_s2': params['gaussian_smooth']['s2'],
        #     'gaussian_smooth_std_s': params['gaussian_smooth']['std_s'],
        #     'binary_threshold': trial.suggest_float(
        #         'binary_threshold', 1e-8, 3e-8
        #     ),
        #     'hough_transform_speed_res': params['hough_transform']['speed_res'],
        #     'hough_transform_length_meters':
        #         params['hough_transform']['length_meters'],
        #     'dbscan_eps_seconds': params['dbscan_eps_seconds']
        # }
        param = {
            'lowpass_filter_freq': params['lowpass_filter_freq'],
            'decimate_t_rate': params['decimate_t_rate'],
            'gaussian_smooth_s1': params['gaussian_smooth']['s1'],
            'gaussian_smooth_s2': params['gaussian_smooth']['s2'],
            'gaussian_smooth_std_s': params['gaussian_smooth']['std_s'],
            'binary_threshold': params['binary_threshold'],
            'hough_transform_speed_res': params['hough_transform']['speed_res'],
            'hough_transform_length_meters':
                params['hough_transform']['length_meters'],
            'dbscan_eps_seconds': params['dbscan_eps_seconds']
        }
        return calculate_loss(**param)

    study.optimize(objective, n_trials=1)
