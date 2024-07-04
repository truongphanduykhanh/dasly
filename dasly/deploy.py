"""Provides realtime extrating information process from hdf5 files.
"""
__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-04-10'

import os
import time
import sys
from datetime import datetime, timedelta
import uuid

import pandas as pd
import numpy as np
import re
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.cluster import DBSCAN

from dasly import dasly_old


# Parameters from the YAML file
###############################################################################
yaml_path = 'config.yml'

# Open and read the YAML file
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)

# Access parameters from the YAML file
input_dir = params['input_dir']
output_dir = params['output_dir']
batch_hdf5 = params['batch_hdf5']
batch = params['dasly']['batch']
batch_gap = params['dasly']['batch_gap']

low_pass_freq = params['low_pass_freq']

decimate_sampling_rate = params['decimate']['sampling_rate']

gauss_s1 = params['gauss_filter']['s1']
gauss_s2 = params['gauss_filter']['s2']
gauss_std = params['gauss_filter']['std']

binary_threshold = params['binary_threshold']
dbscan_eps = params['dbscan_eps']


def extract_dt(path: str) -> str:
    """
    Extract the date and time from the input path with format
    /YYYYMMDD/dphi/HHMMSS.hdf5. Output format: YYYYMMDD HHMMSS
    """
    # Define the regular expression pattern
    pattern = r"/(\d{8})/dphi/(\d{6})\.hdf5$"

    # Search for the pattern in the input string
    match = re.search(pattern, path)

    # Extract the matched groups
    date_part = match.group(1)
    time_part = match.group(2)

    # Combine date and time parts
    result = f"{date_part} {time_part}"
    return result


def add_subtract_dt(t: str, x: int) -> str:
    """
    Add or subtract x seconds from the input time t with format YYYYMMDD HHMMSS
    """
    # Convert input string to datetime object
    input_dt = datetime.strptime(t, "%Y%m%d %H%M%S")

    # Add x seconds to the datetime object
    new_dt = input_dt + timedelta(seconds=x)

    # Format the new datetime object back into the same string format
    new_dt_str = new_dt.strftime("%Y%m%d %H%M%S")
    return new_dt_str


def run_dasly(input_dir: str, output_dir: str, batch: int, first_file_dt: str):
    """
    Run the daslly algorithm on the input directory and save the output to the
    output directory
    """
    detect_lines = pd.DataFrame()  # create an empty data frame
    das = dasly_old.Dasly()
    das.load_data(
        folder_path=input_dir,
        start=first_file_dt,  # start time YYYYMMDD HHMMSS
        duration=batch
    )
    # forward filter
    ###########################################################################
    das.lowpass_filter(low_pass_freq)
    das.decimate(sampling_rate=decimate_sampling_rate)
    das.gauss_filter(gauss_s1, gauss_s2, gauss_std)
    das.sobel_filter()
    das.binary_filter(by_column=False, threshold=binary_threshold)
    das.hough_transform(time=batch*0.5)
    if das.lines is not None:
        das.lines = das.lines.loc[lambda df: df['speed'] > 0]
        detect_lines = pd.concat([  # concat new lines to the data frame
            detect_lines,
            das.lines.assign(batch=das.start)
        ])

    # backward filter
    ###########################################################################
    das.reset()
    das.lowpass_filter(low_pass_freq)
    das.decimate(sampling_rate=decimate_sampling_rate)
    das.gauss_filter(-gauss_s2, -gauss_s1, gauss_std)
    das.sobel_filter()
    das.binary_filter(by_column=False, threshold=binary_threshold)
    das.hough_transform(time=batch*0.5)
    if das.lines is not None:
        das.lines = das.lines.loc[lambda df: df['speed'] < 0]
        detect_lines = pd.concat([  # concat new lines to the data frame
            detect_lines,
            das.lines.assign(batch=das.start)
        ])

    # dbscan
    ###########################################################################
    das.lines = detect_lines
    das.dbscan(eps=dbscan_eps)

    start = pd.Timestamp(das.start).strftime('%Y%m%d%H%M%S')
    end = pd.Timestamp(das.end).strftime('%Y%m%d%H%M%S')
    return start, end, das.lines


def assign_line_id(
    candidate_lines: pd.DataFrame,
    previous_lines: pd.DataFrame
) -> pd.DataFrame:
    """ Assign line IDs to the candidate lines based on the previous lines
    """
    previous_lines = previous_lines.assign(where='previous')
    candidate_lines = candidate_lines.assign(where='new')

    df = pd.concat([previous_lines, candidate_lines]).reset_index(drop=True)

    def __metric(x, y) -> float:
        distance = np.abs(np.array(x) - np.array(y)).mean()
        return distance

    reference = datetime.now()
    df = (
        df
        .assign(left_seconds=lambda df:
                (df['left'] - reference).dt.total_seconds())
        .assign(right_seconds=lambda df:
                (df['right'] - reference).dt.total_seconds())
        .assign(middle_seconds=lambda df:
                (df['middle'] - reference).dt.total_seconds())
    )
    if len(df) > 0:
        cluster = DBSCAN(
            eps=1,
            min_samples=1,
            metric=__metric
        )
        cluster = cluster.fit(
            df[['left_seconds', 'right_seconds', 'middle_seconds']])
        df = (
            df.assign(cluster=cluster.labels_)
            # .loc[lambda df: df['speed'].between(-120, 120)]
        )
        # Find rows with missing values in id
        missing_id_rows = df[df['id'].isnull()]
        for row in missing_id_rows.itertuples():
            index = row.Index
            cluster_value = row.cluster

            # Check if there are 2 or more rows with the same cluster value
            if df['cluster'].value_counts()[cluster_value] >= 2:
                # Get the non-missing value of id for the same cluster
                non_missing_values = df[
                    (df['cluster'] == cluster_value) & (~df['id'].isnull())]['id']
                if not non_missing_values.empty:
                    non_missing_value = non_missing_values.iloc[0]
                    df.at[index, 'id'] = non_missing_value
                else:
                    # If all values of id for the same cluster are missing,
                    # assign a unique UUID
                    unique_id = str(uuid.uuid4())
                    df.at[index, 'id'] = unique_id
            else:
                # If no similar cluster values, assign a unique UUID
                unique_id = str(uuid.uuid4())
                df.at[index, 'id'] = unique_id
    df = (
        df
        .loc[lambda df: df['where'] == 'new']
        .drop(columns=['left_seconds', 'right_seconds', 'middle_seconds', 'where'])
    )
    return df


def export_lines_to_csv(start, end, detect_lines, output_dir) -> None:
    # export to csv file
    ###########################################################################
    file_name = f'lines_{start}_{end}.csv'
    output_file_path = os.path.join(output_dir, file_name)
    detect_lines.to_csv(output_file_path, index=False)


# Define the event handler class
class MyHandler(FileSystemEventHandler):

    def __init__(self, event_thresh):
        """Initialize the event handler with an event threshold
        """
        super().__init__()
        self.event_thresh = event_thresh
        self.event_count = 0
        self.last_created = None
        self.previous_lines = pd.DataFrame().assign(id=None).assign(cluster=None)

    def on_any_event(self, event):
        """Event handler for any file system event
        """
        if event.is_directory:
            return
        if (
            # Check if the event is a new file
            event.event_type == 'moved'
            # Check if the file is an HDF5 file
            and (True if event.dest_path is not None else False)
            # Check if the file is not a duplicate (watchdog bug workaround)
            and event.dest_path != self.last_created
        ):
            time.sleep(1)  # Wait for the file to be completely written
            print(f'New hdf5: {event.dest_path}')
            self.last_created = event.dest_path  # Update the last created file
            self.event_count += 1
            # if self.event_count >= self.event_thresh:
            #     print('Runing dasly...')
            #     max_attempts = 600  # Maximum attempts if error occurs
            #     attempts = 1
            #     while attempts <= max_attempts:
            #         try:
            #             sys.stdout = open('/dev/null', 'w')  # Suppress print
            #             file_dt = extract_dt(event.dest_path)
            #             first_file_dt = add_subtract_dt(file_dt, -batch+10)
            #             start, end, lines = run_dasly(
            #                 input_dir=input_dir,
            #                 output_dir=output_dir,
            #                 batch=batch,
            #                 first_file_dt=first_file_dt
            #             )
            #             if (('previous_lines' not in globals()) or
            #                 ('previous_lines' not in locals())
            #                 ):
            #                 previous_lines = (
            #                     pd.DataFrame()
            #                     .assign(id=None)
            #                     .assign(cluster=None)
            #                 )
            #             print(f'previous line length: {len(previous_lines)}')
            #             print(f'current line length: {len(lines)}')
            #             lines_id = assign_line_id(lines, previous_lines)
            #             previous_lines = lines
            #             export_lines_to_csv(start, end, lines_id, output_dir)
            #             self.event_count = 0  # Reset the event count
            #             sys.stdout = sys.__stdout__  # Restore print output
            #             break  # Exit the loop if successful
            #         except Exception:
            #             attempts += 1
            #             time.sleep(3)  # Wait for 1 seconds before retrying
            #     else:
            #         print("There is an error in the process. Stopping...")

            if self.event_count >= self.event_thresh:
                print('Runing dasly...')
                # max_attempts = 600  # Maximum attempts if error occurs
                # attempts = 1
                # while attempts <= max_attempts:
                #     try:
                # sys.stdout = open('/dev/null', 'w')  # Suppress print
                file_dt = extract_dt(event.dest_path)
                first_file_dt = add_subtract_dt(file_dt, -batch+10)
                start, end, lines = run_dasly(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    batch=batch,
                    first_file_dt=first_file_dt
                )
                # if (('previous_lines' not in globals()) or
                #     ('previous_lines' not in locals())
                #     ):
                #     previous_lines = (
                #         pd.DataFrame()
                #         .assign(id=None)
                #         .assign(cluster=None)
                #     )
                print(f'previous line length: {len(self.previous_lines)}')
                print(f'current line length: {len(lines)}')
                if len(lines) + len(self.previous_lines) > 0:
                    lines_id = assign_line_id(lines, self.previous_lines)
                else:
                    lines_id = pd.DataFrame().assign(id=None).assign(cluster=None)
                self.previous_lines = lines_id
                export_lines_to_csv(start, end, lines_id, output_dir)
                self.event_count = 0  # Reset the event count
                # sys.stdout = sys.__stdout__  # Restore print output
                        # break  # Exit the loop if successful
                #     except Exception:
                #         attempts += 1
                #         time.sleep(3)  # Wait for 1 seconds before retrying
                # else:
                #     print("There is an error in the process. Stopping...")


# Initialize the Observer and EventHandler
event_handler = MyHandler(event_thresh=batch_gap/batch_hdf5)
observer = Observer()
observer.schedule(event_handler, input_dir, recursive=True)

print("Watching directory: ", input_dir)

# Start the observer
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
