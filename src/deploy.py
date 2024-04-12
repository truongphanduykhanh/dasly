import os
import time
from datetime import datetime, timedelta

import pandas as pd
import re
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src import dasly


yaml_path = 'src/config.yml'

# Open and read the YAML file
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)

# Access parameters from the YAML file
input_dir = params['input_dir']
output_dir = params['output_dir']
batch_hdf5 = params['batch_hdf5']
batch = params['dasly']['batch']
batch_gap = params['dasly']['batch_gap']


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
    das = dasly.Dasly()
    das.load_data(
        folder_path=input_dir,
        start=first_file_dt,
        duration=batch
    )
    # forward filter
    #######################################################################
    das.lowpass_filter(0.5)
    das.decimate(sampling_rate=6)
    das.gauss_filter(85, 90)
    das.sobel_filter()
    das.binary_filter(by_column=False, threshold=2.5e-8)
    das.hough_transform()
    if das.lines is not None:
        das.lines = das.lines.loc[lambda df: df['speed'] > 0]
        detect_lines = pd.concat([  # concat new lines to the data frame
            detect_lines,
            das.lines.assign(batch=das.start)
        ])

    # backward filter
    #######################################################################
    das.reset()
    das.lowpass_filter(0.5)
    das.decimate(sampling_rate=6)
    das.gauss_filter(-90, -85)
    das.sobel_filter()
    das.binary_filter(by_column=False, threshold=2.5e-8)
    das.hough_transform()
    if das.lines is not None:
        das.lines = das.lines.loc[lambda df: df['speed'] < 0]
        detect_lines = pd.concat([  # concat new lines to the data frame
            detect_lines,
            das.lines.assign(batch=das.start)
        ])

    # export to csv file
    ###########################################################################
    start = pd.Timestamp(das.start).strftime('%Y%m%d%H%M%S')
    end = pd.Timestamp(das.end).strftime('%Y%m%d%H%M%S')
    file_name = f'lines_{start}_{end}.csv'
    output_file_path = os.path.join(output_dir, file_name)
    detect_lines.to_csv(output_file_path, index=False)


# Define the number of triggers before running dasly
# triggers_before_run = overlap / batch_hdf5
# trigger_count = 0  # Initialize trigger counter


# Define the event handler class
class MyHandler(FileSystemEventHandler):
    def __init__(self, event_thresh):
        super().__init__()
        self.event_thresh = event_thresh
        self.event_count = 0

    def on_created(self, event):
        print(f'New file: {event.src_path}')
        print(f'global trigger_count {self.event_count}')
        self.event_count += 1
        print(f'trigger_count += 1 {self.event_count}')
        if self.event_count % self.event_thresh == 0:
            time.sleep(1)  # Wait for the file to be completely written
            file_dt = extract_dt(event.src_path)
            first_file_dt = add_subtract_dt(file_dt, -batch+10)
            run_dasly(
                input_dir=input_dir,
                output_dir=output_dir,
                batch=batch,
                first_file_dt=first_file_dt
            )


# Initialize the Observer and EventHandler
event_handler = MyHandler(event_thresh=batch_gap/batch_hdf5)
observer = Observer()
observer.schedule(event_handler, input_dir, recursive=True)

print("Input directory: ", input_dir)

# Start the observer
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
