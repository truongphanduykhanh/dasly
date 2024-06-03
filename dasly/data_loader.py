"""Provides convenient process to load DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-06-01'

import logging
import logging.config
from typing import Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dasly.simpledas import simpleDASreader

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def infer_time(
    start: Union[str, datetime] = None,
    duration: int = None,
    end: Union[str, datetime] = None,
    fmt: str = '%Y%m%d %H%M%S'
) -> tuple[datetime, int, datetime]:
    """Infer start, end or duration from the other two of them.

    Infer start if duration and end are provided. Infer duration if start
    and end are provided. Infer end if start and duration are provided.

    Args:
        start (Union[str, datetime], optional): Start time. If string, must
            be in format fmt. Defaults to None.
        duration (int, optional): Duration of the time in seconds. Defaults
            to None.
        end (Union[str, datetime], optional): End time. If string, must
            be in format fmt. Defaults to None.
        fmt (str, optional): Format of start and end. Defaults to
            '%Y%m%d %H%M%S'.

    Raises:
        ValueError: The function accepts two and only two out of three
            (start, duration, end)

    Returns:
        tuple[datetime, int, datetime]: start, end, duration
    """
    # Check if two and only two out of three are inputted
    if (start is None) + (duration is None) + (end is None) != 1:
        raise ValueError("The function accepts two and only two out of "
                         "three (start, end, duration)")
    # Convert string to datetime
    if isinstance(start, str):
        start = datetime.strptime(start, fmt)
    if isinstance(end, str):
        end = datetime.strptime(end, fmt)
    # Infer start, end, duration
    if end is None:
        end = start + timedelta(seconds=duration)
    elif duration is None:
        duration = (end - start).seconds
    elif start is None:
        start = end - timedelta(seconds=duration)

    return start, duration, end


def get_file_paths(
    folder_path: str,
    start: Union[str, datetime] = None,
    duration: int = None
) -> list[str]:
    """Get hdf5 files paths given the folder and time constraints.

    Args:
        folder_path (str): Experiment folder. Must inlcude date folders in
            right next level.
        start (Union[str, datetime], optional): Start time. If string, must
            be in format YYMMDD HHMMSS, specify in argument format
            otherwise. Defaults to None.
        duration (int, optional): Duration of the time in seconds. Defaults
            to None.

    Returns:
        list[str]: HDF5 files paths.
    """
    # Get file paths
    #######################################################################
    file_paths, _, _ = simpleDASreader.find_DAS_files(
        experiment_path=folder_path,
        start=start,
        duration=duration,
        show_header_info=False
    )
    # Verbose
    #######################################################################
    first_file = file_paths[0].split('/')[-1].split('.')[0]
    last_file = file_paths[-1].split('/')[-1].split('.')[0]
    logger.info(
        f'{len(file_paths)} files, from {first_file} to {last_file}')
    return file_paths


class DataLoader:

    def __init__(self, ch_space: float = 1) -> None:
        """Initialize instance.
        """
        # list of attributes
        self.file_paths: list[str] = None
        self.signal_raw: pd.DataFrame = None
        self.signal: pd.DataFrame = None
        self.t_rate: int = None  # temporal sampling rate, in sample per second
        self.s_rate: int = None  # spatial sampling rate, in sample per meter
        self.duration: int = None  # duration of the data, in seconds
        self.ch_space = ch_space  # distance between two channels, in meters

    def _update_t_rate(self) -> None:
        """Update temporal sampling rate of the data.
        """
        time_0 = self.signal.index[0]
        time_1 = self.signal.index[1]
        # Create datetime objects with a common date
        common_date = datetime.today()
        datetime0 = datetime.combine(common_date, time_0)
        datetime1 = datetime.combine(common_date, time_1)
        # Calculate the time difference
        time_diff = datetime1 - datetime0
        time_diff = time_diff.total_seconds()
        time_diff = np.abs(time_diff)
        t_rate = 1 / time_diff
        self.t_rate = t_rate

    def _update_s_rate(self) -> None:
        """Update spatial sampling rate of the data.
        """
        # Calculate the distance between two consecutive samples
        distance = self.signal.columns[1] - self.signal.columns[0]
        distance = np.abs(distance) * self.ch_space
        self.s_rate = 1 / distance

    def _update_duration(self) -> None:
        """Update duration of the data.
        """
        self.duration = len(self.signal) * (1 / self.t_rate)

    def reset(self) -> None:
        """Reset all attributes and transformations on signal.
        """
        self.signal = self.signal_raw.copy()
        self._update_t_rate()
        self._update_s_rate()
        self._update_duration()

    def load_data(
        self,
        file_paths: list[str] = None,
        folder_path: str = None,
        start: Union[str, datetime] = None,
        duration: int = None,
        end: Union[str, datetime] = None,
        fmt: str = '%Y%m%d %H%M%S'
    ) -> None:
        """Load data to the instance.

        Args:
            file_paths (str): File paths. If folder_path and file_paths are
                inputted, prioritize to use file_paths. Defaults to None.
            folder_path (str): Experiment folder. Must inlcude date folders in
                right next level. Defaults to None.
            start (Union[str, datetime], optional): Start time. If string, must
                be in format YYMMDD HHMMSS, specify in argument format
                otherwise. Defaults to None.
            duration (int, optional): Duration of the time in seconds. Defaults
                to None.
            end (Union[str, datetime], optional): End time. If string, must be
                in format YYMMDD HHMMSS, specify in argument format otherwise.
                Defaults to None.
            format (str, optional): Format of start, end. Defaults to
                '%Y%m%d %H%M%S'.
        """
        slicing = False  # not slicing the data later if file_paths is given
        if not file_paths:
            slicing = True  # use for slicing the data later
            # Infer start, end, duration
            start, duration, end = infer_time(
                start=start,
                duration=duration,
                end=end,
                fmt=fmt
            )
            # Get file paths
            file_paths = get_file_paths(
                folder_path=folder_path,
                start=start,
                duration=duration
            )
        # Load data
        signal = simpleDASreader.load_DAS_files(
            filepaths=file_paths,
            chIndex=None,  # default
            samples=None,  # default
            sensitivitySelect=0,  # default
            integrate=False,  # change to False
            unwr=True,  # change to True
            spikeThr=None,  # default
            userSensitivity=None  # default
        )
        signal = pd.DataFrame(signal)
        # Slice the data
        if slicing:
            signal = signal.loc[start:end]  # extact only the range start-end
            signal = signal.iloc[:-1]  # drop the last record (new second)
        self.start = np.min(signal.index)
        self.end = np.max(signal.index)
        # if the data is within one day, just keep the time as index
        # because keeping date makes the index unnecessarily long
        if pd.Series(signal.index).dt.date.nunique() == 1:
            signal.index = pd.to_datetime(signal.index).time
        self.signal_raw = signal  # immutable attribute
        self.signal = self.signal_raw.copy()  # mutable attribute
        self.file_paths = file_paths
        self._update_t_rate()
        self._update_s_rate()
        self._update_duration()
