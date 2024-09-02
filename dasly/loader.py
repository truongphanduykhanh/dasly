"""Provides convenient methods to load DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-06-01'

import logging
import logging.config
from typing import Union
from datetime import datetime, timedelta, time
import re
import os

import numpy as np
import pandas as pd

from dasly.simpledas import simpleDASreader
from dasly.utils import get_date_time


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
        tuple[datetime, int, datetime]: start, duration, end
    """
    # Check if two and only two out of three are inputted
    if (start is None) + (duration is None) + (end is None) != 1:
        raise ValueError('The function accepts two and only two out of '
                         + 'three (start, end, duration)')
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
    else:  # start is None
        start = end - timedelta(seconds=duration)

    return start, duration, end


def get_file_paths(
    folder_path: str,
    start: Union[str, datetime] = None,
    duration: int = None,
    start_exact_second: bool = True
) -> list[str]:
    """Get hdf5 files paths given the folder and time constraints.

    Args:
        folder_path (str): Experiment folder. Must inlcude date folders in
            right next level.
        start (Union[str, datetime], optional): Start time. If string, must
            be in format YYMMDD HHMMSS. If start and duration are not given, it
            will list all hdf5 files in folder_path. Defaults to None.
        duration (int, optional): Duration of the time in seconds. If start and
            duration are not given, it will list all hdf5 files in folder_path.
            Defaults to None.
        start_exact_second (bool, optional): If True, the file paths will
            ensure to include the file that starts at begining second of
            start time, and up tp end time (exclusive). If False, the file
            paths may not start the at the begining second of start time
            (some milliseconds after) and duration is ensured (so the
            actual end time my be some milliseconds after). This argument
            is especially useful when deploying because it will not load
            one additional file. Defaults to True.

    Returns:
        list[str]: HDF5 files paths.
    """
    # Get all hdf5 files paths if no time constraints
    ###########################################################################
    def list_hdf5_files(folder_path: str) -> list[str]:
        """List all hdf5 files in the folder path.

        Args:
            folder_path (str): Folder path.

        Returns:
            list[str]: List of hdf5 files paths.
        """
        hdf5_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.hdf5'):
                    hdf5_files.append(os.path.join(root, file))
        return hdf5_files

    if start is None and duration is None:
        file_paths = list_hdf5_files(folder_path)
        return file_paths

    # Get hdf5 files paths given the time constraints
    ###########################################################################
    file_paths, _, _ = simpleDASreader.find_DAS_files(
        experiment_path=folder_path,
        start=start,
        duration=duration,
        show_header_info=False
    )

    # Be careful with simpleDASreader.find_DAS_files(). It always ensures to
    # load the start (inclusive) and end (inclusive). On the other hand, data
    # in a file does not always start at the begining of the second. So
    # simpleDASreader.find_DAS_files() may include the file that before the
    # start time. For example, if the start is 093015 and the file does not
    # start at the begining of the second, it will include the file 093005.
    # If the start at the begining of the second and duration is 30 seconds, it
    # will include the file 093015 at 093045 (inclusive).

    def extract_date_time(file_path: str) -> str:
        """Extract date and time from the file path.

        Args:
            file_path (str): File path.

        Returns:
            str: Date and time in format 'YYYYMMDD HHMMSS'.
        """
        # Use regular expressions to extract the date and time parts
        date_match = re.search(r'/(\d{8})/', file_path)
        time_match = re.search(r'/(\d{6})\.hdf5$', file_path)
        date = date_match.group(1)
        time = time_match.group(1)
        result = f'{date} {time}'
        return result

    ###########################################################################
    # Check if start is datetime
    start_datetime = isinstance(start, datetime)
    # Convert start to string for comparison
    if start_datetime:
        start = start.strftime('%Y%m%d %H%M%S')

    # if duration is multiple of 10 and start is the exact second, drop the
    # last file because it is redundant. But instead of dropping directly the
    # last file, we slice the file paths in case there is not enough files,
    # i.e. the last file is not redundant.
    if duration % 10 == 0 and start == extract_date_time(file_paths[0]):
        n = duration // 10  # number of files to load
        file_paths = file_paths[: n]  # slice the file paths

    # if duration is multiple of 10 and start is not the exact second but the
    # start_exact_second is False (for deployment), we also slice the file
    # paths
    elif (
        duration % 10 == 0 and
        len(file_paths) > 2 and
        start == extract_date_time(file_paths[1]) and
        not start_exact_second
    ):  # for deployment
        n = duration // 10  # number of files to load
        file_paths = file_paths[1: n]  # slice the file paths

    # Convert start back to datetime
    if start_datetime:
        start = datetime.strptime(start, '%Y%m%d %H%M%S')

    # Verbose
    #######################################################################
    # first_file = get_date_time(file_paths[0])
    # last_file = get_date_time(file_paths[-1])

    # logger.info(
    #     f'{len(file_paths)} files, from {first_file} to {last_file}')
    return file_paths


class DASLoader:
    """Load DAS data."""

    def __init__(self) -> None:
        """Initialize instance.
        """
        # list of attributes
        self.file_paths: list[str] = None
        self.signal_raw: pd.DataFrame = None
        self.signal: pd.DataFrame = None
        self.t_rate: int = None  # temporal sampling rate, in sample per second
        self.s_rate: int = None  # spatial sampling rate, in sample per meter
        self.duration: int = None  # duration of the data, in seconds

    def _update_t_rate(self) -> None:
        """Update temporal sampling rate of the data.
        """
        time0 = self.signal.index[0]
        time1 = self.signal.index[1]
        if isinstance(time0, pd.Timestamp) and isinstance(time1, pd.Timestamp):
            # If both are pandas timestamps, just calculate the difference
            time_diff = time1 - time0
        elif isinstance(time0, time) and isinstance(time1, time):
            # If both are datetime.time objects (only time, not date),
            # convert them to datetime.datetime
            common_date = datetime.today()
            datetime0 = datetime.combine(common_date, time0)
            datetime1 = datetime.combine(common_date, time1)
            time_diff = datetime1 - datetime0
        # Calculate the time difference in seconds
        time_diff = time_diff.total_seconds()
        time_diff = np.abs(time_diff)
        t_rate = 1 / time_diff
        self.t_rate = t_rate

    def _update_s_rate(self) -> None:
        """Update spatial sampling rate of the data.
        """
        # Calculate the distance between two consecutive samples
        distance = self.signal.columns[1] - self.signal.columns[0]
        self.s_rate = 1 / distance

    def _update_duration(self) -> None:
        """Update duration of the data.
        """
        self.duration = len(self.signal) * (1 / self.t_rate)

    def reset(self) -> None:
        """Reset the defalt attribute signal to the raw signal.
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
        start_exact_second: bool = True,
        fmt: str = '%Y%m%d %H%M%S',
        suppress_date: bool = False,
        chIndex: Union[slice, list[int], np.ndarray] = None,
        integrate: bool = True,
        reset_channel_idx: bool = False
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
            start_exact_second (bool, optional): If True, the file paths will
                ensure to include the file that starts at begining second of
                start time, and up tp end time (exclusive). If False, the file
                paths may not start the at the begining second of start time
                (some milliseconds after) and duration is ensured (so the
                actual end time my be some milliseconds after). This argument
                is especially useful when deploying because it will not load
                one additional file. Defaults to True.
            format (str, optional): Format of start, end. Defaults to
                '%Y%m%d %H%M%S'.
            suppress_date (bool, optional): If True, keep only time as index if
                all the data come from the same date. Defaults to False.
            chIndex (Union[slice, list[int], np.ndarray], optional): Channel
                index. See more at simpleDASreader. Defaults to False.
            integrate (bool, optional): If True, integrate the data to get
                strain unit. Otherwise, the data is in strain rate unit. See
                more at simpleDASreader. Defaults to True
            reset_channel_idx (bool, optional): If True, reset the column names
                of the data. This is necessary when chIndex is used to remove
                redundant channels. Defaults to True.
        """
        slicing = False  # not slicing the data later if file_paths is given
        if not file_paths:  # if file_paths is not given
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
                duration=duration,
                start_exact_second=start_exact_second
            )
        first_file = get_date_time(file_paths[0])[-1]
        last_file = get_date_time(file_paths[-1])[-1]
        logger.info(
            f'{len(file_paths)} files, from {first_file} to {last_file}')
        # Load data
        signal = simpleDASreader.load_DAS_files(
            filepaths=file_paths,
            chIndex=chIndex,
            samples=None,  # default
            sensitivitySelect=0,  # default
            integrate=integrate,
            unwr=False,  # default
            spikeThr=None,  # default
            userSensitivity=None  # default
        )
        signal = pd.DataFrame(signal)
        # reset column names. this is necessary when chIndex is used to remove
        # redundant channels
        if reset_channel_idx:
            signal.columns = range(signal.shape[1])
        # Slice the data
        if slicing and start_exact_second:
            signal = signal.loc[start:end]  # extact only the range start-end
            signal = signal.iloc[:-1]  # drop the last record (new second)
        self.start = np.min(signal.index)
        self.end = np.max(signal.index)
        # if the data is within one day, just keep the time as index
        # because keeping date makes the index unnecessarily long
        if suppress_date and pd.Series(signal.index).dt.date.nunique() == 1:
            signal.index = pd.to_datetime(signal.index).time
        self.signal_raw = signal  # immutable attribute
        self.signal = self.signal_raw.copy()  # mutable attribute
        self.file_paths = file_paths
        self._update_t_rate()
        self._update_s_rate()
        self._update_duration()
