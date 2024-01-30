"""Provides end-to-end flow to load, analyze and visualize DAS data.
"""
__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-01-02'

from typing import Union
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from scipy import fft
from scipy.signal import butter, sosfilt, decimate
from scipy.ndimage import gaussian_filter
import sklearn
from sklearn.cluster import DBSCAN
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from src import simpleDASreader, helper


sns.set()
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', 'is_categorical_dtype')
# seaborn v0.13.0 has been released with these updates
# https://github.com/mwaskom/seaborn/issues/3462


class Dasly:

    def __init__(self) -> None:
        print("Welcome to Dasly!")
        # list of attributes
        self.file_paths: list[str] = None
        self.signal_raw: pd.DataFrame = None
        self.signal: pd.DataFrame = None
        self.signal_decimate: pd.DataFrame = None
        self.events: np.ndarray = None
        self.events_df: pd.DataFrame = None
        self.cluster: sklearn.cluster._dbscan.DBSCAN = None
        self.sampling_rate: int = None
        self.duration: int = None
        self.data_type: str = None
        self.channel: float = 1.0  # meter between two consecutive channels

    def update_sampling_rate(self) -> None:
        self.sampling_rate = len(self.signal) / self.duration

    def reset(self) -> None:
        """Reset all transformations on signal
        """
        self.signal = self.signal_raw
        self.update_sampling_rate()

    # @staticmethod
    # def get_all_file_paths(folder_path: str) -> list[str]:
    #     """All hdf5 files in a folder

    #     Args:
    #         folder_path (str): Folder path

    #     Returns:
    #         list[str]: List of file paths
    #     """
    #     file_paths = []
    #     for root, dirs, files in os.walk(folder_path):
    #         for file in files:
    #             file_paths.append(os.path.join(root, file))
    #     file_paths = [file for file in file_paths if file.endswith('.hdf5')]
    #     return file_paths

    # @staticmethod
    # def get_file_paths(
    #     folder_path: str,
    #     start: Union[str, datetime] = None,
    #     end: Union[str, datetime] = None,
    #     duration: int = None,
    #     datatype: str = 'dphi',
    # ) -> list[str]:
    #     """Get file paths given the folder and time constraints.

    #     Args:
    #         folder_path (str): Experiment folder. Must inlcude date folders in
    #             right next level.
    #         start (Union[str, datetime], optional): Start time. If string, must
    #             be in format YYMMDD HHMMSS. Defaults to None.
    #         end (Union[str, datetime], optional): End time. If string, must be
    #             in format YYMMDD HHMMSS. Defaults to None.
    #         duration (int, optional): Duration of the time in seconds. Defaults
    #             to None.

    #     Returns:
    #         list[str]: File paths.
    #     """
    #     sub_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    #     try:
    #         datetime.strptime(sub_folders[0].split('/')[-1], '%Y%m%d')
    #     except (ValueError, IndexError):
    #         warnings.warn(f"""Data folder does not follow standard format
    #         folder/YYYYMMDD/{datatype}/HHMMSS.hdf5""", stacklevel=2)
    #         file_paths = Dasly.get_all_file_paths(folder_path)
    #         # Verbose
    #         first_file = file_paths[0].split('/')[-1].split('.')[0]
    #         last_file = file_paths[-1].split('/')[-1].split('.')[0]
    #         print(
    #             f'{len(file_paths)} files,from {first_file} to {last_file}')
    #         return file_paths

    #     # determine start, end, duration
    #     start, end, duration = Dasly.infer_time(
    #         start=start,
    #         end=end,
    #         duration=duration
    #     )
    #     start_date = start.strftime('%Y%m%d')
    #     end_date = end.strftime('%Y%m%d')
    #     start_date_time = start.strftime('%Y%m%d %H%M%S')
    #     end_date_time = end.strftime('%Y%m%d %H%M%S')

    #     # determine file paths within the dates
    #     sub_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    #     dates = [f.split('/')[-1] for f in sub_folders]
    #     dates = [d for d in dates if d >= start_date and d <= end_date]
    #     file_paths = []
    #     for date in dates:
    #         folder_path_date = os.path.join(folder_path, date, datatype)
    #         file_paths_date = Dasly.get_all_file_paths(folder_path_date)
    #         file_paths.extend(file_paths_date)

    #     # filter file paths within the time
    #     file_dates = [file.split('/')[-3] for file in file_paths]
    #     file_times = [file.split('/')[-1].split('.')[0] for file in file_paths]
    #     file_dates_times = [
    #         ' '.join([date, time])
    #         for date, time in zip(file_dates, file_times)]
    #     file_filter = [
    #         x for x in file_dates_times
    #         if x >= start_date_time and x < end_date_time]
    #     file_filter.sort()
    #     if file_filter[0] > start_date_time:  # not cover yet the start time
    #         min_idx = file_dates_times.index(file_filter[0])
    #         if min_idx != 0:  # if the min_idx is already 0, ignore
    #             file_filter.insert(0, file_dates_times[min_idx - 1])

    #     # re-create the path from date time
    #     file_filter_split = [s.split(' ') for s in file_filter]
    #     file_filter_paths = [
    #         os.path.join(folder_path, s[0], datatype, s[1]) + '.hdf5'
    #         for s in file_filter_split]
    #     file_filter_paths

    #     # Verbose
    #     first_file = file_filter_paths[0].split('/')[-1].split('.')[0]
    #     last_file = file_filter_paths[-1].split('/')[-1].split('.')[0]
    #     print(
    #         f'{len(file_filter_paths)} files,from {first_file} to {last_file}')

    #     return file_filter_paths

    @staticmethod
    def infer_time(
        start: Union[str, datetime] = None,
        duration: int = None,
        end: Union[str, datetime] = None,
        format: str = '%Y%m%d %H%M%S'
    ) -> tuple[datetime, int, datetime]:
        """Infer start if duration and end are provided. Infer duration if
        start and end are provided. Infer end if start and duration are
        provideds.

        Args:
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

        Raises:
            ValueError: The function accepts two and only two out of three
                (start, duration, end)

        Returns:
            tuple[datetime, int, datetime]: start, end, duration
        """
        # Check number of argument
        #######################################################################
        if (start is None) + (duration is None) + (end is None) != 1:
            raise ValueError("The function accepts two and only two out of \
                             three (start, end, duration)")
        # Transform string to datetime type if string is inputted
        #######################################################################
        if isinstance(start, str):
            start = datetime.strptime(start, format)
        if isinstance(end, str):
            end = datetime.strptime(end, format)
        # Infer end or duration or start
        #######################################################################
        if end is None:
            end = start + timedelta(seconds=duration)
        elif duration is None:
            duration = (end - start).seconds
        elif start is None:
            start = end - timedelta(seconds=duration)

        return start, duration, end

    @staticmethod
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
        print(f'{len(file_paths)} files, from {first_file} to {last_file}')
        return file_paths

    def load_data(
        self,
        folder_path: str,
        start: Union[str, datetime] = None,
        duration: int = None,
        end: Union[str, datetime] = None,
        format: str = '%Y%m%d %H%M%S'
    ) -> None:
        """Load data to the instance.

        Args:
            folder_path (str): Experiment folder. Must inlcude date folders in
                right next level.
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
        # Infer time
        #######################################################################
        start, duration, end = Dasly.infer_time(
            start=start,
            duration=duration,
            end=end
        )
        self.start = start
        self.duration = duration
        self.end = end
        # Get files paths
        #######################################################################
        file_paths = Dasly.get_file_paths(
            folder_path=folder_path,
            start=start,
            duration=duration
        )
        self.file_paths = file_paths
        # Load files
        #######################################################################
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
        # Transform dataframe
        #######################################################################
        signal = signal[start:end]  # extact only the range start-end
        signal = signal.iloc[:-1]  # drop the last record (new second already)
        # if the data is within one day, just keep the time as index
        # because keeping date makes the index unnecessarily long
        if pd.Series(signal.index).dt.date.nunique() == 1:
            signal.index = pd.to_datetime(signal.index).time
        self.signal_raw = signal  # immutable attribute
        self.signal = signal  # mutable attribute, that can be changed later
        self.data_type = 'float'
        self.update_sampling_rate()

    def bandpass_filter(
        self,
        low: float,
        high: float,
        order: int = 4,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Band-pass filter.

        Args:
            low (float): Lower bound frequency
            high (float): Upper bound frequency
            order (int, optional): Order of IIR filter. Defaults to 4.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        nyquist = 0.5 * self.sampling_rate
        low = low / nyquist
        high = high / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')
        signal_bandpass = sosfilt(sos, self.signal, axis=0)
        signal_bandpass = pd.DataFrame(
            signal_bandpass,
            index=self.signal.index
        )
        # return
        #######################################################################
        if inplace:
            self.signal = signal_bandpass
        else:
            return signal_bandpass

    def lowpass_filter(
        self,
        cutoff: float,
        order: int = 4,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Low-pass filter. Maintain only low frequency in the data.

        Args:
            cutoff (float): Cut-off frequency.
            order (int, optional): Order of IIR filter. Defaults to 4.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        nyquist = 0.5 * self.sampling_rate
        cutoff = cutoff / nyquist
        sos = butter(order, cutoff, btype='low', output='sos')
        signal_lowpass = sosfilt(sos, self.signal, axis=0)
        signal_lowpass = pd.DataFrame(signal_lowpass, index=self.signal.index)
        # return
        #######################################################################
        if inplace:
            self.signal = signal_lowpass
        else:
            return signal_lowpass

    def sample(
        self,
        factor: int,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Down-sampling according to time (rows).

        Args:
            factor (int): Factor would like to take sampling. Ex: value 5 will
                take average of every 5 rows.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # create antificial groups
        groups = np.arange(len(self.signal)) // factor
        # calculate means by group
        signal = self.signal.groupby(groups).mean()
        # take correct index
        signal.index = self.signal.iloc[::factor, :].index
        # return
        #######################################################################
        if inplace:
            self.signal = signal
            self.update_sampling_rate()
        else:
            return signal

    def decimate(
        self,
        factor: int = None,
        frequency: float = None,
        sampling_rate: int = None,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Decimating the data. The final sampling rate is adjusted to be a
        divisor of original sampling rate. E.g. 100 to 50, 25 or 5. There are
            three options to decimate:
            1. by factor
            2. by frequency
            3. by sampling_rate
            If by frequency, it automatically choose the smallest sampling rate
            that ensure the quality of the frequency (>= twice frequency). Note
            that this method different from method sample(). sample() simply
            takes the average of every n rows, regardless of the frequencies.

        Args:
            factor (int, optional): Downsampling factor.
            frequency (float, optional): Maintain >= frequency.
            sampling_rate (int, optional): Sampling rate.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # check arguments
        #######################################################################
        if (
            (factor is None) +
            (frequency is None) +
            (sampling_rate is None)
        ) != 2:
            raise ValueError(
                "The function accepts one and only one out of three \
                    (factor, frequency, sampling_rate)")

        # decimating
        #######################################################################
        if frequency is not None:
            sampling_rate = frequency * 2
        if factor is None:
            factor = self.sampling_rate / sampling_rate
        divisors = helper.find_divisors(int(self.sampling_rate))
        factor_adjusted = helper.largest_smaller_than_threshold(
            divisors, factor)

        # scipy.signal.decimate
        signal_decimate = decimate(
            x=self.signal,
            q=factor_adjusted,
            axis=0
        )
        # correct the index
        #######################################################################
        idx = slice(0, len(self.signal.index), factor_adjusted)
        idx = self.signal.index[idx]
        signal_decimate = pd.DataFrame(signal_decimate, index=idx)
        # return
        #######################################################################
        if inplace:
            self.signal = signal_decimate
            self.update_sampling_rate()
            print(f'Downsampling factor: {factor_adjusted:.0f}')
            print(f'New sampling rate: {self.sampling_rate:.0f}')
        else:
            return signal_decimate

    def binary_filter(
        self,
        quantile: float = 0.95,
        threshold: float = None,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Transform data to binary. First take absolute value. Then map to 1
            if greater than or equal to threshold, 0 otherwise.

        Args:
            quantile (float, optional): Quantile as a threshold.
                Either quantile or threshold is be inputed. Cannot input both.
                Defaults to 0.95.
            threshold (float, optional): Threshold value. Defaults
                to None.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # check arguments
        #######################################################################
        if (threshold is None) + (quantile is None) != 1:
            raise ValueError(
                "The function accepts one and only one out of two \
                    (threshold, quantile)")
        if threshold is None:
            threshold = np.quantile(np.abs(self.signal), quantile)
            print(f'Threshold: {threshold:.3g}')
        signal_binary = (np.abs(self.signal) >= threshold).astype(int)
        # return
        #######################################################################
        if inplace:
            self.signal = signal_binary
        else:
            return signal_binary

    def gray_filter(self, inplace: bool = True) -> Union[None, pd.DataFrame]:
        """Transform data to grayscale 0 to 255 using min-max scalling.

        Args:
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # min-max normalization
        signal_gray = np.abs(self.signal)
        signal_gray = ((signal_gray - np.min(signal_gray)) /
                       (np.max(signal_gray) - np.min(signal_gray)) * 255)
        signal_gray = signal_gray.round(0).astype(np.uint8)
        # return
        #######################################################################
        if inplace:
            self.signal = signal_gray
        else:
            return signal_gray

    def gauss_filter(
        self,
        beta: float = 20,  # 20 channels
        beta_alpha_factor: float = 25,  # 1 channel equals 1/25 seconds
        alpha: float = None
    ) -> None:
        """Apply 2d Gaussian filter.

        Args:
            beta (float, optional): Sigma along the column axis (channel).
                Defaults to 20.
            beta_alpha_factor (float, optional): 1 channel equals
                beta_alpha_factor seconds. Defaults to 25.
            alpha (float, optional): Sigma along the row axis (time).
                Defaults to None.

        Raises:
            ValueError: The function accepts two and only two out of three \
                (alpha, beta, beta_alpha_factor)
        """
        # Check number of argument
        #######################################################################
        if (alpha is None) + (beta is None) + (beta_alpha_factor is None) != 1:
            raise ValueError("The function accepts two and only two out of \
                             three (alpha, beta, beta_alpha_factor)")
        frequency = len(self.signal) / self.duration
        if alpha is None:
            alpha = beta / beta_alpha_factor * frequency
        if beta is None:
            beta = alpha * beta_alpha_factor / frequency
        gauss_df = gaussian_filter(np.abs(self.signal), sigma=(alpha, beta))
        gauss_df = pd.DataFrame(gauss_df, index=self.signal.index)
        self.signal = gauss_df

    def check_data_type(
        self,
        data: Union[pd.DataFrame, np.ndarray] = None
    ) -> str:
        """Check data type in attribute signal or input data.

        Args:
            data (Union[pd.DataFrame, np.ndarray], optional): Input data.
            If None, attribute signal is assessed. Defaults to None.

        Returns:
            str: Either ['gray', 'binary', 'positive', 'float']
        """
        # if data.isin([0, 1]).all().all():
        # unique_values = pd.Series(data.values.flatten()).unique()
        # if set(unique_values).issubset(set([0, 1])):
        # the above methods are more precise but slow
        if data is None:
            data = self.signal
        data = pd.DataFrame(data)
        if np.min(data) == 0 and np.max(data) == 1:
            data_type = 'binary'
        elif np.min(data) == 0 and np.max(data) == 255:
            data_type = 'gray'
        elif np.min(data) >= 0:
            data_type = 'positive'
        else:
            data_type = 'float'
        return data_type

    @staticmethod
    def find_duration(data: pd.DataFrame) -> int:
        """Infer duration in seconds of a Pandas data frame having index as
            datetime.time.

        Args:
            data (pd.DataFrame): Input data frame.
        Returns:
            int: duration in seconds of the data.
        """
        # Two datetime.time objects
        time0 = data.index[0]
        time1 = data.index[1]
        # Convert the time to datetime.datetime objects with a common date
        common_date = datetime.today().date()
        datetime0 = datetime.combine(common_date, time0)
        datetime1 = datetime.combine(common_date, time1)
        # Calculate the time difference
        time_difference = datetime1 - datetime0
        # Get the time difference in seconds
        duration = time_difference.total_seconds() * len(data)
        return duration

    def heatmap(
        self,
        data: Union[pd.DataFrame, np.ndarray] = None,
        vmin: float = None,
        vmax: float = None,
        time_precision: str = 'seconds',
        time_gap: int = None,
        channel_gap: int = None
    ) -> None:
        """Plot heatmap.

        Args:
            data (Union[pd.DataFrame, np.ndarray], optional): Data to be
                plotted heatmap. If None, attribute signal is used. Defaults to
                None.
            vmin (Union[float, str], optional): Values to anchor the colormap.
                None will take negative 95th percentile. Defaults to None.
            vmax (Union[float, str], optional): Values to anchor the colormap.
                None will take 95th percentile. Defaults to None.
            time_precision (str, optional): Precision of time in y-axis.
                Can be in ['auto', 'hours', 'minutes', 'seconds',
                'milliseconds', 'microseconds']. Defaults to 'seconds'.
            time_gap (int, optional): Gap in seconds between labels in time
                axis. If None, automatically choose. Defaults to None.
            channel_gap (int, optional): Gap in channels between labels in
                channel axis. If None, automatically choose. Defaults to None.
        """
        # Data input
        #######################################################################
        if data is None:
            data = self.signal
        data = pd.DataFrame(data)  # in case input data is np.ndarray

        # Warning if the data is too big
        #######################################################################
        # 50*(10**6) cells takes about 1 second to plot
        relative_data_size = data.count().sum() / (50*(10**6))
        if relative_data_size > 10:
            print(f"""Expect to display in {relative_data_size:.0f} seconds.
                  Consider to sample the data.""")

        # Check number of argument
        #######################################################################
        data_type = self.check_data_type(data)
        if data_type == 'binary':
            cmap = 'gray'
            vmin = 0
            vmax = 1
        elif data_type in ['gray', 'positive']:
            cmap = 'viridis'
            vmin = 0
            if vmax is None:
                percentile = np.quantile(data, 0.95)
                vmax = percentile
        else:
            cmap = 'RdBu'
            if (vmin is None) or (vmax is None):
                percentile = np.quantile(np.abs(self.signal), 0.95)
                vmin = - percentile
                vmax = percentile
                print(f'Heatmap with vmin {vmin:.3g}, vmax {vmax:.3g}')
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vmax=vmax,
            vcenter=(vmin + vmax) / 2
        )
        # if data_type == 'category':
        #     cmap = 'tab10'
        #     norm = None

        # plot heatmap
        #######################################################################
        plt.imshow(
            X=data,
            aspect=data.shape[1] / data.shape[0],  # square
            cmap=cmap,
            norm=norm,
            interpolation='none',  # no interpolation
            # to see the last values of x-axis
            # extent=[0, self.signal.shape[1], 0, self.signal.shape[0]],
            origin='lower'
        )
        # adjust the y-axis to time
        #######################################################################
        try:  # in case input data is np.ndarray, which doesn't have time index
            times = data.index  # values of y-axis
            times = [i.isoformat(timespec=time_precision) for i in times]
            divisors = helper.find_divisors(len(times))
            nlabels = helper.largest_smaller_than_threshold(divisors, 20)
            if time_gap is None:
                time_gap = Dasly.find_duration(data) / nlabels
            sample_gap = time_gap * len(times) / Dasly.find_duration(data)
            sample_gap = int(sample_gap)
            time_positions = np.arange(0, len(times), sample_gap)
            time_labels = times[::sample_gap]
            plt.yticks(time_positions, time_labels)
        except AttributeError:
            pass

        # adjust the x-axis to channel
        #######################################################################
        channels = data.columns  # values of x-axis
        divisors = helper.find_divisors(len(channels))
        nlabels = helper.largest_smaller_than_threshold(divisors, 20)
        if channel_gap is None:
            channel_gap = (channels[-1] - channels[0]) / nlabels
        column_gap = channel_gap * len(channels) / (channels[-1] - channels[0])
        column_gap = int(column_gap)
        channel_positions = np.arange(0, len(channels), column_gap)
        channel_labels = channels[::column_gap]
        plt.xticks(channel_positions, channel_labels, rotation=60)

        # if data_type == 'category':
        #     plt.colorbar()

    def convolve(
        self,
        s1: float = 80,
        s2: float = 85,
        std_space: float = 10
    ):
        """Gaussian filter the data 

        Args:
            s1 (float, optional): _description_. Defaults to 80.
            s2 (float, optional): _description_. Defaults to 85.
            std_space (float, optional): _description_. Defaults to 10.
        """
        cov_mat = helper.cal_cov_mat(s1, s2, std_space)
        gauss_filter = helper.create_gauss_filter(
            cov_mat=cov_mat,
            sampling_rate=self.sampling_rate
        )
        signal_tensor = torch.tensor(
            self.signal.values.copy(),
            dtype=torch.float32
        ).to('cuda')
        filter_tensor = torch.tensor(
            gauss_filter,
            dtype=torch.float32
        ).to('cuda')
        pad_t = np.floor(filter_tensor.shape[0] / 2).astype(int)
        pad_s = np.floor(filter_tensor.shape[1] / 2).astype(int)
        signal = F.conv2d(
            signal_tensor.unsqueeze(0).unsqueeze(0),
            filter_tensor.unsqueeze(0).unsqueeze(0),
            padding=(pad_t, pad_s)
        )
        signal = signal[:, :, 0: self.signal.shape[0], 0: self.signal.shape[1]]
        signal = signal.squeeze(0).squeeze(0)
        signal = signal.detach().cpu().numpy()
        self.signal = pd.DataFrame(signal, index=self.signal.index)

    def fft(self):
        """Fourier transform
        """
        fft_results = np.zeros(self.signal.shape, dtype=complex)
        for column_name, column_data in self.signal.items():
            column_fft = fft.fft(column_data.values)
            fft_results[
                :,
                self.signal.columns.get_loc(column_name)
            ] = column_fft
        # take the average
        average_fft = np.mean(fft_results, axis=1)
        freq = np.linspace(
            0,
            self.sampling_rate / 2,
            round(len(self.signal) / 2) + 1
        )
        # plot
        plt.figure(figsize=(8, 4))
        plt.stem(
            freq,
            abs(average_fft)[0: len(freq)],
            'b',
            markerfmt=" ",
            basefmt="-b"
        )
        plt.xlabel('Frequency F (Hz)')
        plt.ylabel('Amplitude')

    def fft2d(self):
        """2D Fourier transform
        """
        signal_fft = np.fft.fft2(self.signal)  # 2D FFT
        # shift the zero frequency component to the center
        signal_fft = np.fft.fftshift(signal_fft)

        # convert the index to frequencies
        nyq_f = self.sampling_rate / 2  # time nyquist frequency
        nyq_k = (1 / self.channel) / 2  # space nyquist frequency

        f_labels = np.linspace(-nyq_f, nyq_f, signal_fft.shape[0] + 1)
        f_labels = f_labels[:-1]
        k_labels = np.linspace(-nyq_k, nyq_k, signal_fft.shape[1] + 1)
        k_labels = k_labels[:-1]

        self.signal_fft = pd.DataFrame(
            signal_fft,
            index=f_labels,
            columns=k_labels
        )

    @staticmethod
    def fft2d_plot(
        fft_df: pd.DataFrame,
        vmin: Union[float, str] = 'auto',
        vmax: Union[float, str] = 'auto',
    ):
        if (vmin == 'auto') or (vmax == 'auto'):
            vmin = np.quantile(np.abs(fft_df), 0.01)
            vmax = np.quantile(np.abs(fft_df), 0.99)
            print(f'Heatmap with vmin {vmin:.3g}, vmax {vmax:.3g}')
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vmax=vmax,
            vcenter=(vmin + vmax) / 2
        )
        plt.imshow(
            np.abs(fft_df),
            aspect=fft_df.shape[1] / fft_df.shape[0],
            norm=norm,
            origin='lower'
        )
        plt.colorbar()

        # adjust the x-axis to channel
        x = fft_df.columns  # values of y-axis
        nx = len(x)
        no_labels = 11  # how many labels to see on axis x
        x_positions = np.linspace(0, nx, no_labels)  # label position
        x_labels = np.linspace(np.min(x), np.max(x), no_labels)  # labels
        x_labels = [f'{i:.2f}' for i in x_labels]
        plt.xticks(x_positions, x_labels, rotation=60)
        plt.xlabel('Wavenumber k')

        # adjust the y-axis to time
        y = fft_df.index  # values of y-axis
        ny = len(y)
        no_labels = 11  # how many labels to see on axis y
        y_positions = np.linspace(0, ny, no_labels)  # label position
        y_labels = np.linspace(np.min(y), np.max(y), no_labels)  # labels
        y_labels = [int(round(i, 0)) for i in y_labels]
        plt.yticks(y_positions, y_labels)
        plt.ylabel('Frequency f')

    def detect_events(
            self,
            eps: float = 100,
            min_samples: int = 100
    ) -> None:
        """Detect events.

        Args:
            eps (float, optional): _description_. Defaults to 3.
            min_samples (int, optional): _description_. Defaults to 2.
        """
        SPEED_MS = 1  # 25meter ~ 1 second
        # Detect events
        #######################################################################
        points = np.argwhere(self.signal.values == 1)
        points = points * np.array([SPEED_MS / self.sampling_rate, 1])
        cluster = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        cluster = cluster.fit(points)
        # replace the cluster label to the signal data
        self.signal.replace(0, np.nan, inplace=True)
        points = points / np.array([SPEED_MS / self.sampling_rate, 1])
        points = points.astype(int)
        for i in range(len(points)):
            self.signal.iloc[points[i][0], points[i][1]] = cluster.labels_[i]
        events = (
            pd.DataFrame(points, columns=['Time', 'Channel'])
            .assign(Cluster=cluster.labels_)
            .groupby('Cluster')
            .agg({
                'Time': 'min',
                'Channel': ['mean', 'count']})
            .round(0)
            .astype(int)
        )
        events.columns = ['Time', 'Channel', 'Count']
        self.points = points
        self.cluster = cluster
        self.events = events

    def save_events(
            self,
            folder_path: str = '../data/interim/',
            event_time: float = 0.2,
            event_space: float = 50,
    ) -> None:
        """Save events in separated hdf5 files.

        Args:
            folder_path (str, optional): _description_. Defaults to
                '../data/interim/'.
            event_time (float, optional): _description_. Defaults to 4.
            event_space (float, optional): _description_. Defaults to 50.
        """
        cluster = self.events_df.index.to_list()
        if -1 in cluster:
            cluster.remove(-1)
        for i in cluster:
            # Find time and space center
            ###################################################################
            time_center = self.events_df.loc[i]['Time']
            space_center = self.events_df.loc[i]['Channel']
            # Move the region if the center is too close to the border
            ###################################################################
            # if time_center - (event_time/2)*self.frequency < 0:
            #     time_bottom = 0
            #     time_top = event_time * self.frequency
            # elif time_center + (event_time/2)*self.frequency > \
            #         len(self.signal):
            #     time_top = len(self.signal)
            #     time_bottom = len(self.signal) - (event_time * self.frequency)
            # else:
            #     time_bottom = time_center - (event_time/2)*self.frequency
            #     time_top = time_center + (event_time/2)*self.frequency

            if time_center < 0:
                time_bottom = 0
                time_top = event_time * self.sampling_rate
            elif time_center > len(self.signal):
                time_top = len(self.signal)
                time_bottom = len(self.signal) - (event_time * self.sampling_rate)
            else:
                time_bottom = time_center - (1/8) * event_time * self.sampling_rate
                time_top = time_center + (7/8) * event_time * self.sampling_rate

            if space_center - event_space/2 < 0:
                space_left = 0
                space_right = event_space
            elif space_center + event_space/2 > self.signal.shape[1]:
                space_right = self.signal.shape[1]
                space_left = self.signal.shape[1] - event_space
            else:
                space_left = space_center - event_space/2
                space_right = space_center + event_space/2
            # Find region around the center
            ###################################################################
            data_cut = (
                self.signal_decimate.iloc[
                    round(time_bottom): round(time_top),
                    round(space_left): round(space_right)
                ]
            )
            # Save the data frame to file
            ###################################################################
            time_center_name = (
                self.signal.index[time_center]
                .strftime('%H%M%S')
            )
            space_center_name = f'{space_center:03d}'
            cluster_name = f'{i:02d}'
            file_name = f'{folder_path}{time_center_name}_{space_center_name}_{cluster_name}.hdf5'
            data_cut.to_hdf(file_name, key='abc')


if __name__ == "__main__":

    periods = [
        (111545, 111630),  # first drop 111551
        # (103540, 104010),
        # (104730, 105911),
        # (105935, 110455),
    ]
    periods_split = helper.split_periods(periods=periods, time_span=10)

    # Run the flow at every small period
    ###########################################################################
    for period in periods_split:
        das = Dasly()
        das.load_data(
            folder_path='../data/raw/Campus_test_20230628_2kHz/',
            start=period[0],
            end=period[1]
        )
        das.high_pass_filter()
        das.low_pass_filter()
        das.decimate()
        das.signal = das.signal.iloc[200:, 50:]
        das.signal_decimate = das.signal_decimate.iloc[200:, 50:]
        # das.gauss_filter()
        das.detect_events(
            plot=False,
            threshold=5e-8,
            eps=50,
            min_samples=50
        )
        das.save_events()
