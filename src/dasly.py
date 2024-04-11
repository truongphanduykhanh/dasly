"""Provides end-to-end flow to load, analyze and visualize DAS data.
"""
__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-01-02'

from typing import Union
from datetime import datetime, timedelta
import warnings
import math

import numpy as np
import pandas as pd
import scipy
from scipy import fft
from scipy.signal import butter, sosfilt, decimate
from scipy.ndimage import convolve
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import cv2

from src import simpleDASreader, helper


sns.set_theme()
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', 'is_categorical_dtype')
# seaborn v0.13.0 has been released with these updates
# https://github.com/mwaskom/seaborn/issues/3462


class Dasly:

    def __init__(self) -> None:
        print('Welcome to Dasly!')
        # list of attributes
        self.file_paths: list[str] = None
        self.signal_raw: pd.DataFrame = None
        self.signal: pd.DataFrame = None
        self.sampling_rate: int = None
        self.sampling_rate_channel: int = None
        self.duration: int = None
        self.lines: pd.DataFrame = None
        self.channel = 1

    def update_sampling_rate(self) -> None:
        """Update sampling rate of the data.
        """
        time_0 = self.signal.index[0]
        time_1 = self.signal.index[1]
        # Create datetime objects with a common date
        common_date = datetime.today()
        datetime0 = datetime.combine(common_date, time_0)
        datetime1 = datetime.combine(common_date, time_1)
        # Calculate the time difference
        time_difference = datetime1 - datetime0
        time_difference = time_difference.total_seconds()
        time_difference = np.abs(time_difference)
        sampling_rate = 1 / time_difference
        self.sampling_rate = sampling_rate

    def reset(self) -> None:
        """Reset all attributes and transformations on signal.
        """
        self.signal = self.signal_raw
        self.lines = None
        self.update_sampling_rate()

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
        folder_path: str = None,
        file_paths: list[str] = None,
        start: Union[str, datetime] = None,
        duration: int = None,
        end: Union[str, datetime] = None,
        format: str = '%Y%m%d %H%M%S'
    ) -> None:
        """Load data to the instance.

        Args:
            folder_path (str): Experiment folder. Must inlcude date folders in
                right next level. Defaults to None.
            file_paths (str): File paths. If folder_path and file_paths are
                inputted, prioritize to use file_paths. Defaults to None.
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
        if file_paths is None:
            # Infer time
            ###################################################################
            start, duration, end = Dasly.infer_time(
                start=start,
                duration=duration,
                end=end
            )
            # Get files paths
            ###################################################################
            file_paths = Dasly.get_file_paths(
                folder_path=folder_path,
                start=start,
                duration=duration
            )
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
        # signal = signal.loc[:, 50000: 100000-1]
        # Transform dataframe
        #######################################################################
        if file_paths is None:
            signal = signal[start:end]  # extact only the range start-end
            signal = signal.iloc[:-1]  # drop the last record (new second)
        self.start = np.min(signal.index)
        self.end = np.max(signal.index)
        # if the data is within one day, just keep the time as index
        # because keeping date makes the index unnecessarily long
        if pd.Series(signal.index).dt.date.nunique() == 1:
            signal.index = pd.to_datetime(signal.index).time
        self.signal_raw = signal  # immutable attribute
        self.signal = signal  # mutable attribute, that can be changed later
        self.update_sampling_rate()
        self.duration = len(signal) * (1 / self.sampling_rate)
        self.file_paths = file_paths

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
            index=self.signal.index,
            columns=self.signal.columns
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
        signal_lowpass = pd.DataFrame(
            signal_lowpass,
            index=self.signal.index,
            columns=self.signal.columns
        )
        # return
        #######################################################################
        if inplace:
            self.signal = signal_lowpass
        else:
            return signal_lowpass

    def sample(
        self,
        seconds: int,
        channels: int,
        func_name: str = 'mean',
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
        factor_rows = seconds * self.sampling_rate
        group_rows = np.arange(len(self.signal)) // factor_rows

        channels_gap = self.signal.columns[1] - self.signal.columns[0]
        factor_columns = channels / channels_gap
        group_columns = np.arange(len(self.signal.columns)) // factor_columns

        signal_sample = self.signal.copy()
        signal_sample.index = group_rows
        signal_sample.columns = group_columns

        # group by rows and columns
        #######################################################################
        signal_sample = (
            signal_sample
            .stack()
            .groupby(level=[0, 1])
            .agg(func_name)
            .unstack()
        )
        # update indices and columns
        #######################################################################
        signal_sample.index = self.signal.index[::int(factor_rows)]
        signal_sample.columns = self.signal.columns[::int(factor_columns)]

        # return
        #######################################################################
        if inplace:
            self.signal = signal_sample
            self.update_sampling_rate()
        else:
            return signal_sample

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
        signal_decimate = pd.DataFrame(
            signal_decimate,
            index=idx,
            columns=self.signal.columns
        )
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
        quantile: float = 0.90,
        threshold: float = None,
        by_column: bool = False,
        inplace: bool = True,
    ) -> Union[None, pd.DataFrame]:
        """Transform data to binary. First take absolute value. Then map to 1
            if greater than or equal to threshold, 0 otherwise.

        Args:
            quantile (float, optional): Quantile as a threshold. Defaults to
                0.90.
            threshold (float, optional): Threshold value, apply one threshold
                to all data frame. Defaults to None.
            by_column (bool, optional): get binary by applying different
                thresholds for every column. Defaults to False.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # check arguments
        #######################################################################
        if threshold is None:
            threshold = np.quantile(np.abs(self.signal), quantile)
            if not by_column:
                print(f'threshold: {threshold}')
        # by each columns
        #######################################################################
        if by_column:
            mean = np.mean(np.abs(self.signal), axis=0)
            std = np.std(np.abs(self.signal), axis=0)
            threshold = mean + scipy.stats.norm.ppf(quantile) * std
        signal_binary = (np.abs(self.signal) >= threshold).astype(np.uint8)
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

    def gray_filter_cv2(
        self,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Transform data to grayscale 0 to 255 using cv2 scalling.

        Args:
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # cv2 methods
        signal_scaled = self.signal / np.quantile(self.signal, 0.5)
        signal_gray = cv2.convertScaleAbs(signal_scaled)
        # return
        #######################################################################
        if inplace:
            self.signal = signal_gray
        else:
            return signal_gray

    def gauss_filter(
        self,
        s1: float = 85,
        s2: float = 90,
        std_space: float = 10,
        cov_mat: np.ndarray = None,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Gaussian filter the data. There are 2 ways of input infomration of
            the filter:
            1. Input covariance matrix cov_mat
            2. Input (s1, s2, std_space). From that, the program will infer the
            covariance matrix. This is particular useful for straight line
            shape alike application.

        Args:
            s1 (float, optional): Lower speed limit. Defaults to 85.
            s2 (float, optional): Upper speed limit. Defaults to 90.
            std_space (float, optional): Standard deviation along space axis.
                Defaults to 10.
            cov_mat (np.ndarray, optional):  The covariance matrix has the form
                of [[a, b], [b, c]], in which a is the variance along the space
                axis (not time!), c is the variance along the time axis, b is
                the covariance between space and time. The unit of time is
                second, the unit of space is channel. Defaults to None.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.

        """
        # calculate covariance matrix if not inputted
        #######################################################################
        if cov_mat is None:
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
        signal_gaussian = F.conv2d(
            signal_tensor.unsqueeze(0).unsqueeze(0),
            filter_tensor.unsqueeze(0).unsqueeze(0),
            padding=(pad_t, pad_s)
        )
        signal_gaussian = signal_gaussian[
            :, :, 0: self.signal.shape[0], 0: self.signal.shape[1]]
        signal_gaussian = signal_gaussian.squeeze(0).squeeze(0)
        signal_gaussian = signal_gaussian.detach().cpu().numpy()
        signal_gaussian = pd.DataFrame(
            signal_gaussian,
            index=self.signal.index,
            columns=self.signal.columns
        )
        # return
        #######################################################################
        if inplace:
            self.signal = signal_gaussian
        else:
            return signal_gaussian

    def sobel_filter(self, inplace: bool = True) -> Union[None, pd.DataFrame]:
        """Apply Sobel operator to detect edges.

        Args:
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # Define the Sobel operator kernels
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply Sobel operator in the x and y directions
        sobel_x = convolve(self.signal.values, sobel_kernel_x)
        sobel_y = convolve(self.signal.values, sobel_kernel_y)

        # Filter only possitive gradients
        sobel_x = np.maximum(sobel_x, 0)
        sobel_y = np.maximum(sobel_y, 0)

        # Calculate the magnitude of the gradient
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        signal_sobel = pd.DataFrame(
            gradient_magnitude,
            index=self.signal.index,
            columns=self.signal.columns
        )

        # return
        #######################################################################
        if inplace:
            self.signal = signal_sobel
        else:
            return signal_sobel

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
            print(f'vmax: {vmax:.3g}')
        else:
            cmap = 'RdBu'
            if (vmin is None) or (vmax is None):
                percentile = np.quantile(np.abs(data), 0.95)
                vmin = - percentile
                vmax = percentile
                print(f'vmin: {vmin:.3g}, vmax: {vmax:.3g}')
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vmax=vmax,
            vcenter=(vmin + vmax) / 2
        )

        # plot heatmap
        #######################################################################
        plt.imshow(
            X=data,
            aspect=data.shape[1] / data.shape[0],  # square
            cmap=cmap,
            norm=norm,
            interpolation='none',  # no interpolation
            # to see the last values of x-axis
            extent=[0, data.shape[1], 0, data.shape[0]],
            origin='lower'
        )
        if self.lines is not None:
            for line in self.lines.iloc[:, 0:4].values:  # loop over the lines
                x1, y1, x2, y2 = line
                plt.plot([x1, x2], [y1, y2])

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

        plt.colorbar()

    def hough_transform(self, time=30) -> None:
        """Apply Hough transform to detect lines in the data.
        """
        # angle resolution (theta)
        #######################################################################
        average_speed = 85  # in km/h
        speed_resolution = 0.2
        # angle in radian
        angle1 = math.atan(self.sampling_rate / (average_speed / 3.6))
        angle2 = math.atan(
            self.sampling_rate / ((average_speed + speed_resolution) / 3.6))
        angle_resolution = np.abs(angle1 - angle2)  # radian resolution needed

        # length of vehicle (lines) want to predict
        #######################################################################
        # mimimum time in seconds a vehicle needs to be on the bridge to be
        # detected. The higher this value is, the more accurate the detection,
        # but the larger the delay of detection (need to wait longer)
        # time = 30  # in second
        distance = average_speed / 3.6 * time  # in meter
        length = np.sqrt((time * self.sampling_rate) ** 2 + distance ** 2)

        lines = cv2.HoughLinesP(
            self.signal.values,
            rho=1,  # distance resolution
            # angle resolution in radian need to have speed resolution ~0.1k/m
            theta=angle_resolution,
            threshold=int(0.5 * length),  # needs to covert 50% of the length
            minLineLength=0.8 * length,  # needs to at least 80% of the length
            maxLineGap=0.2 * length  # must not interupt > 20% of the length
        )
        if lines is not None:
            print(f'{len(lines)} lines are detected')
            self.lines = np.squeeze(lines, axis=1)
            self.__line_df()

    def __line_df(self) -> None:
        """Infer properties from the detected lines such as speed, time, ...
        """
        # Calculate additional values for each line
        lines_with_info = []
        for line in self.lines:
            x1, y1, x2, y2 = line

            # length of the line
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # angle of the line with the horizontal (note where is the origin)
            angle = np.arctan2(y2 - y1, x2 - x1)
            with warnings.catch_warnings():  # ignore warning if y2 - y1 == 0
                warnings.filterwarnings(
                    'ignore',
                    message='divide by zero encountered in scalar divide')
                speed = ((x2 - x1) / (y2 - y1)) * self.sampling_rate * 3.6

            # distance from the origin to the line (top right)
            # distance = np.abs((x2 - x1) * (y1 - 0) - (y2 - y1) * (x1 - 0)) /
            # np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # ignore warning if angle=+-pi/2, so np.tan(angle) gets infinitive
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='invalid value encountered in cast')
                # intersection with left vertical boundary
                left_intersection = np.int32(y1 + (0 - x1) * np.tan(angle))
                left_intersection = self.start + timedelta(
                    seconds=left_intersection / self.sampling_rate)

                # intersection with middle vertical
                middle_intersection = np.int32(
                    y1 + (int(self.signal.shape[1]) / 2 - x1) * np.tan(angle))
                middle_intersection = self.start + timedelta(
                    seconds=middle_intersection / self.sampling_rate)

                # intersection with right vertical boundary
                right_intersection = np.int32(
                    y1 + (int(self.signal.shape[1]) - x1) * np.tan(angle))
                right_intersection = self.start + timedelta(
                    seconds=right_intersection / self.sampling_rate)

            # Append the additional values to the line
            lines_with_info.append(np.array([
                x1, y1, x2, y2,
                length, angle, speed,
                left_intersection,
                middle_intersection,
                right_intersection
            ]))

        # concadinate to a data frame
        #######################################################################
        lines_df = (
            pd.DataFrame(
                lines_with_info,
                columns=[
                    'x1', 'y1', 'x2', 'y2',
                    'length', 'angle', 'speed',
                    'left', 'middle', 'right']
            )
            .sort_values('middle')
            .reset_index(drop=True)
            )

        self.lines = lines_df

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


if __name__ == "__main__":

    from tqdm import tqdm

    start = '20231005 082445'
    end = '20231005 110545'

    starts = helper.generate_list_time(start, end, 10)

    detect_lines = pd.DataFrame()

    for i, start in enumerate(tqdm(starts)):
        das = Dasly()
        das.load_data(
            folder_path=(
                '/media/kptruong/yellow02/Aastfjordbrua/Aastfjordbrua/'),
            start=start,
            duration=60
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
            detect_lines = pd.concat([
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
            detect_lines = pd.concat([
                detect_lines,
                das.lines.assign(batch=das.start)
            ])

    # export to csv file
    ###########################################################################
    start = '_'.join(start.split())
    end = '_'.join(end.split())
    detect_lines.to_csv(f'lines_{start}_{end}_3.csv', index=False)

    # from tqdm import tqdm

    # start = '20231005 082445'
    # end = '20231005 110545'

    # starts = helper.generate_list_time(start, end, 10)

    # detect_lines = pd.DataFrame()

    # for i, start in enumerate(tqdm(starts)):
    #     das = Dasly()
    #     das.load_data(
    #         folder_path=(
    #             '/media/kptruong/yellow02/Aastfjordbrua/Aastfjordbrua/'),
    #         start=start,
    #         duration=60
    #     )
    #     # forward filter
    #     #######################################################################
    #     das.lowpass_filter(0.5)
    #     das.decimate(sampling_rate=6)
    #     das.gauss_filter(85, 90)
    #     das.sobel_filter()

    #     # background noise
    #     try:  # calculate weighted mean and weighted std
    #         step_mean = np.mean(np.abs(das.signal), axis=0)
    #         step_std = np.std(np.abs(das.signal), axis=0)
    #         mean = step_mean * (1 / (i + 1)) + mean * (i / (i + 1))
    #         std = step_std * (1 / (i + 1)) + std * (i / (i + 1))
    #     except NameError:  # for the first iteration
    #         mean = np.mean(np.abs(das.signal), axis=0)
    #         std = np.std(np.abs(das.signal), axis=0)

    #     threshold = mean + scipy.stats.norm.ppf(0.90) * std
    #     binany_forward = (np.abs(das.signal) >= threshold).astype(np.uint8)
    #     das.signal = binany_forward

    #     das.hough_transform()
    #     if das.lines is not None:
    #         das.lines = das.lines.loc[lambda df: df['speed'] > 0]
    #         detect_lines = pd.concat([
    #             detect_lines,
    #             das.lines.assign(batch=das.start)
    #         ])

    #     # backward filter
    #     #######################################################################
    #     das.reset()
    #     das.lowpass_filter(0.5)
    #     das.decimate(sampling_rate=6)
    #     das.gauss_filter(-90, -85)
    #     das.sobel_filter()
    #     binany_backward = (np.abs(das.signal) >= threshold).astype(np.uint8)
    #     das.signal = binany_backward

    #     das.hough_transform()
    #     if das.lines is not None:
    #         das.lines = das.lines.loc[lambda df: df['speed'] < 0]
    #         detect_lines = pd.concat([
    #             detect_lines,
    #             das.lines.assign(batch=das.start)
    #         ])

    # # export to csv file
    # ###########################################################################
    # start = '_'.join(start.split())
    # end = '_'.join(end.split())
    # detect_lines.to_csv(f'lines_{start}_{end}_2.csv', index=False)
