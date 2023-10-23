"""Provides end-to-end flow to load, analyze and visualize DAS data.
"""
__author__ = 'khanhtruong'
__date__ = '2022-06-16'

from typing import Union
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
import sklearn
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import colors
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
        self.scale_factor: float = 1e-10

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
                (start, end, duration)

        Returns:
            tuple[datetime, int, datetime]: start, duration, end
        """
        # Check number of argument
        #######################################################################
        if (start is None) + (end is None) + (duration is None) != 1:
            raise ValueError("The function accepts two and only two out of \
                             three (start, end, duration)")
        # Transform to datetime type if string is inputted
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
        duration: int = None,
        end: Union[str, datetime] = None,
    ) -> list[str]:
        """Get file paths given the folder and time constraints.

        Args:
            folder_path (str): Experiment folder. Must inlcude date folders in
                right next level.
            start (Union[str, datetime], optional): Start time. If string, must
                be in format YYMMDD HHMMSS. Defaults to None.
            duration (int, optional): Duration of the time in seconds. Defaults
                to None.
            end (Union[str, datetime], optional): End time. If string, must be
                in format YYMMDD HHMMSS. Defaults to None.

        Returns:
            list[str]: File paths.
        """
        # Check and infer start, duration, end
        #######################################################################
        start, duration, end = Dasly.infer_time(
            start=start,
            duration=duration,
            end=end
        )
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
        first_file = file_paths[0].split("/")[-1].split(".")[0]
        last_file = file_paths[-1].split("/")[-1].split(".")[0]
        print(f'{len(file_paths)} files, from {first_file} to {last_file}')
        return file_paths

    def load_data(
            self,
            folder_path: str,
            start: Union[str, datetime] = None,
            duration: int = None,
            end: Union[str, datetime] = None,
    ) -> None:
        """Load data to the instance. New attribute:
        - signal_raw: unmutable data set
        - signal: mutable date set, which will be transformed if later methods
            are used

        Args:
            folder_path (str): Experiment folder. Must inlcude date folders in
                right next level.
            start (Union[str, datetime], optional): Start time. If string, must
                be in format YYMMDD HHMMSS. Defaults to None.
            duration (int, optional): Duration of the time in seconds. Defaults
                to None.
            end (Union[str, datetime], optional): End time. If string, must be
                in format YYMMDD HHMMSS. Defaults to None.
        """
        start, duration, end = Dasly.infer_time(
            start=start,
            duration=duration,
            end=end
        )
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
        signal = signal.div(self.scale_factor)
        # Transform dataframe
        #######################################################################
        signal = signal[start:end]  # extact only the range start-end
        signal = signal.iloc[:-1]  # drop the last record (new second already)
        # if the data is within one day, just keep the time as index
        # because keeping date makes the index unnecessarily long
        if pd.Series(signal.index).dt.date.nunique() == 1:
            signal.index = pd.to_datetime(signal.index).time
        self.signal_raw = signal
        self.signal = signal
        self.data_type = 'float'
        self.duration = duration
        self.sampling_rate = len(self.signal) / duration

    def high_pass_filter(
            self,
            cutoff_freq: int = 1,
            nyq_freq: float = 0.5/(1/2000),
            order: int = 4,
            axis: int = 0
    ) -> None:
        """Apply high pass filter.

        Args:
            cutoff_freq (int, optional): _description_. Defaults to 1.
            nyq_freq (float, optional): _description_. Defaults to
                0.5/(1/2000).
            order (int, optional): _description_. Defaults to 4.
            axis (int, optional): _description_. Defaults to 0.
        """
        sos = signal.butter(
            order,
            cutoff_freq/nyq_freq,
            btype='highpass',
            output='sos'
        )
        signal_lowcut = signal.sosfilt(sos, self.signal, axis=axis)
        signal_lowcut = pd.DataFrame(signal_lowcut, index=self.signal.index)
        self.signal = signal_lowcut

    def low_pass_filter(
            self,
            cutoff_freq: int = 200,
            nyq_freq: float = 0.5/(1/2000),
            order: int = 4,
            axis: int = -1
    ) -> None:
        """Apply low pass filter.

        Args:
            cutoff_freq (int, optional): _description_. Defaults to 200.
            nyq_freq (float, optional): _description_. Defaults to
                0.5/(1/2000).
            order (int, optional): _description_. Defaults to 4.
            axis (int, optional): _description_. Defaults to -1.
        """
        sos = signal.butter(
            order,
            cutoff_freq/nyq_freq,
            btype='lowpass',
            output='sos'
        )
        signal_highcut = signal.sosfilt(sos, self.signal, axis=axis)
        signal_highcut = pd.DataFrame(signal_highcut, index=self.signal.index)
        self.signal = signal_highcut

    def decimate(
            self,
            dt: float = 1/2000,
            dt_out: float = 1/2000*16,
            axis=0,
            padtype='line'
    ) -> None:
        """Resample a seismic trace to coarser time sampling interval. Resample
        to a coarser grid. dt time interval can be in any unit, but dt_in and
        dt_out must be in the same unit.

        Args:
            dt (float, optional): _description_. Defaults to 1/2000.
            dt_out (float, optional): _description_. Defaults to 1/2000*16.
            axis (int, optional): _description_. Defaults to 0.
            padtype (str, optional): _description_. Defaults to 'line'.
        """
        signal_sample = signal.resample_poly(
            self.signal,
            int(dt*1e6),
            int(dt_out*1e6),
            axis=axis,
            padtype=padtype
        )
        idx = slice(0, len(self.signal.index), int(dt_out/dt))
        idx = self.signal.index[idx]
        signal_sample = pd.DataFrame(signal_sample, index=idx)
        self.signal_decimate = signal_sample
        self.signal = signal_sample
        self.sampling_rate = len(self.signal) / self.duration

    def get_sampling_rate(self) -> float:
        sampling_rate = len(self.signal) / self.duration
        return sampling_rate

    def sample(self, group_factor: int) -> None:
        """Take sampling according to time (rows).

        Args:
            group_factor (int): Factor would like to take sampling. Ex: value 5
                will take average of every 5 rows
        """
        # create antificial groups
        groups = np.arange(len(self.signal)) // group_factor
        # calculate means by group
        signal = self.signal.groupby(groups).mean()
        # take correct index
        signal.index = self.signal.iloc[::group_factor, :].index
        self.signal = signal
        self.sampling_rate = self.get_sampling_rate()

    def binary_filter(
            self,
            threshold: Union[float, str] = 'auto',
            inplace: bool = True
    ) -> None:
        """Transform data to binary. First take absolute value. Then map to 1
            if greater than or equal to threshold, 0 otherwise.

        Args:
            threshold (Union[float, str], optional): If 'auto', take 95th
                percentile. Defaults to 'auto'.
            inplace (bool, optional): If overwite self.signal. Defaults to
                True.
        """
        if threshold == 'auto':
            threshold = np.quantile(np.abs(self.signal), 0.95)
        print(f'Threshold {threshold:.3g}')
        signal_binary = (np.abs(self.signal) >= threshold).astype(int)
        if inplace:
            self.signal = signal_binary
        return signal_binary

    def grey_filter(self, inplace: bool = True) -> None:
        """Transform data to greyscale 0 to 255 using min-max scalling.

        Args:
            inplace (bool, optional): If overwite self.signal. Defaults to
                True.
        """
        # min-max normalization
        signal_grey = np.abs(self.signal)
        signal_grey = ((signal_grey - np.min(signal_grey)) /
                       (np.max(signal_grey) - np.min(signal_grey)) * 255)
        signal_grey = signal_grey.round(0).astype(np.uint8)
        if inplace:
            self.signal = signal_grey
        return signal_grey

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

    def reset(self) -> None:
        """Reset all transformations on signal
        """
        self.signal = self.signal_raw

    def check_data_type(self) -> str:
        """Check data type

        Returns:
            str: Either ['float', 'greyscale', 'binary']
        """
        # if self.signal.isin([0, 1]).all().all():  # more correct but slow
        if np.min(self.signal) == 0 and np.max(self.signal) == 1:
            data_type = 'binary'
        elif np.min(self.signal) == 0 and np.max(self.signal) == 255:
            data_type = 'grey'
        elif np.min(self.signal) == -1 or np.min(self.signal) == 0:
            data_type = 'category'
        elif np.min(self.signal) >= 0:
            data_type = 'positive'
        else:
            data_type = 'float'
        return data_type

    def heatmap(
            self,
            vmin: Union[float, str] = 'auto',
            vmax: Union[float, str] = 'auto',
            time_precision: str = 'seconds'
    ) -> None:
        """Plot heatmap.

        Args:
            vmin (Union[float, str], optional): Values to anchor the colormap.
                'auto' will take negative 95th percentile. Defaults to 'auto'.
            vmax (Union[float, str], optional): Values to anchor the colormap.
                'auto' will take 95th percentile. Defaults to 'auto'.
            time_precision (str, optional): Precision of time in y-axis.
                Can be in ['auto', 'hours', 'minutes', 'seconds',
                'milliseconds', 'microseconds']. Defaults to 'seconds'.
        """
        # 50*(10**6) cells takes about 1 second to plot
        relative_data_size = self.signal.count().sum() / (50*(10**6))
        if relative_data_size > 10:
            print(f"""Expect to display in {relative_data_size:.0f} seconds.
                  Consider to sample the data.""")
        data_type = self.check_data_type()
        if data_type == 'binary':
            cmap = 'gray'
            if (vmin == 'auto') or (vmax == 'auto'):
                vmin = 0
                vmax = 1
        elif data_type in ['grey', 'positive']:
            cmap = 'viridis'
            if (vmin == 'auto') or (vmax == 'auto'):
                percentile = np.quantile(np.abs(self.signal), 0.95)
                vmin = 0
                vmax = percentile
                print(f'Heatmap with vmin {vmin:.3g}, vmax {vmax:.3g}')
        else:
            cmap = 'RdBu'
            if (vmin == 'auto') or (vmax == 'auto'):
                percentile = np.quantile(np.abs(self.signal), 0.95)
                vmin = - percentile
                vmax = percentile
                print(f'Heatmap with vmin {vmin:.3g}, vmax {vmax:.3g}')
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vmax=vmax,
            vcenter=(vmin + vmax) / 2
        )
        if data_type == 'category':
            cmap = 'tab10'
            norm = None
        plt.imshow(
            X=self.signal,
            aspect=self.signal.shape[1] / self.signal.shape[0],  # square
            cmap=cmap,
            norm=norm,
            interpolation='none',  # no interpolation
            # to see the last values of x-axis
            extent=[0, self.signal.shape[1], 0, self.signal.shape[0]],
            origin='lower'
        )
        # adjust the y-axis to time
        y = self.signal.index  # values of y-axis
        y = [i.isoformat(timespec=time_precision) for i in y]
        ny = len(y)
        no_labels = 15  # how many labels to see on axis y
        step_y = int(ny / (no_labels - 1))  # step between consecutive labels
        y_positions = np.arange(0, ny, step_y)  # pixel count at label position
        y_labels = y[::step_y]  # labels you want
        plt.yticks(y_positions, y_labels)
        if data_type == 'category':
            plt.colorbar()

    def convolve(
        self,
        s1: float = 80,
        s2: float = 85,
        std_space: float = 10
    ):
        cov_mat = helper.cal_cov_mat(s1, s2, std_space)
        gauss_filter = helper.create_gauss_filter(
            cov_mat=cov_mat,
            sampling_rate=self.sampling_rate
        )
        signal_tensor = torch.tensor(
            self.signal.values,
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
        SPEED_MS = 25  # 25meter ~ 1 second
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
