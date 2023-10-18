"""Provides end-to-end flow to load, analyze and visualize DAS data.
"""
__author__ = 'khanhtruong'
__date__ = '2022-06-16'


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

from src import simpleDASreader

sns.set()
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


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
        self.clustering: sklearn.cluster._dbscan.DBSCAN = None
        self.sampling_rate: int = None
        self.duration: int = None
        self.data_type: str = None

    @staticmethod
    def infer_time(
        start: str | datetime = None,
        duration: int = None,
        end: str | datetime = None,
        format: str = '%Y%m%d %H%M%S'
    ) -> tuple[datetime, int, datetime]:
        """Infer start if duration and end are provided. Infer duration if
        start and end are provided. Infer end if start and duration are
        provideds.

        Args:
            start (str | datetime, optional): Start time. If string, must be in
                format YYMMDD HHMMSS, specify in argument format otherwise.
                Defaults to None.
            duration (int, optional): Duration of the time in seconds. Defaults
                to None.
            end (str | datetime, optional): End time. If string, must be in
                format YYMMDD HHMMSS, specify in argument format otherwise.
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
        start: str | datetime = None,
        duration: int = None,
        end: str | datetime = None,
    ) -> list[str]:
        """Get file paths given the folder and time constraints.

        Args:
            folder_path (str): Experiment folder. Must inlcude date folders in
                right next level.
            start (str | datetime, optional): Start time. If string, must be in
                format YYMMDD HHMMSS. Defaults to None.
            duration (int, optional): Duration of the time in seconds. Defaults
                to None.
            end (str | datetime, optional): End time. If string, must be in
                format YYMMDD HHMMSS. Defaults to None.

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
            start: str | datetime = None,
            duration: int = None,
            end: str | datetime = None,
    ) -> None:
        """Load data to the instance. New attribute:
        - signal_raw: unmutable data set
        - signal: mutable date set, which will be transformed if later methods
            are used

        Args:
            folder_path (str): Experiment folder. Must inlcude date folders in
                right next level.
            start (str | datetime, optional): Start time. If string, must be in
                format YYMMDD HHMMSS. Defaults to None.
            duration (int, optional): Duration of the time in seconds. Defaults
                to None.
            end (str | datetime, optional): End time. If string, must be in
                format YYMMDD HHMMSS. Defaults to None.
        """
        start, duration, end = Dasly.infer_time(start, duration, end)
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
        signal = signal[start:end]  # take extact the range start-end
        signal = signal.iloc[:-1]  # drop the last record (new second already)
        # signal = signal.iloc[::-1]  # reverse order of time, to plot easier
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

    def sample(
            self,
            group_factor: int
    ) -> pd.DataFrame:
        """Take sampling according to time (rows).

        Args:
            group_factor (int): Factor would like to take sampling. Ex: value 5
                will take average of every 5 rows
        """
        # create antificial groups
        groups = np.arange(len(self.signal_raw)) // group_factor
        # calculate means by group
        signal = self.signal_raw.groupby(groups).mean()
        # take correct index
        signal.index = self.signal_raw.iloc[::group_factor, :].index
        self.signal = signal
        return signal

    def binary_filter(
            self,
            threshold: float | str = 'auto',
            inplace: bool = True
    ) -> None:
        """Transform data to binary. First take absolute value. Then map to 1
            if greater than or equal to threshold, 0 otherwise.

        Args:
            threshold (float | str, optional): If 'auto', take 95th percentile.
                Defaults to 'auto'.
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

    def greyscale_filter(self, inplace: bool = True) -> None:
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

    def check_data_type(self) -> str:
        """Check data type

        Returns:
            str: Either ['float', 'greyscale', 'binary']
        """
        if np.min(self.signal == 0) & np.max(self.signal == 255):
            return 'greyscale'
        elif self.signal.isin([0, 1]).all().all():
            return 'binary'
        else:
            return 'float'

    def heatmap(
            self,
            color_type: str = 'auto',
            vmin: float | str = 'auto',
            vmax: float | str = 'auto',
            binary: bool = False,
            threshold: float | str = 'auto',
            greyscale: bool = False,
            time_precision: str = 'seconds'
    ) -> None:
        """Plot heatmap. There are two options:
        1. Plot signal data frame with as-is float values.
        2. Plot signal data frame with transformed binary values. Should
            provide an accompanying threshold, take 95th percentile otherwise.

        Args:
            vmin (float | str, optional): Values to anchor the colormap. 'auto'
                will take negative 95th percentile. Defaults to 'auto'.
            vmax (float | str, optional): Values to anchor the colormap. 'auto'
                will take 95th percentile. Defaults to 'auto'.
            binary (bool, optional): If plot binary. Defaults to False.
            threshold (float | str, optional): Threshold to convert to binary.
                Defaults to 'auto'.
            greyscale (bool, optional): If plot greyscale. Defaults to False.
            time_precision (str, optional): Precision of time in y-axis.
                Can be in ['auto', 'hours', 'minutes', 'seconds',
                'milliseconds', 'microseconds']. Defaults to 'seconds'.
        """
        data_type = self.check_data_type()
        if color_type == 'auto':
            color_type = data_type

        if color_type != data_type:
            warnings.warn(f"""The data is in {data_type}, not valid for
                          {color_type}.""")

        if (color_type == 'binary') & :
            data = self.binary_filter(threshold=threshold, inplace=False)
            cmap = 'gray'
            vmin = 0
            vmax = 1
        elif greyscale:
            data = self.greyscale_filter(inplace=False)
            cmap = 'viridis'
            if (vmin == 'auto') or (vmax == 'auto'):
                percentile = np.quantile(np.abs(data), 0.95)
                vmin = 0
                vmax = percentile
                print(f'Heatmap with vmin {vmin:.3g}, vmax {vmax:.3g}')
        else:
            data = self.signal
            cmap = 'RdBu'
            if (vmin == 'auto') or (vmax == 'auto'):
                percentile = np.quantile(np.abs(data), 0.95)
                vmin = - percentile
                vmax = percentile
                print(f'Heatmap with vmin {vmin:.3g}, vmax {vmax:.3g}')
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vmax=vmax,
            vcenter=(vmin + vmax) / 2
        )
        plt.imshow(
            X=data.iloc[::-1],
            aspect=data.shape[1] / data.shape[0],  # square
            cmap=cmap,
            norm=norm,
            interpolation='none',  # no interpolation
            # to see the last values of x-axis
            extent=[0, data.shape[1], 0, data.shape[0]]
        )
        # adjust the y-axis to time
        y = self.signal.iloc[::-1].index  # values of y-axis
        y = [i.isoformat(timespec=time_precision) for i in y]
        ny = len(y)
        no_labels = 15  # how many labels to see on axis y
        step_y = int(ny / (no_labels - 1))  # step between consecutive labels
        y_positions = np.arange(0, ny, step_y)  # pixel count at label position
        y_labels = y[::step_y]  # labels you want
        plt.yticks(y_positions, y_labels)

    def heatmap_old(
            self,
            vmin: float | str = 'auto',
            vmax: float | str = 'auto',
            binary: bool = False,
            threshold: float | str = 'auto',
            sampling: bool = True
    ) -> None:
        """Plot heatmap. There are two options:
        1. Plot signal data frame with as-is float values.
        2. Plot signal data frame with transformed binary values. Must provide
            an accompanying threshold.

        Args:
            vmin (float | str, optional): _description_. Defaults to 'auto'.
            vmax (float | str, optional): _description_. Defaults to 'auto'.
            binary (bool, optional): _description_. Defaults to False.
            threshold (float | str, optional): _description_. Defaults to 4e-8.
            sampling (bool, optional): _description_. Defaults to True.
        """
        # if the data is too large, sampling the data to plot faster
        # 10**6 cells are plotted roughly in 1 second
        # below code takes sampling if the data is more than 10*(10**6) rows
        # i.e. more than 10 seconds to plot
        relative_data_size = self.signal.count().sum() / (10*(10**6))
        if sampling and (relative_data_size > 1):
            group_factor = int(np.ceil(relative_data_size))
            self.sample(group_factor=group_factor)
            print(f'The data is sampled with factor {group_factor}.')
        if binary:
            if threshold == 'auto':
                threshold = np.quantile(np.abs(self.signal), 0.95)
                print(f'Binary heatmap with threshold {threshold:.3g}')
            ax = sns.heatmap(
                np.abs(self.signal.iloc[::-1]) >= threshold,
                cmap='RdBu',
                center=0
            )
        else:
            if vmin == 'auto':
                percentile = np.quantile(np.abs(self.signal), 0.95)
                vmin = - percentile
                vmax = percentile
                print(f'Heatmap with vmin {vmin:.9g}, vmax {vmax:.9g}')
            ax = sns.heatmap(
                self.signal.iloc[::-1],
                cmap='RdBu',
                center=0,
                vmin=vmin,
                vmax=vmax
            )
        self.ax = ax

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

    def detect_events(
            self,
            threshold: float = 4e-8,
            eps: float = 100,
            min_samples: int = 100,
            plot: bool = True
    ) -> None:
        """Detect events.

        Args:
            threshold (float, optional): _description_. Defaults to None.
            eps (float, optional): _description_. Defaults to 3.
            min_samples (int, optional): _description_. Defaults to 2.
            plot (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: Must set threshold if not yet plot heatmap binary.
        """
        # Detect events
        #######################################################################
        events = np.argwhere(np.abs(self.signal.values) >= threshold)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(events)
        events_df = (
            pd.DataFrame(events, columns=['Time', 'Channel'])
            .assign(Cluster=clustering.labels_)
            .groupby('Cluster')
            .agg({
                'Time': 'min',
                'Channel': ['mean', 'count']})
            .round(0)
            .astype(int)
        )
        events_df.columns = ['Time', 'Channel', 'Count']
        self.events = events
        self.events_df = events_df
        self.clustering = clustering
        # Plot events
        #######################################################################
        if plot:
            fig_size = plt.rcParams.get('figure.figsize')
            plt.figure(figsize=(fig_size[0] * 0.8, fig_size[1]))
            # if max(clustering.labels_) > 15:
            #     legend = False
            # else:
            #     legend = 'auto'
            ax = sns.scatterplot(
                x=events[:, 1],  # xaxis of events - space
                y=events[:, 0],  # yaxis of events - time
                s=0.5,
                hue=clustering.labels_,
                palette='tab10',
                # legend=legend
            )
            ax.set_xticks(self.ax.get_xticks())
            ax.set_xticklabels(self.ax.get_xticklabels(), rotation=90)
            ax.set_yticks(self.ax.get_yticks()[::-1])
            ax.set_yticklabels(self.ax.get_yticklabels())
            plt.xlim(0, self.signal.shape[1])
            plt.ylim(0, self.signal.shape[0])

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


def split_period(
        period: tuple[str | int, str | int],
        time_span: int = 10,
        date: str = '20230628'
) -> list[tuple[datetime, datetime]]:
    """Split a period into many smaller periods.

    Args:
        periods (tuple[str | int, str | int]: (start, end) of period. Format
            '%H%M%S'.
        time_span (int, optional): Period duration in seconds to split.
            Defaults to 10.
        date (str, optional): Date of the data. Format '%Y%m%d'. Defaults to
            '20230628'.

    Returns:
        list[tuple[datetime, datetime]]: List of all smaller periods.
    """
    # Convert start, end to datetime
    ###########################################################################
    start = period[0]
    end = period[1]
    if start > end:
        raise ValueError('start must be smaller or equal to end.')
    start = datetime.strptime(f'{date} {start}', '%Y%m%d %H%M%S')
    end = datetime.strptime(f'{date} {end}', '%Y%m%d %H%M%S')
    # Split the duration into many smaller durations
    ###########################################################################
    duration = (end - start).seconds
    number_split = duration // time_span
    remaining = duration % time_span
    duration_split = [time_span] * number_split + [remaining]
    # Identify the start, end of each duration
    ###########################################################################
    periods = []
    for i in duration_split:
        end = start + timedelta(seconds=i)
        period_i = (start, end)
        periods.append(period_i)
        start = end
    return periods


def split_periods(
        periods: list[tuple[datetime, datetime]],
        time_span: int = 10,
        date: str = '20230628'
) -> list[tuple[datetime, datetime]]:
    """Split periods into many smaller periods.

    Args:
        period (list[tuple[datetime, datetime]]): (start, end) of periods.
            Format '%H%M%S'.
        time_span (int, optional): Period duration in seconds to split.
            Defaults to 10.
        date (str, optional): Date of the data. Format '%Y%m%d'. Defaults to
            '20230628'.

    Returns:
        list[tuple[datetime, datetime]]: List of all smaller periods.
    """
    periods_split = []
    for period in periods:
        periods_i = split_period(
            period=period,
            time_span=time_span,
            date=date
        )
        periods_split.extend(periods_i)
    return periods_split


if __name__ == "__main__":

    periods = [
        (111545, 111630),  # first drop 111551
        # (103540, 104010),
        # (104730, 105911),
        # (105935, 110455),
    ]
    periods_split = split_periods(periods=periods, time_span=10)

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
