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
        self.events: np.ndarray = None
        self.events_df: pd.DataFrame = None
        self.clustering: sklearn.cluster._dbscan.DBSCAN = None
        self.frequency: int = None
        self.duration: int = None

    @staticmethod
    def infer_start_end(
        start: str | datetime = None,
        duration: int = None,
        end: str | datetime = None
    ) -> tuple[datetime, int, datetime]:
        """Infer start if duration and end are provided. Infer duration if
        start and end are provided. Infer end if start and duration are
        provideds.

        Args:
            start (str | datetime, optional): Start time. If string, must be in
                format YYMMDD HHMMSS. Defaults to None.
            duration (int, optional): Duration of the time in seconds. Defaults
                to None.
            end (str | datetime, optional): End time. If string, must be in
                format YYMMDD HHMMSS. Defaults to None.

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
            start = datetime.strptime(start, '%Y%m%d %H%M%S')
        if isinstance(end, str):
            end = datetime.strptime(end, '%Y%m%d %H%M%S')
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
        start, duration, end = Dasly.infer_start_end(
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
        start, duration, end = Dasly.infer_start_end(start, duration, end)
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
            integrate=True,  # default
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
        self.duration = duration
        self.frequency = len(self.signal) / duration

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
        self.signal = signal_sample
        self.frequency = len(self.signal) / self.duration

    def heatmap(
            self,
            binary: bool = False,
            vmin: float = -1e-7,
            vmax: float = 1e-7,
            threshold: float = 4e-8
    ) -> None:
        """Plot heatmap. There are two options:
        1. Plot signal data frame with as-is float values.
        2. Plot signal data frame with transformed binary values. Must provide
            an accompanying threshold.

        Args:
            binary (bool, optional): _description_. Defaults to False.
            vmin (float, optional): _description_. Defaults to -1e-7.
            vmax (float, optional): _description_. Defaults to 1e-7.
            threshold (float, optional): _description_. Defaults to 4e-8.
        """
        if binary:
            ax = sns.heatmap(
                self.signal.iloc[::-1] >= threshold,
                cmap='RdBu',
                center=0
            )
        else:
            ax = sns.heatmap(
                self.signal.iloc[::-1],
                cmap='RdBu',
                center=0,
                vmin=vmin,
                vmax=vmax
            )
        self.ax = ax

    def gauss_filter(self, sigma: float = 10) -> None:
        """Apply 2d Gaussian filter.

        Args:
            sigma (float, optional): Standard deviation of the 2d Gaussian.
                Defaults to 10.
        """
        gauss_df = gaussian_filter(np.abs(self.signal), sigma=sigma)
        gauss_df = pd.DataFrame(gauss_df, index=self.signal.index)
        self.signal = gauss_df

    def detect_events(
            self,
            threshold: float = 4e-8,
            eps: float = 3,
            min_samples: int = 2
    ) -> None:
        """Detect events.

        Args:
            threshold (float, optional): _description_. Defaults to None.
            eps (float, optional): _description_. Defaults to 3.
            min_samples (int, optional): _description_. Defaults to 2.

        Raises:
            ValueError: Must set threshold if not yet plot heatmap binary.
        """
        # Detect events
        #######################################################################
        events = np.argwhere(self.signal.values >= threshold)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(events)
        events_df = (
            pd.DataFrame(events, columns=['Time', 'Channel'])
            .assign(Cluster=clustering.labels_)
            .groupby('Cluster')
            .agg({
                'Time': 'mean',
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
        fig_size = plt.rcParams.get('figure.figsize')
        plt.figure(figsize=(fig_size[0] * 0.8, fig_size[1]))
        ax = sns.scatterplot(
            x=events[:, 1],  # xaxis of events - space
            y=events[:, 0],  # yaxis of events - time
            s=0.5,
            hue=clustering.labels_,
            palette='tab10',
            legend='auto'
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
        for i in self.events_df.index:
            # Find time and space center
            ###################################################################
            time_center = self.events_df.iloc[i].to_list()[0]
            space_center = self.events_df.iloc[i].to_list()[1]
            # Move the region if the center is too close to the border
            ###################################################################
            if time_center - (event_time/2)*self.frequency < 0:
                time_bottom = 0
                time_top = event_time * self.frequency
            elif time_center + (event_time/2)*self.frequency > \
                    len(self.signal):
                time_top = len(self.signal)
                time_bottom = len(self.signal) - (event_time * self.frequency)
            else:
                time_bottom = time_center - (event_time/2)*self.frequency
                time_top = time_center + (event_time/2)*self.frequency

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
                self.signal.iloc[
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
            file_name = f'{folder_path}{time_center_name}_\
                {space_center_name}.hdf5'
            data_cut.to_hdf(file_name, key='abc')


if __name__ == "__main__":

    periods: list[tuple[int | str, int | str]] = [
        # (103030, 103055),
        # (103540, 104010),
        # (104730, 105911),
        # (105935, 110455),
    ]
    for period in periods:
        das = Dasly()
        das.load_data(
            folder_path='../data/raw/Campus_test_20230628_2kHz/',
            start=f'20230628 {period[0]}',
            end=f'20230628 {period[1]}'
        )
        das.high_pass_filter()
        das.low_pass_filter()
        das.decimate()
        das.gauss_filter()
        das.detect_events()
        das.save_events()
