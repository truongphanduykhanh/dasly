"""Provides end-to-end flow to load, analyze and visualize DAS data.
"""
__author__ = 'khanhtruong'
__date__ = '2022-06-12'


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src import simpleDASreader


class Dasly:

    def __init__(self) -> None:
        # list of attributes
        self.file_paths: list[str] = None
        self.signal_raw: pd.DataFrame = None
        self.signal: pd.DataFrame = None
        self.threshold: float = None
        self.events: np.ndarray = None

    @staticmethod
    def infer_start_end(
        start: str | datetime = None,
        duration: int = None,
        end: str | datetime = None
    ) -> tuple[datetime, int, datetime]:
        """_summary_

        Args:
            int (_type_): _description_
            datetime (_type_): _description_
            start (_type_, optional): _description_. Defaults to None
            duration: int = None

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if (start is None) + (end is None) + (duration is None) != 1:
            raise ValueError("The function accepts two and only two of (start,\
                            end, duration)")

        if isinstance(start, str):
            start = datetime.strptime(start, '%Y%m%d %H%M%S')
        if isinstance(end, str):
            end = datetime.strptime(end, '%Y%m%d %H%M%S')

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
        """_summary_

        Args:
            folder_path (str): _description_
            start (str | datetime, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            list[str]: _description_
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
        signal = signal.iloc[::-1]  # reverse the order of time, to plot easier
        self.signal_raw = signal
        self.signal = signal

    def lowcut_filter(
            self,
            cutoff_freq: int = 1,
            nyq_freq: float = 0.5/(1/2000),
            order: int = 4,
            axis: int = 0
    ) -> None:
        """
        # Apply lowcut filter to data
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

    def highcut_filter(
            self,
            cutoff_freq: int = 200,
            nyq_freq: float = 0.5/(1/2000),
            order: int = 4,
            axis: int = -1
    ) -> None:
        """
        # Apply highcut filter to data
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
        """
        Resample a seismic trace to coarser time sampling interval
        # Resample to a coarser grid
        # dt time interval can be in any unit,
        #   but dt_in and dt_out must be in the same unit.
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

    def heatmap(
            self,
            binary: bool = False,
            vmin: float = -1e-7,
            vmax: float = 1e-7,
            threshold: float = 4e-8
    ) -> matplotlib.axes._axes.Axes:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            vmin (float, optional): _description_. Defaults to -1e-7.
            vmax (float, optional): _description_. Defaults to 1e-7.

        Returns:
            matplotlib.axes._axes.Axes: _description_
        """
        if binary:
            ax = sns.heatmap(self.signal >= threshold, cmap='RdBu', center=0)
            self.threshold = threshold

        else:
            ax = sns.heatmap(
                self.signal,
                cmap='RdBu',
                center=0,
                vmin=vmin,
                vmax=vmax
            )
        return ax

    def gauss_filter(self, sigma: float = 10) -> pd.DataFrame:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            sigma (float, optional): _description_. Defaults to 10.

        Returns:
            pd.DataFrame: _description_
        """
        gauss_df = gaussian_filter(np.abs(self.signal), sigma=sigma)
        gauss_df = pd.DataFrame(gauss_df, index=self.signal.index)
        self.signal = gauss_df

    def detect_events(
            self,
            threshold: float = 4e-8
    ) -> matplotlib.axes._axes.Axes:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            threshold (float, optional): _description_. Defaults to 4e-8.

        Returns:
            matplotlib.axes._axes.Axes: _description_
        """
        events = np.argwhere(self.signal.values >= threshold)
        self.events = events
        clustering = DBSCAN(eps=3, min_samples=2).fit(events)
        ax = sns.scatterplot(
            x=events[:, 1],
            y=self.signal.shape[0] - events[:, 0],
            s=0.1,
            hue=clustering.labels_,
            palette='tab10',
            legend=False
        )
        plt.xlim(0, self.signal.shape[1])
        plt.ylim(0, self.signal.shape[0])
        return ax

    def save_events(
            self,
            folder_path: str = '../data/interim/'
    ) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
        """
        clustering = DBSCAN(eps=3, min_samples=2).fit(self.events)
        events_df = (
            pd.DataFrame(self.events, columns=['Time', 'Channel'])
            .assign(Time=lambda df: len(self.signal) - df['Time'])
            .assign(Cluster=clustering.labels_)
            .groupby('Cluster')
            .mean()
            .round(0)
            .astype(int)
        )
        for i in events_df.index:
            time_center = len(self.signal) - events_df.iloc[i].to_list()[0]
            channel_center = events_df.iloc[i].to_list()[1]
            time_name = (
                self.signal
                .index[time_center]
                .to_pydatetime()
                .strftime('%H%M%S')
            )
            channel_name = f'{channel_center:03d}'
            data_cut = (
                self.signal.iloc[
                    time_center - 60: time_center + 60,
                    channel_center - 20: channel_center + 20
                ]
            )
            data_cut.to_hdf(
                f'{folder_path}{time_name}_{channel_name}.hdf5',
                key='abc'
            )
