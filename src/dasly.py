from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


import simpleDASreader


def load_das(
        folder_path: str,
        start: str | datetime = None,  # yyyyMMdd HHmmss
        end: str | datetime = None,  # yyyyMMdd HHmmss
        duration: int = None,
) -> simpleDASreader.DASDataFrame:
    """Load DAS files into simpleDASreader.DASDataFrame (pandas dataframe)

    Args:
        folder_path (str): File path containing DAS files
        start (str | datetime, optional): Start time. Defaults to None.
        end (str | datetime, optional): Upto time, excluded. Defaults to None.
        duration (int, optional): Duration in seconds. Defaults to None.

    Raises:
        ValueError: Two and only two arguments of (start, end, duration)

    Returns:
        simpleDASreader.DASDataFrame: Output data frame
    """

    # Check arguements
    ###########################################################################
    if (start is None) + (end is None) + (duration is None) != 1:
        raise ValueError("The function accepts two and only two of (start, \
                         end, duration)")

    if isinstance(start, str):
        start = datetime.strptime(start, '%Y%m%d %H%M%S')
    if isinstance(end, str):
        end = datetime.strptime(end, '%Y%m%d %H%M%S')

    if duration is None:
        duration = (end - start).seconds
    elif start is None:
        start = end - timedelta(seconds=duration)
    elif end is None:
        end = start + timedelta(seconds=duration)

    # Get file paths
    ###########################################################################
    file_paths, _, _ = simpleDASreader.find_DAS_files(
        experiment_path=folder_path,
        start=start,
        duration=duration,
        show_header_info=False
    )
    first_file = file_paths[0].split("/")[-1].split(".")[0]
    last_file = file_paths[-1].split("/")[-1].split(".")[0]
    print(f'{len(file_paths)} files, from {first_file} to {last_file}')

    # Load files
    ###########################################################################
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

    # Transform dataframe
    ###########################################################################
    signal = signal[start:end]  # take extact the range start-end
    signal = signal.iloc[:-1]  # drop the last record (new second already)
    signal = signal.iloc[::-1]  # reverse the order of time, to plot easier
    return signal


def lowcut_filter(
        data: pd.DataFrame,
        cutoff_freq,
        nyq_freq,
        order=4,
        axis=-1
):
    """
    # Apply lowcut filter to data
    """
    sos = signal.butter(
        order,
        cutoff_freq/nyq_freq,
        btype='highpass',
        output='sos'
    )
    y = signal.sosfilt(sos, data, axis=axis)
    y = pd.DataFrame(y, index=data.index)
    return y


def highcut_filter(data, cutoff_freq, nyq_freq, order=4, axis=-1):
    """
    # Apply highcut filter to data
    """
    sos = signal.butter(
        order,
        cutoff_freq/nyq_freq,
        btype='lowpass',
        output='sos'
    )
    y = signal.sosfilt(sos, data, axis=axis)
    y = pd.DataFrame(y, index=data.index)
    return y


def decimate(data, dt, dt_out, axis=-1, padtype='line'):
    """
    Resample a seismic trace to coarser time sampling interval
    # Resample to a coarser grid
    # dt time interval can be in any unit,
    #   but dt_in and dt_out must be in the same unit.
    """
    y = signal.resample_poly(
        data,
        int(dt*1e6),
        int(dt_out*1e6),
        axis=axis,
        padtype=padtype
    )
    idx = slice(0, len(data.index), int(dt_out/dt))
    idx = data.index[idx]
    y = pd.DataFrame(y, index=idx)
    return y


def heatmap(
        data: pd.DataFrame,
        vmin: float = -1e-7,
        vmax: float = 1e-7
) -> matplotlib.axes._axes.Axes:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        vmin (float, optional): _description_. Defaults to -1e-7.
        vmax (float, optional): _description_. Defaults to 1e-7.

    Returns:
        matplotlib.axes._axes.Axes: _description_
    """
    ax = sns.heatmap(data, cmap='RdBu', center=0, vmin=vmin, vmax=vmax)
    return ax


def gauss_filter(data: pd.DataFrame, sigma: float = 10) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        sigma (float, optional): _description_. Defaults to 10.

    Returns:
        pd.DataFrame: _description_
    """
    gauss_df = gaussian_filter(np.abs(data), sigma=sigma)
    gauss_df = pd.DataFrame(gauss_df, index=data.index)
    return gauss_df


def plot_events(
        gauss_df: pd.DataFrame,
        threshold: float = 4e-8
) -> matplotlib.axes._axes.Axes:
    """_summary_

    Args:
        gauss_df (pd.DataFrame): _description_
        threshold (float, optional): _description_. Defaults to 4e-8.

    Returns:
        matplotlib.axes._axes.Axes: _description_
    """
    ax = sns.heatmap(gauss_df > threshold, cmap='RdBu', center=0)
    return ax


def detect_events(
        gauss_df: pd.DataFrame,
        threshold: float = 4e-8
) -> np.ndarray:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        threshold (float, optional): _description_. Defaults to 4e-8.

    Returns:
        np.ndarray: _description_
    """
    events = np.argwhere(gauss_df.values >= threshold)
    return events


def classify_events(
        gauss_df: pd.DataFrame,
        threshold: float = 4e-8
) -> matplotlib.axes._axes.Axes:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        threshold (float, optional): _description_. Defaults to 4e-8.

    Returns:
        matplotlib.axes._axes.Axes: _description_
    """
    events = np.argwhere(gauss_df.values >= threshold)
    clustering = DBSCAN(eps=3, min_samples=2).fit(events)
    ax = sns.scatterplot(
        x=events[:, 1],
        y=gauss_df.shape[0] - events[:, 0],
        s=0.1,
        hue=clustering.labels_,
        palette='tab10',
        legend=False
    )
    plt.xlim(0, gauss_df.shape[1])
    plt.ylim(0, gauss_df.shape[0])
    return ax


def save_das(
        data: pd.DataFrame,
        events: np.ndarray,
        folder_path: str = 'data/'
) -> None:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
    """
    clustering = DBSCAN(eps=3, min_samples=2).fit(events)
    events_df = (
        pd.DataFrame(events, columns=['Time', 'Channel'])
        .assign(Time=lambda df: len(data) - df['Time'])
        .assign(Cluster=clustering.labels_)
        .groupby('Cluster')
        .mean()
        .round(0)
        .astype(int)
    )
    for i in events_df.index:
        time_center = len(data) - events_df.iloc[i].to_list()[0]
        channel_center = events_df.iloc[i].to_list()[1]
        time_name = data.index[time_center].to_pydatetime().strftime('%H%M%S')
        channel_name = f'{channel_center:03d}'
        data_cut = (
            data.iloc[
                time_center - 60: time_center + 60,
                channel_center - 20: channel_center + 20
            ]
        )
        data_cut.to_hdf(
            f'{folder_path}{time_name}_{channel_name}.hdf5',
            key='abc'
        )
