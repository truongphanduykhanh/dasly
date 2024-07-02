"""Provides convenient methods to visualize DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-07-01'

import logging
from typing import Literal, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DataVisualizer:
    """Visualize DAS data."""

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
