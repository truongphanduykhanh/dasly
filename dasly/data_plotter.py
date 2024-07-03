"""Provides convenient methods to visualize DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-07-01'

import logging
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set_theme()


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DataPlotter:
    """Visualize DAS data."""

    def __init__(self):
        self.signal: pd.DataFrame = None
        self.lines: pd.DataFrame = None

    def check_data_type(
        self,
        data: Union[pd.DataFrame, np.ndarray] = None
    ) -> str:
        """Check data type in attribute signal or input data.

        Args:
            data (Union[pd.DataFrame, np.ndarray], optional): Input data.
            If None, attribute signal is assessed. Defaults to None.

        Returns:
            str: Either ['binary', 'gray', 'positive', 'float']
        """
        if data is None:
            data = self.signal
        data = pd.DataFrame(data)
        if ((data == 0) | (data == 1)).all().all():
            data_type = 'binary'
        elif ((data >= 0) & (data <= 255)).all().all():
            data_type = 'gray'
        elif (data > 0).all().all():
            data_type = 'positive'
        else:
            data_type = 'float'
        return data_type

    def heatmap(
        self,
        data: Union[pd.DataFrame, np.ndarray] = None,
        vmin: float = None,
        vmax: float = None,
        aspect: float = None,
        xlabel: str = 'Channel',
        ylabel: str = 'Time'
    ) -> None:
        """Plot heatmap.

        Args:
            data (Union[pd.DataFrame, np.ndarray], optional): Data to be
                heatmaped. If None, attribute signal is used. Defaults to None.
            vmin (Union[float, str], optional): Values to anchor the colormap.
                If None, automatically choose suitable value. Defaults to None.
            vmax (Union[float, str], optional): Values to anchor the colormap.
                If None, automatically choose suitable value. Defaults to None.
            xlabel (str, optional): Label for x-axis. Defaults to 'Channel'.
            ylabel (str, optional): Label for y-axis. Defaults to 'Time'.
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
            logger.info('Expect to display in {relative_data_size:.0f}',
                        ' seconds. Consider to sample the data.')

        # Define cmap and norm based on data type
        #######################################################################
        data_type = self.check_data_type(data)
        if data_type == 'binary':
            cmap = colors.ListedColormap(['black', 'white'])
            # cmap = 'gray'
            vmin = 0
            vmax = 1
        elif data_type in ['gray', 'positive']:
            cmap = 'viridis'
            vmin = 0
            if vmax is None:
                percentile = np.quantile(data, 0.95)
                vmax = percentile
            logger.info(f'vmax: {vmax:.3g}')
        else:
            cmap = 'RdBu'
            if (vmin is None) or (vmax is None):
                percentile = np.quantile(np.abs(data), 0.95)
                vmin = - percentile
                vmax = percentile
                logger.info(f'vmin: {vmin:.3g}, vmax: {vmax:.3g}')
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vmax=vmax,
            vcenter=(vmin + vmax) / 2
        )

        # plot heatmap
        #######################################################################
        if aspect is None:
            aspect = data.shape[1] / data.shape[0]  # square
        im = plt.imshow(
            X=data,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            interpolation='none',
            origin='lower'
        )

        # Adjust ticks
        #######################################################################
        # Get the current ticks
        current_xticks = plt.gca().get_xticks()[1:-1]
        current_yticks = plt.gca().get_yticks()[1:-1]

        # Create new tick
        new_xticks = data.columns[current_xticks.astype(int)]
        new_yticks = data.index[current_yticks.astype(int)]

        # Set the new tick
        plt.gca().set_xticks(current_xticks)
        plt.gca().set_xticklabels(new_xticks)

        plt.gca().set_yticks(current_yticks)
        plt.gca().set_yticklabels(new_yticks)

        # Axis labels
        #######################################################################
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Add colorbar
        #######################################################################
        if data_type == 'binary':  # discrete ticks
            cbar = plt.colorbar(im, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['0', '1'])
        else:
            # following code is just incase the "aspect" parameter in imshow()
            # is set, which makes the colorbar not match the plot height.
            # Otherwise, just just plt.colorbar() for simplicity
            ax = plt.gca()
            fig = plt.gcf()
            pos = ax.get_position()  # Get the position of the plot
            # Manually adjust the position of the colorbar
            cbar_ax = fig.add_axes([pos.x1 + 0.03, pos.y0, 0.03, pos.height])
            cbar = fig.colorbar(ax.images[0], cax=cbar_ax)

        # Add Hough transform lines if available
        #######################################################################
        if self.lines is not None:
            for line in self.lines.iloc[:, 0:4].values:  # loop over the lines
                x1, y1, x2, y2 = line
                plt.plot([x1, x2], [y1, y2])
