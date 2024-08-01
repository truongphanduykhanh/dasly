"""Provides convenient methods to sample DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-07-01'

import logging
import logging.config
from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import decimate

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DASSampler:

    def __init__(self):
        self.signal: pd.DataFrame = None
        self.t_rate: float = None
        self.s_rate: float = None

    def sample(
        self,
        seconds: int = None,
        meters: int = None,  # should be a multipler of the channels gap
        agg_func: callable = np.nanmean,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Sample the data by a given number of seconds and meters.

        The data is grouped by the given number of seconds and meters, and then
        aggregated by a given function. The function can be one of the
        following: 'first', 'last', 'mean', 'median', 'max', 'min', 'std',
        'var', 'sum'.
        Args:
            seconds (int, optional): Number of seconds to group the data.
                Defaults to None.
            meters (int, optional): Number of meters to group the data.
                Defaults to None.
            agg_func (callable, optional): Aggregation function. Should be one
                that ignore NaN values. Defaults to np.nanmean.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # check arguments
        if seconds is None:  # keep the original sampling rate
            seconds = 1 / self.t_rate
        if meters is None:  # keep the original sampling rate
            meters = 1 / self.s_rate

        signal_np = self.signal.to_numpy()
        signal_shape = signal_np.shape
        block_shape = (int(seconds * self.t_rate), int(meters * self.s_rate))

        # Calculate the new shape of the reduced image
        signal_shape_new = (
            int(np.ceil(signal_shape[0] / block_shape[0])),
            int(np.ceil(signal_shape[1] / block_shape[1]))
        )

        # Calculate padding for height
        pad_height = (
            (block_shape[0] - signal_shape[0] % block_shape[0])
            % block_shape[0]
        )
        # Calculate padding for width
        pad_width = (
            (block_shape[1] - signal_shape[1] % block_shape[1])
            % block_shape[1]
        )

        # Pad the image with zeros so that its dimensions are exactly divisible
        # by the block size
        signal_padded = np.pad(
            signal_np,
            ((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=np.nan
        )
        signal_reshaped = signal_padded.reshape((
            signal_shape_new[0],
            block_shape[0],
            signal_shape_new[1],
            block_shape[1]
        ))
        signal_sample = agg_func(signal_reshaped, axis=(1, 3))

        # index and columns
        df_idx = self.signal.index[::block_shape[0]]
        df_cols = self.signal.columns[::block_shape[1]]

        signal_sample = pd.DataFrame(
            signal_sample,
            index=df_idx,
            columns=df_cols
        )
        # return
        #######################################################################
        if inplace:
            self.signal = signal_sample
            self._update_t_rate()  # in das_loader.py
            self._update_s_rate()  # in das_loader.py
            logger.info('Signal updated with new temporal sampling rate '
                        + f'{self.t_rate:.0f} and new spatial sampling rate '
                        + f'{self.s_rate:.3f}.')
            return None
        else:
            return signal_sample

    def decimate(
        self,
        factor: int = None,
        freq: float = None,
        t_rate: int = None,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Decimate the data. The final sampling rate is adjusted to be a
        divisor of original sampling rate. E.g. 100 to 50, 25 or 5. There are
        three options to decimate:
            1. by factor
            2. by frequency
            3. by sampling_rate
        - If by factor, the data is decimated by the given factor or smaller,
        to ensure the final temporal sampling rate is still a rounded number.
        - If by frequency, the output data will have smallest sampling rate
        that is at least as twice as the given desired frequency.
        - If by sampling rate, the output data will have smallest sampling rate
        that is greater than the given desired sampling rate.
        Args:
            factor (int, optional): Desired downsampling factor.
            freq (float, optional): Desried frequency.
            t_rate (int, optional): Desired temportal sampling rate.
            inplace (bool, optional): If overwrite attribute signal. Defaults
                to True.

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
        """
        # check arguments
        #######################################################################
        provided_args = sum(arg is not None for arg in [factor, freq, t_rate])
        if provided_args != 1:
            raise ValueError('The function accepts one and only one out of '
                             + 'three (factor, freq, t_rate).')

        # decimating
        #######################################################################
        def find_factor(t_rate: int, factor: int) -> int:
            """Find the largest divisor of t_rate that is smaller than factor.

            For example, let assume t_rate = 100. If factor = 5, 6, 7, 8, 9 or
            10, the function will return 5, 5, 5, 5, 5, 10, respectively.
            """
            for i in range(factor, 1, -1):
                if t_rate % i == 0:
                    return i
            return 1

        def decimate_by_factor(
            data: pd.DataFrame,
            t_rate: int,
            desired_factor: int
        ) -> pd.DataFrame:
            """Decimate the data by a desired factor.
            """
            adjusted_factor = find_factor(t_rate, desired_factor)
            decimated_data = decimate(data, adjusted_factor, axis=0)
            return decimated_data, adjusted_factor

        def decimate_by_freq(
            data: pd.DataFrame,
            t_rate: int,
            desired_freq: int
        ) -> pd.DataFrame:
            """Decimate the data by a desired frequency.
            """
            target_rate = max(desired_freq * 2, 1)
            adjusted_factor = find_factor(
                t_rate,
                int(t_rate / target_rate))
            decimated_data = decimate(data, adjusted_factor, axis=0)
            return decimated_data, adjusted_factor

        def decimate_by_sampling_rate(
            data: pd.DataFrame,
            t_rate: int,
            desired_t_rate: int
        ) -> pd.DataFrame:
            """Decimate the data by a desired sampling rate.
            """
            adjusted_factor = find_factor(
                t_rate,
                int(t_rate / desired_t_rate))
            decimated_data = decimate(data, adjusted_factor, axis=0)
            return decimated_data, adjusted_factor

        if factor is not None:
            signal_decimate, adjusted_factor = decimate_by_factor(
                data=self.signal,
                t_rate=self.t_rate,
                desired_factor=factor
            )

        elif freq is not None:
            signal_decimate, adjusted_factor = decimate_by_freq(
                data=self.signal,
                t_rate=self.t_rate,
                desired_freq=freq
            )

        else:
            signal_decimate, adjusted_factor = decimate_by_sampling_rate(
                data=self.signal,
                t_rate=self.t_rate,
                desired_t_rate=t_rate
            )

        # correct the index
        #######################################################################
        idx = slice(0, len(self.signal.index), adjusted_factor)
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
            self._update_t_rate()  # in das_loader.py
            logger.info('Signal updated with new temporal sampling rate '
                        + f'{self.t_rate:.0f}.')
            return None
        else:
            return signal_decimate
