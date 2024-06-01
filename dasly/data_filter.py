"""Provides convenient process to filter DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-06-01'

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, sosfilt
from scipy.ndimage import convolve
import cv2
import torch
import torch.nn.functional as F

from dasly import helper


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DataFilter:
    """Filter DAS data."""

    def __init__(self):
        self.signal: pd.DataFrame = None
        self.sampling_rate: float = None

    def bandpass_filter(
        self,
        low: float,
        high: float,
        order: int = 4,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Apply a band-pass filter to the signal.

        Args:
            low (float): Lower bound frequency
            high (float): Upper bound frequency
            order (int, optional): Order of IIR filter. Defaults to 4.
            inplace (bool, optional): If True, overwrite the signal attribute.
                Defaults to True.

        Returns:
            Optional[pd.DataFrame]: Filtered signal as a DataFrame if
                inplace=False.
        """
        if not (0 < low < high < 0.5 * self.sampling_rate):
            error_msg = """Invalid frequency bounds.
            Ensure 0 < low < high < nyquist."""
            logger.error(error_msg)
            raise ValueError(error_msg)
        nyquist = 0.5 * self.sampling_rate
        normalized_low = low / nyquist
        normalized_high = high / nyquist
        sos = butter(
            order,
            [normalized_low, normalized_high],
            btype='band',
            output='sos'
        )
        signal_bandpass = sosfilt(sos, self.signal, axis=0)
        signal_bandpass = pd.DataFrame(
            signal_bandpass,
            index=self.signal.index,
            columns=self.signal.columns
        )
        # return
        if inplace:
            self.signal = signal_bandpass
            logger.info('Signal updated with band-pass filter.')
            return None
        else:
            return signal_bandpass

    def lowpass_filter(
        self,
        cutoff: float,
        order: int = 4,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Apply a low-pass filter to the signal.

        Args:
            cutoff (float): Cut-off frequency.
            order (int, optional): Order of IIR filter. Defaults to 4.
            inplace (bool, optional): If True, overwrite the signal attribute.
                Defaults to True.

        Returns:
            Optional[pd.DataFrame]: Filtered signal as a DataFrame if
                inplace=False.
        """
        if not (0 < cutoff < 0.5 * self.sampling_rate):
            error_msg = """Invalid cutoff frequency.
            Ensure 0 < cutoff < nyquist."""
            logger.error(error_msg)
            raise ValueError(error_msg)
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = cutoff / nyquist
        sos = butter(order, normalized_cutoff, btype='low', output='sos')
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
            logger.info('Signal updated with low-pass filter.')
            return None
        else:
            return signal_lowpass

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

    def gray_filter(self, inplace: bool = True) -> Optional[pd.DataFrame]:
        """Transform data to grayscale 0 to 255 using min-max scalling.

        Args:
            inplace (bool, optional): If True, overwrite the signal attribute.
                Defaults to True.

        Returns:
            Optional[pd.DataFrame]: Grayscale signal as a DataFrame if
                inplace=False.
        """
        # min-max normalization
        signal_gray = np.abs(self.signal)
        min_val = np.min(signal_gray)
        max_val = np.max(signal_gray)
        signal_gray = ((signal_gray - min_val) / (max_val - min_val) * 255)
        signal_gray = signal_gray.round(0).astype(np.uint8)
        # return
        #######################################################################
        if inplace:
            self.signal = signal_gray
            logger.info('Signal updated with grayscale filter.')
            return None
        else:
            return signal_gray

    def gray_filter_cv2(
        self,
        inplace: bool = True
    ) -> Union[None, pd.DataFrame]:
        """Transform data to grayscale 0 to 255 using cv2 scalling.

        Args:
            inplace (bool, optional): If True, overwrite the signal attribute.
                Defaults to True.

        Returns:
            Optional[pd.DataFrame]: Grayscale signal as a DataFrame if
                inplace=False.
        """
        # cv2 methods
        signal_scaled = self.signal / np.quantile(self.signal, 0.5)
        signal_gray = cv2.convertScaleAbs(signal_scaled)
        # return
        #######################################################################
        if inplace:
            self.signal = signal_gray
            logger.info('Signal updated with cv2 grayscale filter.')
            return None
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
        )
        filter_tensor = torch.tensor(
            gauss_filter,
            dtype=torch.float32
        )
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
