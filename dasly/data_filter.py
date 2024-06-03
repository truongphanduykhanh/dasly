"""Provides convenient process to filter DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-06-01'

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.ndimage import convolve
import cv2
import torch
import torch.nn.functional as F


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DataFilter:
    """Filter DAS data."""

    def __init__(self):
        self.signal: pd.DataFrame = None
        self.t_rate: float = None
        self.s_rate: float = None

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
        if not (0 < low < high < 0.5 * self.t_rate):
            error_msg = """Invalid frequency bounds.
            Ensure 0 < low < high < nyquist."""
            logger.error(error_msg)
            raise ValueError(error_msg)
        nyquist = 0.5 * self.t_rate
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
        if not (0 < cutoff < 0.5 * self.t_rate):
            error_msg = """Invalid cutoff frequency.
            Ensure 0 < cutoff < nyquist."""
            logger.error(error_msg)
            raise ValueError(error_msg)
        nyquist = 0.5 * self.t_rate
        normalized_cutoff = cutoff / nyquist
        sos = butter(order, normalized_cutoff, btype='low', output='sos')
        signal_lowpass = sosfilt(sos, self.signal, axis=0)
        signal_lowpass = pd.DataFrame(
            signal_lowpass,
            index=self.signal.index,
            columns=self.signal.columns
        )
        # return
        if inplace:
            self.signal = signal_lowpass
            logger.info('Signal updated with low-pass filter.')
            return None
        else:
            return signal_lowpass

    def binary_transform(
        self,
        quantile: float = None,
        threshold: float = None,
        inplace: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Transform the signal attribute to a binary DataFrame.

        Takes the absolute value of the signal and maps it to 1 if it is
        greater  than or equal to the threshold, otherwise maps to 0.

        Args:
            quantile (float, optional): Quantile to compute threshold if no
                threshold  is provided. Defaults to None.
            threshold (float, optional): Fixed threshold value for the entire
                DataFrame. If None, the quantile is used to compute the
                threshold. Defaults to None.
            inplace (bool, optional): If True, modifies the signal attribute
                in place. Otherwise, returns a new DataFrame. Defaults to True.

        Returns:
            Optional[pd.DataFrame]: Returns a binary DataFrame if inplace is
                False. Otherwise, returns None.
        """
        signal_abs = np.abs(self.signal)
        # compute threshold
        if threshold is None:
            threshold = np.quantile(signal_abs, quantile)
            logger.info(f'threshold: {threshold}')
        signal_binary = (signal_abs >= threshold).astype(np.uint8)
        # return
        if inplace:
            self.signal = signal_binary
            logger.info('Signal updated with binary filter.')
            return None
        else:
            return signal_binary

    def grayscale_transform(
        self,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
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

    def grayscale_transform_cv2(
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

    @staticmethod
    def _inverse_pca(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray
    ) -> np.ndarray:
        """Calculate covariance matrix from eigenvalues and eigenvectors.

        This is a helper function for the Gaussian smoothing method.

        Args:
            eigenvalues (np.ndarray): Eigenvalues
            eigenvectors (np.ndarray): Eigenvectors

        Returns:
            np.ndarray: Covariance matrix
        """
        eigenvectors = np.matrix(eigenvectors)
        eigenvalues = eigenvalues * np.identity(len(eigenvalues))
        cov_mat = eigenvectors * eigenvalues * eigenvectors.T
        cov_mat = np.asarray(cov_mat)
        return cov_mat

    @staticmethod
    def _cal_cov_mat(s1: float, s2: float, std_space: float) -> np.matrix:
        """Calculate covariance matrix from speed limits and space standard
            deviation.

        This is a helper function for the Gaussian smoothing method.

        Args:
            s1 (float): Low speed limit in km/h
            s2 (float): High speed limit in km/h
            std_space (float): Standard deviation along the spatial dimension.

        Returns:
            np.matrix: Covariance matrix
        """
        var_space = std_space ** 2
        theta1 = np.arctan(3.6 / s1)
        theta2 = np.arctan(3.6 / s2)
        theta = 0.5 * (theta1 + theta2)
        eigenvector_1 = [1, np.tan(theta)]
        eigenvector_2 = [np.tan(theta), -1]
        eigenvalue_1 = var_space / np.cos(theta)
        eigenvalue_2 = np.tan(theta1 - theta) * eigenvalue_1
        cov_mat = DataFilter.inverse_pca(
            eigenvalues=[eigenvalue_1, eigenvalue_2],
            eigenvectors=np.array([eigenvector_1, eigenvector_2]).T
        )
        return cov_mat

    @staticmethod
    def _create_gaussian_kernel(
        cov_mat: np.ndarray,
        t_rate: int,
        s_rate: int,
    ) -> np.ndarray:
        """
        Create a Gaussian kernel from a covariance matrix.

        Args:
            cov_mat (np.ndarray): Covariance matrix (2x2 matrix).
            t_rate (int, optional): Sampling rate for temporal dimension.
            s_rate (int, optional): Sampling rate for spatial dimension.

        Returns:
            np.ndarray: Gaussian kernel.
        """
        # grid of points
        size_t = np.sqrt(cov_mat[1][1]) * 2 * 2 * t_rate  # 2 std
        size_s = np.sqrt(cov_mat[0][0]) * 2 * 2 * s_rate  # 2 std
        size_t = int(round(size_t, 0))
        size_s = int(round(size_s, 0))
        range_t = np.arange(-size_t // 2, size_t // 2 + 1) / t_rate
        range_s = np.arange(-size_s // 2, size_s // 2 + 1) / s_rate

        # Create a meshgrid of x (spatial) and y (temporal) coordinates
        x, y = np.meshgrid(range_s, range_t)

        # Calculate the determinant and inverse of the covariance matrix
        det_cov_mat = np.linalg.det(cov_mat)
        inv_cov_mat = np.linalg.inv(cov_mat)

        # Calculate the exponent of the Gaussian function
        exponent = -0.5 * (
            inv_cov_mat[0, 0] * x**2 +
            2 * inv_cov_mat[0, 1] * x * y +
            inv_cov_mat[1, 1] * y**2
        )

        # Compute the Gaussian filter
        gaussian_kernel = np.exp(exponent) / (2 * np.pi * np.sqrt(det_cov_mat))

        # Normalize the filter so that the sum of all elements is 1
        gaussian_kernel /= np.sum(gaussian_kernel)

        return gaussian_kernel

    def gaussian_smooth(
        self,
        s1: float,
        s2: float,
        std_space: float,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Apply a Gaussian smoothing filter to the signal.

        Args:
            s1 (float): Lower speed limit in km/h
            s2 (float): Upper speed limit in km/h
            std_space (float): Standard deviation along the space axis
            inplace (bool, optional): If True, overwrite the signal attribute.
                Defaults to True.

        Returns:
            Optional[pd.DataFrame]: Filtered signal as a DataFrame if
                inplace=False.
        """
        # calculate covariance matrix
        cov_mat = DataFilter._cal_cov_mat(s1, s2, std_space)
        gauss_filter = DataFilter._create_gaussian_kernel(
            cov_mat=cov_mat,
            t_rate=self.t_rate,
            s_rate=self.s_rate
        )
        # apply filter
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
        if inplace:
            self.signal = signal_gaussian
            logger.info('Signal updated with Gaussian smooth.')
            return None
        else:
            return signal_gaussian

    def sobel_filter(self, inplace: bool = True) -> Optional[pd.DataFrame]:
        """Apply Sobel operator to detect edges in the signal attribute.

        The Sobel operator is used to detect gradients in the x and y
        directions. The magnitude of the gradient is then calculated to
        highlight edges.

        Args:
            inplace (bool, optional): If True, modifies the signal attribute in
                place. Otherwise, returns a new DataFrame. Defaults to True.

        Returns:
            Optional[pd.DataFrame]: Returns a DataFrame with Sobel filter
                applied if inplace is False. Otherwise, returns None.
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
        if inplace:
            self.signal = signal_sobel
            logger.info('Signal updated with Sobel filter.')
            return None
        else:
            return signal_sobel
