"""Provides convenient process to filter DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-06-01'

import logging
from typing import Literal, Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.ndimage import convolve
import cv2


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
        greater than or equal to the threshold, otherwise maps to 0.

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
    def _cal_speed_angle(
        s1: float,
        s2: float,
        unit: Literal['km/h', 'm/s'] = 'km/h'  # 'km/h' or 'm/s'
    ) -> Tuple[float, float, float, float]:
        """Calculate the angles of the speed limits and their average.

        Args:
            s1 (float): Speed of the first velocity vector.
            s2 (float): Speed of the second velocity vector.
            unit (Literal['km/h', 'm/s'], optional): Unit of the speed.
                Defaults to 'km/h'.

        Returns:
            Tuple[float, float, float]: Angles of the speed limits and the
                average.
        """
        # convert to m/s
        if unit == 'km/h':
            s1 = s1 / 3.6
            s2 = s2 / 3.6
        # calculate angle
        theta1 = np.arctan(1 / s1)
        theta2 = np.arctan(1 / s2)
        theta = 0.5 * (theta1 + theta2)
        return theta1, theta2, theta

    @staticmethod
    def _cal_cov_mat(
        sigma11: float = None,
        sigma22: float = None,
        eigvec: Union[np.ndarray, list] = None,
        eigval_prop: float = None
    ) -> np.ndarray:
        """Reconstruct the covariance matrix (of a 2D Gaussian distribution)
            given one variance, the first eigenvector (upto a scalar multiple),
            and the proportion of eigenvalues.

        Args:
            sigma11 (float, optional): Variance along the first axis (x-axis).
                Defaults to None.
            sigma22 (float, optional): Variance along the second axis (y-axis).
                Defaults to None.
            eigvec (Union[np.ndarray, list], optional): First eigenvector v1.
                Defaults to None.
            eigval_prop (float, optional): Proportion of the two eigenvalues
                (lambda1 / lambda2). Defaults to None.

        Returns:
            np.ndarray: The reconstructed covariance matrix.
        """
        if (sigma11 is None) + (sigma22 is None) != 1:
            raise ValueError('Either sigma11 or sigma22 must be provided.')
        if sigma11:
            denominator = eigval_prop * (eigvec[0] ** 2) + (eigvec[1] ** 2)
            sigma12 = eigvec[0] * eigvec[1] * (eigval_prop - 1)
            sigma12 = sigma11 * sigma12 / denominator
            sigma22 = eigval_prop * (eigvec[1] ** 2) + (eigvec[0] ** 2)
            sigma22 = sigma11 * sigma22 / denominator
        else:
            denominator = eigval_prop * (eigvec[1] ** 2) + (eigvec[0] ** 2)
            sigma12 = eigvec[0] * eigvec[1] * (eigval_prop - 1)
            sigma12 = sigma22 * sigma12 / denominator
            sigma11 = eigval_prop * (eigvec[0] ** 2) + (eigvec[1] ** 2)
            sigma11 = sigma22 * sigma11 / denominator
        cov_mat = np.array([[sigma11, sigma12], [sigma12, sigma22]])
        return cov_mat

    @staticmethod
    def _create_gauss_kernel(
        cov_mat: np.ndarray,
        s_rate: float,
        t_rate: float,
        std_multi: float = 2
    ):
        """Create a 2D Gaussian kernel from the covariance matrix.

        Args:
            cov_mat (np.ndarray): Covariance matrix.
            s_rate (float): Spatial sampling rate.
            t_rate (float): Temporal sampling rate.
            std_multi (float, optional): Number of standard deviation to cover.
                Defaults to 2.

        Returns:
            np.ndarray: Gaussian kernel.
        """
        # Adjust the covariance matrix to the actual sampling rates
        cov_mat_adj = cov_mat.copy()
        cov_mat_adj[0, 0] *= s_rate ** 2
        cov_mat_adj[1, 1] *= t_rate ** 2
        cov_mat_adj[0, 1] *= s_rate * t_rate
        cov_mat_adj[1, 0] = cov_mat_adj[0, 1]

        # Get the standard deviations
        sigma_s = np.sqrt(cov_mat_adj[0, 0])
        sigma_t = np.sqrt(cov_mat_adj[1, 1])

        # Define the size of the kernel to cover 2 std each side
        kernel_size_s = int(np.ceil(sigma_s * std_multi))
        kernel_size_t = int(np.ceil(sigma_t * std_multi))

        # Create a grid of (x, y) coordinates
        range_s = np.arange(-kernel_size_s, kernel_size_s + 1)
        range_t = np.arange(-kernel_size_t, kernel_size_t + 1)
        x, y = np.meshgrid(range_s, range_t)

        # Calculate the Gaussian values
        det_cov_mat = np.linalg.det(cov_mat_adj)  # determinant of cov. matrix
        inv_cov_mat = np.linalg.inv(cov_mat_adj)  # inverse covariance matrix
        exponent = -0.5 * (
            inv_cov_mat[0, 0] * x**2 +
            2 * inv_cov_mat[0, 1] * x * y +
            inv_cov_mat[1, 1] * y**2
        )
        gauss_kernel = np.exp(exponent) / (2 * np.pi * np.sqrt(det_cov_mat))
        # Normalize the filter
        gauss_kernel /= np.sum(gauss_kernel)

        return gauss_kernel

    def gaussian_smooth(
        self,
        s1: float,
        s2: float,
        std_s: float = None,
        std_t: float = None,
        unit: Literal['km/h', 'm/s'] = 'km/h',  # 'km/h' or 'm/s'
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Apply a Gaussian smoothing filter to the signal.

        Args:
            s1 (float): Lower speed limit.
            s2 (float): Upper speed limit.
            std_s (float, optional): Standard deviation along the space
                dimension. In meters. Defaults to None.
            std_t (float, optional): Standard deviation along the time
                dimension. In seconds. Defaults to None.
            inplace (bool, optional): If True, overwrite the signal attribute.
                Defaults to True.
            unit (Literal['km/h', 'm/s'], optional): Unit of the speed.
                Defaults to 'km/h'.

        Returns:
            Optional[pd.DataFrame]: Filtered signal as a DataFrame if
                inplace=False.
        """
        # calculate speed angles
        theta1, _theta2, theta = DataFilter._cal_speed_angle(s1, s2, unit)

        # calculate covariance matrix
        cov_mat = DataFilter._cal_cov_mat(
            sigma11=std_s ** 2 if std_s else None,  # variance in space
            sigma22=std_t ** 2 if std_t else None,  # variance in time
            eigvec=[1, np.tan(theta)],
            eigval_prop=1 / np.tan(theta1 - theta)
        )
        gauss_kernel = DataFilter._create_gauss_kernel(
            cov_mat=cov_mat,
            s_rate=self.s_rate,
            t_rate=self.t_rate
        )
        self.gauss_kernel = gauss_kernel

        # cv2 filter (correlation). correlation does not flip the kernel.
        # convolution flips the kernel, which is not desired for image blurring
        # so when the kernel is not symmetric, correlation is preferred
        # when the kernel is symmetric, correlation and convolution will have
        # same result (even though correlation is more ideally correct)
        signal_gauss = cv2.filter2D(self.signal, -1, gauss_kernel)
        signal_gauss = pd.DataFrame(
            signal_gauss,
            index=self.signal.index,
            columns=self.signal.columns
        )
        # return
        #######################################################################
        if inplace:
            self.signal = signal_gauss
            logger.info('Signal updated with Gaussian smoothing.')
            return None
        else:
            return signal_gauss

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
