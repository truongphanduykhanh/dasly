"""Provides convenient methods to filter DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-06-01'

import logging
from typing import Literal, Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from dasly.analyzer import DASAnalyzer

sns.set_theme()

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DASFilter:
    """Filter DAS data."""

    def __init__(
        self,
        signal: pd.DataFrame = None,
        t_rate: float = None,
        s_rate: float = None
    ):
        self.signal = signal
        self.t_rate = t_rate
        self.s_rate = s_rate

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
            raise ValueError('Invalid frequency bounds. Ensure 0 < low < high '
                             + '< nyquist.')
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
            raise ValueError('Invalid cutoff frequency. Ensure 0 < cutoff < '
                             + 'nyquist.')
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

    def highpass_filter(
        self,
        cutoff: float,
        order: int = 4,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Apply a high-pass filter to the signal.

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
            raise ValueError('Invalid cutoff frequency. Ensure 0 < cutoff < '
                             + 'nyquist.')
        nyquist = 0.5 * self.t_rate
        normalized_cutoff = cutoff / nyquist
        sos = butter(order, normalized_cutoff, btype='high', output='sos')
        signal_highpass = sosfilt(sos, self.signal, axis=0)
        signal_highpass = pd.DataFrame(
            signal_highpass,
            index=self.signal.index,
            columns=self.signal.columns
        )
        # return
        if inplace:
            self.signal = signal_highpass
            logger.info('Signal updated with high-pass filter.')
            return None
        else:
            return signal_highpass

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
        signal_binary = (signal_abs >= threshold).astype(np.uint8)
        # return
        if inplace:
            self.signal = signal_binary
            logger.info('Signal updated with binary transform with threshold '
                        + f'{threshold:.3g}.')
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
        theta1, _theta2, theta = DASFilter._cal_speed_angle(s1, s2, unit)

        # calculate covariance matrix
        cov_mat = DASFilter._cal_cov_mat(
            sigma11=std_s ** 2 if std_s else None,  # variance in space
            sigma22=std_t ** 2 if std_t else None,  # variance in time
            eigvec=[1, np.tan(theta)],
            eigval_prop=1 / np.tan(theta1 - theta)
        )
        self.cov_mat = cov_mat

        gauss_kernel = DASFilter._create_gauss_kernel(
            cov_mat=cov_mat,
            s_rate=self.s_rate,
            t_rate=self.t_rate
        )

        # cv2 filter (correlation). correlation does not flip the kernel.
        # convolution flips the kernel, which is not desired for image blurring
        # so when the kernel is not symmetric, correlation is preferred
        # when the kernel is symmetric, correlation and convolution will have
        # same result (even though correlation is more ideally correct)
        signal_gauss = cv2.filter2D(
            np.asarray(self.signal), -1, np.asarray(gauss_kernel)
        )
        signal_gauss = pd.DataFrame(
            signal_gauss,
            index=self.signal.index,
            columns=self.signal.columns
        )

        # Create a DataFrame for the Gaussian kernel for saving as an attribute
        idx = [i * 1/self.t_rate for i in range(gauss_kernel.shape[0])]
        col = [i * 1/self.s_rate for i in range(gauss_kernel.shape[1])]
        self.gauss_kernel = pd.DataFrame(
            gauss_kernel,
            index=idx,
            columns=col
        )
        # return
        #######################################################################
        if inplace:
            self.signal = signal_gauss
            logger.info('Signal updated with Gaussian smoothing.')
            return None
        else:
            return signal_gauss

    def sobel_filter(
        self,
        pos_grads: bool = True,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Apply Sobel operator to detect edges in the signal attribute.

        The Sobel operator is used to detect gradients in the x and y
        directions. The magnitude of the gradient is then calculated to
        highlight edges.

        Args:
            pos_grads (bool, optional): If True, only positive gradients are
                kept. Negative gradients are set to zero. Defaults to True.
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

        if pos_grads:
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

    def fft(
        self,
        data: np.ndarray = None,
        agg_func: callable = np.mean,
        flim: tuple = None,  # (low, high) inclusive
        plot: bool = True,
        power_lim: Tuple[int, int] = (-1, 1),
        ylim: Tuple[float, float] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute the Fast Fourier Transform (FFT) of the signal.

        Args:
            data (np.ndarray, optional): Data to compute the FFT. If None, the
                signal attribute is used. Defaults to None.
            agg_func (callable, optional): Aggregation function to apply to the
                FFT results. Defaults to np.mean.
            flim (tuple, optional): Frequency limits to filter the results.
            plot (bool, optional): If True, plot the FFT results. If False,
                return the frequencies and aggregated spectrum. Defaults to
                True.
            power_lim (Tuple[int, int], optional): Threshold for scientific
                notation. Defaults to (-1, 1).
            ylim (Tuple[float, float], optional): Y-axis limits for the plot.
                Defaults to None.

        Returns:
            Union[None, Tuple[np.ndarray, np.ndarray]]: Returns None if plot is
                True. Otherwise, returns the frequencies and aggregated
                spectrum as a tuple.
        """
        if data is None:
            data = self.signal.to_numpy()
        # Compute the FFT for each time series (each column) across all rows
        fft_results = np.fft.fft(data, axis=0)
        # Compute the magnitude of the FFT results
        fft_magnitudes = np.abs(fft_results)
        # Aggregate the magnitudes across space indices
        agg_spectrum = agg_func(fft_magnitudes, axis=1)
        # Calculate frequency axis
        n_samples = data.shape[0]
        frequencies = np.fft.fftfreq(n_samples, d=1/self.t_rate)

        # Only keep the positive half of the spectrum
        agg_spectrum = agg_spectrum[:n_samples//2]
        frequencies = frequencies[:n_samples//2]

        if flim:
            mask = (frequencies >= flim[0]) & (frequencies <= flim[1])
            agg_spectrum = agg_spectrum[mask]
            frequencies = frequencies[mask]

        if not plot:  # Return the frequencies and aggregated spectrum
            return frequencies, agg_spectrum  # end function

        # Plot the FFT results
        fig, ax = plt.subplots()
        ax.plot(frequencies, agg_spectrum)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])

        # Setting y-axis to scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(power_lim)  # Set the threshold for notation
        ax.yaxis.set_major_formatter(formatter)

    def fft_windows(
        self,
        line: np.ndarray,
        t: int,
        data: np.ndarray = None,
        agg_func: callable = np.mean,
        flim: tuple = None,  # (low, high) inclusive
        plot: bool = True,
        power_lim: Tuple[int, int] = (-1, 1),
        ylim: Tuple[float, float] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute the windows FFT for of the signal.

        Args:
            line (np.ndarray): Line to extract the section.
            t (int): Half-width of the section from the row index.
            data (np.ndarray, optional): Data to compute the FFT. If None, the
                signal attribute is used. Defaults to None.
            agg_func (callable, optional): Aggregation function to apply to the
                FFT results. Defaults to np.mean.
            flim (tuple, optional): Frequency limits.
            plot (bool, optional): If True, plot the FFT results. If False,
                return the frequencies and aggregated spectrum. Defaults to
                True.
            power_lim (Tuple[int, int], optional): Threshold for scientific
                notation. Defaults to (-1, 1).
            ylim (Tuple[float, float], optional): Y-axis limits for the plot.
                Defaults to None.

        Returns:
            Union[None, Tuple[np.ndarray, np.ndarray]]: Returns None if plot is
                True. Otherwise, returns the frequencies and aggregated
                spectrum as a tuple.
        """
        if data is None:
            data = self.signal.to_numpy()

        line_rehsaped = line.reshape(1, 4)
        y_vals = DASAnalyzer._y_vals_lines(
            lines=line_rehsaped,
            x_coords=np.arange(line[0], line[2] + 1)
        )
        y_vals = np.round(y_vals).astype(int)
        y_vals = y_vals.reshape(-1)

        def extract_section(
            data: np.ndarray,  # shape (m, n)
            col_idx: np.ndarray,  # shape (n ,)
            row_idx: np.ndarray,  # shape (n ,)
            t: int
        ) -> np.ndarray:
            """Extract a section of the data.

            Args:
                data (np.ndarray): 2D data array.
                col_idx (np.ndarray): Column indices to extract.
                row_idx (np.ndarray): Corresponding row indices to extract.
                t (int): Half-width of the section from the row index.

            Returns:
                np.ndarray: Extracted section of the data.
            """
            # Generate the row index ranges
            row_range = np.arange(-t, t + 1)
            row_indices_expanded = row_idx[:, None] + row_range

            # Clip the indices to ensure they are within the valid range
            row_indices_expanded = np.clip(
                row_indices_expanded, 0, data.shape[0] - 1)

            # Use advanced indexing to extract the data
            result = data[row_indices_expanded, col_idx[:, None]]

            # Transpose the result to get the desired shape (2*t + 1, m)
            result = result.T
            return result

        # Extract the section of the signal
        section = extract_section(
            data=data,
            col_idx=np.arange(line[0], line[2] + 1),
            row_idx=y_vals,
            t=t
        )

        # Compute the FFT for the extracted section
        return self.fft(
            data=section,
            agg_func=agg_func,
            flim=flim,
            plot=plot,
            power_lim=power_lim,
            ylim=ylim
        )
