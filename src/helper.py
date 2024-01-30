import os
from typing import Union
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
sns.set()


def speed_to_angle(speed_kmh: float, duration_s: int = 60) -> float:
    """Convert speed in km/h to angle (to horizon) in a square image

    Args:
        speed_kmh (float): speed in km/h

    Returns:
        float: angle with horizontal in degree
    """
    DISTANCE_M = 800
    speed_ms = speed_kmh * 1000 / 3600
    time_s = DISTANCE_M / speed_ms
    angle_radian = np.arctan(time_s / duration_s)
    angle_degree = math.degrees(angle_radian)
    return angle_degree


def time_to_speed(time_s: float) -> float:
    """Convert time to finish the distance to speed in km/h

    Args:
        time_s (float): Time to finish the length, in seconds

    Returns:
        float: speed in km/h
    """
    DISTANCE_M = 800
    speed_ms = DISTANCE_M / time_s
    speed_kmh = speed_ms / 1000 * 3600
    return speed_kmh


def pca(cov_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate eigenvalues and eigenvectors (PCA) from covariance matrix

    Args:
        cov_mat (np.ndarray): Covariance.

    Returns:
        tuple[np.ndarray, np.ndarray]: eigenvalues, eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    return eigenvalues, eigenvectors


def inverse_pca(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray
) -> np.ndarray:
    """Calculate covariance matrix from eigenvalues and eigenvectors.

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


def cal_cov_mat(s1: float, s2: float, std_space: float) -> np.matrix:
    """Calculate covariance matrix from speed limits and space standard
        deviation.

    Args:
        s1 (float): Low speed limit
        s2 (float): High speed limit
        std_space (float): Space standard deviation

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
    cov_mat = inverse_pca(
        eigenvalues=[eigenvalue_1, eigenvalue_2],
        eigenvectors=np.array([eigenvector_1, eigenvector_2]).T
    )
    return cov_mat


def plot_factor(cov_mat: np.ndarray, mean: list = [0, 0]) -> None:
    """Heatmap

    Args:
        cov_mat (np.ndarray): Covariance matrix
        mean (list, optional): Mean. Defaults to [0, 0].
    """
    T = 60
    D = 800
    FACTOR = D / T
    plt.figure(figsize=(4, 4))
    data = np.random.multivariate_normal(mean, cov_mat, 1000)
    data = pd.DataFrame(data, columns=['Space', 'Time'])
    sns.kdeplot(data=data, x='Space', y='Time', fill=True)
    xlim = np.max(data['Space'])
    ylim = np.max(data['Time']) * FACTOR
    lim = np.max([xlim, ylim])
    plt.xlim(-lim, lim)
    plt.ylim(-lim / FACTOR, lim / FACTOR)
    degree = 45
    radian = degree / 180 * np.pi
    tan = np.tan(radian)
    plt.axline((0, 0), (1, tan / FACTOR))


def create_gauss_filter(
    cov_mat: np.ndarray,
    sampling_rate: int = 1000
) -> np.ndarray:
    """Create Gaussian filter from covariance matrix

    Args:
        cov_mat (np.ndarray): Covariance matrix.
        sampling_rate (int, optional): Sampling rate. Defaults to 1000.

    Returns:
        np.ndarray: Gaussian filter
    """
    # grid of points
    size_s = np.sqrt(cov_mat[0][0]) * 2 * 2  # 2 std
    size_t = np.sqrt(cov_mat[1][1]) * 2 * 2 * sampling_rate  # 2 std
    size_s = int(round(size_s, 0))
    size_t = int(round(size_t, 0))
    range_t = np.arange(-size_t // 2, size_t // 2 + 1) / sampling_rate
    range_s = np.arange(-size_s // 2, size_s // 2 + 1)
    x, y = np.meshgrid(range_s, range_t)
    # Calculate the Gaussian values
    det_cov_mat = np.linalg.det(cov_mat)  # determinant of covariance matrix
    inv_cov_mat = np.linalg.inv(cov_mat)  # inverse covariance matrix
    exponent = -0.5 * (
        inv_cov_mat[0, 0] * x**2 +
        2 * inv_cov_mat[0, 1] * x * y +
        inv_cov_mat[1, 1] * y**2
    )
    gauss_filter = np.exp(exponent) / (2 * np.pi * np.sqrt(det_cov_mat))
    # Normalize the filter
    gauss_filter /= np.sum(gauss_filter)
    return gauss_filter


def extend_filter(
    gauss_filter: np.ndarray,
    space: int = 800,
    time: int = 60,
    sampling_rate: int = 1000
) -> np.ndarray:
    """Extend (modify) the input filter to obtain designed shape ratio (by
        adding 0s). Ratio is sampling_rate * time / space

    Args:
        gauss_filter (np.ndarray): Input filter.
        space (int, optional): Length along space (in meters or channels).
            Defaults to 800.
        time (int, optional): Length along time (in seconds). Defaults to 60.
        sampling_rate (int, optional): Sampling rate. Defaults to 1000.

    Returns:
        np.ndarray: output filter
    """
    ratio = sampling_rate * time / space
    size_t = gauss_filter.shape[0]
    size_s = gauss_filter.shape[1]
    if size_t / size_s < ratio:
        size_t_add = round(size_s * ratio) - size_t
        shape_add = (int(round(size_t_add / 2)), gauss_filter.shape[1])
        # Create lines with the same shape as the existing array
        lines_add = np.zeros(shape_add)
        # Add lines to the top and bottom
        filter_extended = np.vstack((lines_add, gauss_filter, lines_add))
    else:
        size_s_add = round(size_t / ratio) - size_s
        shape_add = (gauss_filter.shape[0], int(round(size_s_add / 2)))
        # Create columns with zeros to add to the left and right
        columns_add = np.zeros(shape_add)
        # Add columns to the left and right
        filter_extended = np.hstack((columns_add, gauss_filter, columns_add))
    return filter_extended


def heatmap_filter(filter_arr: np.ndarray, sampling_rate: int = 1000) -> None:
    """Visualize filter in heatmap

    Args:
        filter_arr (np.ndarray): Filter array
        sampling_rate (int, optional): Sampling rate. Defaults to 1000.
    """
    percentile = np.quantile(np.abs(filter_arr), 0.95)
    vmin = 0
    vmax = percentile
    norm = colors.TwoSlopeNorm(
        vmin=vmin,
        vmax=vmax,
        vcenter=(vmin + vmax) / 2
    )
    cmap = 'Blues'
    plt.imshow(
        X=filter_arr,
        aspect=filter_arr.shape[1] / filter_arr.shape[0],  # square
        cmap=cmap,
        norm=norm,
        interpolation='none',  # no interpolation
        origin='lower'
    )
    plt.xlabel('Channels')
    plt.ylabel('Seconds')
    # adjust the y-axis to time
    ny = filter_arr.shape[0]
    no_labels = int(np.ceil(ny / sampling_rate))  # number of labels on axis y
    step_y = sampling_rate  # step between consecutive labels
    y_positions = np.arange(0, ny, step_y)  # pixel count at label position
    y_labels = range(no_labels)  # labels you want
    plt.yticks(y_positions, y_labels)


def das_distance(
    x: np.ndarray,
    y: np.ndarray,
    sampling_rate: float
) -> float:
    """Calculate the distance in DAS data. Assume 90km/h (25m/s) makes a 45
        degrees line when plotting.

    Args:
        x (np.ndarray): E.g. array([a, b]) where a is in time, b is in space
        y (np.ndarray): E.g. array([a, b]) where a is in time, b is in space
        sampling_rate (float): How many data were sampling per second.

    Returns:
        float: Distance in DAS data.
    """
    SPEED_KMH = 90  # 25m/s: 25meter ~ 1second
    speed_ms = SPEED_KMH / 3.6  # 25m/s: 25meter ~ 1second
    x = x * np.array([speed_ms / sampling_rate, 1])
    y = y * np.array([speed_ms / sampling_rate, 1])
    distance = np.sqrt(np.sum((x - y)**2))
    return distance


def split_period(
        period: tuple[Union[int, str], Union[int, str]],
        time_span: int = 10,
        date: str = '20230628'
) -> list[tuple[datetime, datetime]]:
    """Split a period into many smaller periods.

    Args:
        periods (tuple[Union[int, str], Union[int, str]]: (start, end) of
            period. Format '%H%M%S'.
        time_span (int, optional): Period duration in seconds to split.
            Defaults to 10.
        date (str, optional): Date of the data. Format '%Y%m%d'. Defaults to
            '20230628'.

    Returns:
        list[tuple[datetime, datetime]]: List of all smaller periods.
    """
    # Convert start, end to datetime
    ###########################################################################
    start = period[0]
    end = period[1]
    if start > end:
        raise ValueError('start must be smaller or equal to end.')
    start = datetime.strptime(f'{date} {start}', '%Y%m%d %H%M%S')
    end = datetime.strptime(f'{date} {end}', '%Y%m%d %H%M%S')
    # Split the duration into many smaller durations
    ###########################################################################
    duration = (end - start).seconds
    number_split = duration // time_span
    remaining = duration % time_span
    duration_split = [time_span] * number_split + [remaining]
    # Identify the start, end of each duration
    ###########################################################################
    periods = []
    for i in duration_split:
        end = start + timedelta(seconds=i)
        period_i = (start, end)
        periods.append(period_i)
        start = end
    return periods


def split_periods(
        periods: list[tuple[datetime, datetime]],
        time_span: int = 10,
        date: str = '20230628'
) -> list[tuple[datetime, datetime]]:
    """Split periods into many smaller periods.

    Args:
        period (list[tuple[datetime, datetime]]): (start, end) of periods.
            Format '%H%M%S'.
        time_span (int, optional): Period duration in seconds to split.
            Defaults to 10.
        date (str, optional): Date of the data. Format '%Y%m%d'. Defaults to
            '20230628'.

    Returns:
        list[tuple[datetime, datetime]]: List of all smaller periods.
    """
    periods_split = []
    for period in periods:
        periods_i = split_period(
            period=period,
            time_span=time_span,
            date=date
        )
        periods_split.extend(periods_i)
    return periods_split


def files_in_folder(folder_path: str) -> list[str]:
    """List all files in a folder

    Args:
        folder_path (str): Folder path

    Returns:
        list[str]: all files
    """
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


# helper function
#######################################################################
def find_divisors(number: int) -> list[int]:
    """List all divisors of an integer number.

    Args:
        number (int): Input number.

    Returns:
        list[int]: List of divisors of input number.
    """
    divisors = []
    for i in range(1, number + 1):
        if number % i == 0:
            divisors.append(i)
    return divisors


# helper function
#######################################################################
def largest_smaller_than_threshold(
    lst: list[float], threshold: float
) -> Union[float, None]:
    """Find the largest number in a list that is smaller than or equal
    to a threshold.

    Args:
        lst (list[float]): List of number
        threshold (float): Threshold.

    Returns:
        list[float]: List of number.
    """
    filtered_values = [x for x in lst if x <= threshold]
    if filtered_values:
        return max(filtered_values)
    else:
        return None  # If there are no values smaller than threshold