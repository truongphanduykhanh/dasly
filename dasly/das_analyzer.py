"""Provides convenient methods to advanced analyze DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-07-01'

import math
from typing import Literal, Union
import logging
import logging.config

import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import DBSCAN

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DASAnalyzer:
    """Analyze DAS data."""

    def __init__(self):
        self.signal: pd.DataFrame = None
        self.t_rate: float = None
        self.s_rate: float = None

    def _hough_transform_theta(
        self,
        target_speed: float,
        speed_res: float,
        unit: Literal['km/h', 'm/s'] = 'km/h',  # 'km/h' or 'm/s'
    ) -> float:
        """Calculate theta for Hough transform.

        Theta is the angle resolution of the accumulator in radians. Increasing
        theta might speed up the computation but can make it less accurate, and
        vice versa. The suitable theta can be infered back based on the desired
        speed resolution. Important: note that this is a heuristic approach,
        because when combining theta and rho, the actual speed resolution will
        much finner than the desired speed resolution.

        Args:
            target_speed (float): Desired speed to be tracked.
            speed_res (float): Desired speed resolution.
            unit (Literal['km/h', 'm/s'], optional): Unit of speed. Defaults
                to 'km/h'.

        Returns:
            float: Theta for Hough transform
        """
        target_speed = np.abs(target_speed)
        if unit == 'km/h':
            target_speed = target_speed / 3.6
            speed_res = speed_res / 3.6
        angle1 = math.atan((1 / target_speed) * (self.t_rate / self.s_rate))
        angle2 = math.atan(
            (1 / (target_speed + speed_res)) * (self.t_rate / self.s_rate))
        theta = np.abs(angle1 - angle2)
        return theta

    def _hough_transform_lengh(
        self,
        target_speed: float,
        length_meters: float = None,
        length_seconds: float = None,
        unit: Literal['km/h', 'm/s'] = 'km/h',  # 'km/h' or 'm/s'
    ) -> float:
        """Calculate length in pixel (for Hough transform) of the line to be
        detected, from the length in either meters or seconds of the signal.
        Note that the length is calculated based on the Pythagorean theorem, in
        which assume that spatial dimension and temporal dimension are equally.
        This is a heuristic approach and might not be accurate in all cases.
        For example, if we want to detect a signal having length of 100 meters,
        this function will output length in pixel. But the Hough transform
        might detect much shoter signal (in meters) if the signal is more
        vertial and the temporal sampling rate is high.

        Args:
            target_speed (float): Desired speed to be tracked.
            length_meters (float, optional): Typical length of the signal to be
                tracked in meters. Defaults to None.
            length_seconds (float, optional): Typical length of the signal to
                be tracked in seconds. Defaults to None.
            unit (Literal['km/h', 'm/s'], optional): Unit of speed. Defaults to
                'km/h'.

        Returns:
            float: Length for Hough transform (in pixel).
        """
        if (
            (length_meters is None and length_seconds is None) or
            (length_meters is not None and length_seconds is not None)
        ):
            raise ValueError('Either meters or seconds must be provided.')
        if unit == 'km/h':
            target_speed = target_speed / 3.6
        if length_meters:
            length_seconds = length_meters / target_speed
        else:
            length_meters = target_speed * length_seconds
        length = np.sqrt(
            (length_meters * self.s_rate) ** 2 +
            (length_seconds * self.t_rate) ** 2
        )
        return length

    @staticmethod
    def lines_intersect_edges(
        lines: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """"Calculate intersections of lines (in pixel) with all edges
        boundaries of the image, and intersections with extended y-axis (to
        calculate the time of the signal to reach the edge).
        """
        width = width - 1  # zero-based index
        height = height - 1  # zero-based index

        x1 = lines[:, 0]
        y1 = lines[:, 1]
        x2 = lines[:, 2]
        y2 = lines[:, 3]

        # Suppress divide by zero warning, multiply inf and nan values by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # Formular of a straight line: y = mx + b
            # np.array([0, 0, 1, -1]) /  np.array([1, -1, 0, 0])
            # -> [0, -0, inf, -inf]
            m = (y2 - y1) / (x2 - x1)
            # np.array([0, 0, 0]) *  np.array([np.nan, np.inf, -np.inf]) ->
            # [nan, nan, nan]
            # np.array([1, 1, 1]) *  np.array([np.nan, np.inf, -np.inf]) ->
            # [ nan,  inf, -inf]
            # np.array([-1, -1, -1]) * v np.array([np.nan, np.inf, -np.inf]) ->
            # [ nan, -inf,  inf]
            b = y1 - m * x1

            # Calculate the intersection of the lines with the (extended) edges
            y0 = m * 0 + b  # when x = 0
            y_width = m * width + b  # when x = width
            x0 = (0 - b) / m  # when y = 0
            x_height = (height - b) / m  # when y = height

        # Clipping values to image boundaries if the intersection is outside
        # and different from infinity. Clip NA will remain NA.
        y0_edge = np.where(np.isinf(y0), y0, np.clip(y0, 0, height))
        y_width_edge = np.where(
            np.isinf(y_width), y_width, np.clip(y_width, 0, height))
        x0_edge = np.where(np.isinf(x0), x0, np.clip(x0, 0, width))
        x_height_edge = np.where(
            np.isinf(x_height), x_height, np.clip(x_height, 0, width))

        # Create boolean masks
        mask_ge_0 = m >= 0
        mask_lt_0 = m < 0

        # Initialize the result array [x1, y1, x2, y2], where (x1, y1) is the
        # intersection with the lower edges; (x2, y2) is the intersection with
        # the upper edges,

        intersect_edges = np.zeros((len(m), 4))

        # Fill the result array based on the conditions
        intersect_edges[mask_ge_0] = np.stack((
            x0_edge[mask_ge_0],  # always fixed because y=0
            y0_edge[mask_ge_0],  # when m>=0, it will intersect left edge first
            x_height_edge[mask_ge_0],  # always fixed because y=height
            y_width_edge[mask_ge_0],
        ), axis=1)
        intersect_edges[mask_lt_0] = np.stack((
            x0_edge[mask_lt_0],  # always fixed because y=0
            y_width_edge[mask_lt_0],  # m<0, it intersects the right edge first
            x_height_edge[mask_lt_0],  # always fixed because y=height
            y0_edge[mask_lt_0]
        ), axis=1)

        intersect_edges_ext = np.stack((
            y0,  # intersection with the extended left edge
            y_width  # intersection with the extended right edge
        ), axis=1)

        intersect = np.hstack([intersect_edges, intersect_edges_ext])

        return intersect

    def _lines_df(self):
        """Create a DataFrame of lines, convert the pixel intersections to
        space and time convinient information.
        """
        # Ordered columns to make it more readable and convenient
        ordered_columns = [
            'speed_kmh', 'speed_ms', 's', 't',
            's1', 't1', 's2', 't2',
            's1_edge', 't1_edge', 's2_edge', 't2_edge',
            't1_edge_ext', 't2_edge_ext',
            'x1', 'y1', 'x2', 'y2',
            'x1_edge', 'y1_edge', 'x2_edge', 'y2_edge',
            'y1_edge_ext', 'y2_edge_ext',
        ]
        if len(self.lines) == 0:  # no lines detected
            self.lines_df = pd.DataFrame(columns=ordered_columns)
            return  # exit the function from here

        intersect = DASAnalyzer.lines_intersect_edges(
            self.lines,
            self.signal.shape[1],
            self.signal.shape[0]
        )

        lines_df = np.hstack([self.lines, intersect])
        columns = [
            'x1', 'y1', 'x2', 'y2',
            'x1_edge', 'y1_edge', 'x2_edge', 'y2_edge',
            'y1_edge_ext', 'y2_edge_ext'
        ]
        lines_df = pd.DataFrame(lines_df, columns=columns)

        # Define helper functions
        def get_signal_column(index):
            return index / self.s_rate + self.signal.columns[0]

        def get_signal_index(index):
            return (pd.to_timedelta(index / self.t_rate, unit='s')
                    + self.signal.index[0])

        # Assign new columns
        lines_df = lines_df.assign(
            s1=lambda df: df['x1'].map(get_signal_column),
            t1=lambda df: df['y1'].map(get_signal_index),
            s2=lambda df: df['x2'].map(get_signal_column),
            t2=lambda df: df['y2'].map(get_signal_index),
            s1_edge=lambda df: df['x1_edge'].map(get_signal_column),
            t1_edge=lambda df: df['y1_edge'].map(get_signal_index),
            s2_edge=lambda df: df['x2_edge'].map(get_signal_column),
            t2_edge=lambda df: df['y2_edge'].map(get_signal_index),
            t1_edge_ext=lambda df: df['y1_edge_ext'].map(get_signal_index),
            t2_edge_ext=lambda df: df['y2_edge_ext'].map(get_signal_index),
            s=lambda df: df['s2'] - df['s1'],
            t=lambda df: (df['t2'] - df['t1']).dt.total_seconds(),
            speed_ms=lambda df: df['s'] / df['t'],
            speed_kmh=lambda df: df['speed_ms'] * 3.6,
        )

        # Reorder columns to make it more readable and convenient
        ordered_columns = [
            'speed_kmh', 'speed_ms', 's', 't',
            's1', 't1', 's2', 't2',
            's1_edge', 't1_edge', 's2_edge', 't2_edge',
            't1_edge_ext', 't2_edge_ext',
            'x1', 'y1', 'x2', 'y2',
            'x1_edge', 'y1_edge', 'x2_edge', 'y2_edge',
            'y1_edge_ext', 'y2_edge_ext',
        ]
        lines_df = lines_df[ordered_columns]
        self.lines_df = lines_df

    @staticmethod
    def reorder_coordinates(lines: np.ndarray) -> np.ndarray:
        """Reorder the 2 endpoints of lines segments so that y1 <= y2.
        """
        lines_cp = lines.copy()
        # Extract the coordinates
        x1 = lines_cp[:, 0]
        y1 = lines_cp[:, 1]
        x2 = lines_cp[:, 2]
        y2 = lines_cp[:, 3]

        # Create a mask where y1 > y2
        mask = y1 > y2

        # Swap coordinates where the mask is True
        x1[mask], x2[mask] = x2[mask], x1[mask]
        y1[mask], y2[mask] = y2[mask], y1[mask]

        # Combine the coordinates back into the lines array
        lines_cp = np.stack((x1, y1, x2, y2), axis=1)
        return lines_cp

    def hough_transform(
        self,
        target_speed: float,
        speed_res: float,
        length_meters: float = None,
        length_seconds: float = None,
        speed_unit: Literal['km/h', 'm/s'] = 'km/h',  # 'km/h' or 'm/s',
        threshold_percent: float = 0.6,  # needs to cover 60% of the length
        min_line_length_percent: float = 1,  # needs to equal to the length
        max_line_gap_percent: float = 0.2,  # mustn't interupt > 20% the length
    ) -> None:
        """Apply Hough transform to detect lines in the data.

        Args:
            target_speed (float): Desired speed to be tracked.
            speed_res (float): Desired speed resolution.
            length_meters (float, optional): Typical length of the signal to be
                tracked in meters. Defaults to None.
            length_seconds (float, optional): Typical length of the signal to
                be tracked in seconds. Defaults to None.
            speed_unit (Literal['km/h', 'm/s'], optional): Unit of speed.
                Defaults to 'km/h'.
            threshold_percent (float, optional): Accumulator threshold factor.
                Defaults to 0.5.
            min_line_length_percent (float, optional): Minimum line length
                factor. Defaults to 1.
            max_line_gap_percent (float, optional): Maximum line gap factor.
                Defaults to 0.2.

        """
        theta = self._hough_transform_theta(
            target_speed, speed_res, speed_unit)
        length = self._hough_transform_lengh(
            target_speed, length_meters, length_seconds, speed_unit)
        lines = cv2.HoughLinesP(
            self.signal.values,
            rho=1,
            theta=theta,
            threshold=int(threshold_percent * length),
            minLineLength=min_line_length_percent * length,
            maxLineGap=max_line_gap_percent * length
        )
        if lines is None:
            lines = np.empty((0, 4))  # empty array
        else:  # lines are detected
            # note that the default shape of output lines is (N, 1, 4), where n
            # is the number of lines. The additional dimension in the middle is
            # designed to maintain a consistent multi-dimensional structure,
            # providing compatibility with other OpenCV functions. In this
            # project's context, we only need the 4 coordinates of each line,
            # so we can remove the middle dimension by np.squeeze() function.
            # lines = np.squeeze(lines, axis=1)
            # but this is not good for the case of 1 line detected, in which
            # they will be squeezed to 1D array. So we use reshape instead.
            lines = lines.reshape(-1, 4)
            lines = DASAnalyzer.reorder_coordinates(lines)

        self.lines = lines
        self._lines_df()
        logger.info(f'{len(self.lines):,} lines are detected.')

    @staticmethod
    def _y_vals_lines(
        lines: np.ndarray,
        x_coords: np.ndarray,
    ) -> np.ndarray:
        """Calculate y values of lines at x-coordinates, which is the distances
        (in pixels) from lines to x-coordinates.

        Args:
            lines (np.ndarray): Lines in the format of (x1, y1, x2, y2).
            x_coords (np.ndarray): X-coordinates to calculate y values.

        Returns:
            np.ndarray: Y values (distance) from lines to x-coordinates
        """
        x1 = lines[:, 0]
        y1 = lines[:, 1]
        x2 = lines[:, 2]
        y2 = lines[:, 3]

        # Suppress divide by zero warning, multiply inf and nan values by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # Formular of a straight line: y = mx + b
            # Calculate the slope (m) and intercept (b)
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # Calculate y-values at all x-coordinates
            m_reshaped = m.reshape(-1, 1)
            x_coords_reshaped = x_coords.reshape(1, -1)
            b_reshaped = b.reshape(-1, 1)
            y_vals = m_reshaped * x_coords_reshaped + b_reshaped

        # Assign an arbitrary number (max int32) to y values outside the range
        # should not be NAN because DBSCAN cannot handle NAN
        # Create a 2D array of column indices
        col_idx = np.arange(y_vals.shape[1])
        # Create the mask using broadcasting
        left_lim = np.min([x1, x2], axis=0)
        right_lim = np.max([x1, x2], axis=0)
        mask = (
            (col_idx >= left_lim[:, np.newaxis]) &
            (col_idx <= right_lim[:, np.newaxis]))
        # Apply the mask to replace values outside the limits with the number
        y_vals_mask = np.where(mask, y_vals, np.iinfo(np.int32).max)
        y_vals_mask[np.isnan(y_vals_mask)] = np.iinfo(np.int32).max
        return y_vals_mask

    @staticmethod
    def _x_vals_lines(
        lines: np.ndarray,
        y_coords: np.ndarray
    ) -> np.ndarray:
        """Calculate x values of lines at y-coordinates, which is the distances
        (in pixels) from lines to y-coordinates.

        Args:
            lines (np.ndarray): Lines in the format of (x1, y1, x2, y2).
            y_coords (np.ndarray): Y-coordinates to calculate x values.

        Returns:
            np.ndarray: X values (distance) from lines to y-coordinates
        """
        x1 = lines[:, 0]
        y1 = lines[:, 1]
        x2 = lines[:, 2]
        y2 = lines[:, 3]

        # Suppress divide by zero warning, multiply inf and nan values by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # Formular of a straight line: y = mx + b
            # Calculate the slope (m) and intercept (b)
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # Calculate y-values at all x-coordinates
            m_reshaped = m.reshape(-1, 1)
            y_coords_reshaped = y_coords.reshape(1, -1)
            b_reshaped = b.reshape(-1, 1)
            x_vals = (y_coords_reshaped - b_reshaped) / m_reshaped

        # Assign an arbitrary number (max int32) to y values outside the range
        # should not be NAN because DBSCAN cannot handle NAN
        # Create a 2D array of column indices
        row_idx = np.arange(x_vals.shape[0]).reshape(-1, 1)
        # Create the mask using broadcasting
        bottom_lim = np.min([y1, y2], axis=0)
        top_lim = np.max([y1, y2], axis=0)
        mask = (
            (y_coords_reshaped >= bottom_lim[row_idx]) &
            (y_coords_reshaped <= top_lim[row_idx]))
        # Apply the mask to replace values outside the limits with the number
        x_vals_mask = np.where(mask, x_vals, np.iinfo(np.int32).max)
        x_vals_mask[np.isnan(x_vals_mask)] = np.iinfo(np.int32).max
        return x_vals_mask

    @staticmethod
    def _metric(x: np.ndarray, y: np.ndarray) -> Union[np.ndarray, float]:
        """Calculate the distance between lines in spatial-temporal data.
        It is the average distance by seconds or average distance by meters.

        Args:
            x (np.ndarray): Array of x values of shape (m, n) or (, n).
            y (np.ndarray): Array of y values of shape (p, n) or (, n).

        Returns:
            Union[np.ndarray, float]: Distance between lines.
        """

        # Replace the arbitrary number max int32 with nan
        ARB_NUM = np.iinfo(np.int32).max
        x = np.where(x == ARB_NUM, np.nan, x)
        y = np.where(y == ARB_NUM, np.nan, y)

        # Check if x and y are one-dimensional
        x_is_1d = x.ndim == 1
        y_is_1d = y.ndim == 1

        # If both x and y are one-dimensional, reshape them to (1, d)
        if x_is_1d and y_is_1d:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)

        # Expand dimensions of x and y for broadcasting
        x_exp = np.expand_dims(x, axis=1)  # Shape (m, 1, n)
        y_exp = np.expand_dims(y, axis=0)  # Shape (1, p, n)

        # Calculate absolute distance and handle NaNs
        abs_dist = np.abs(x_exp - y_exp)  # Shape (m, p, n)
        # Sum of non-nan values along the last axis, shape (m, p)
        sum_dist = np.nansum(abs_dist, axis=-1)
        # Count of non-nan values along the last axis, shape (m, p)
        count_non_nan = np.sum(~np.isnan(abs_dist), axis=-1)
        # Calculate mean distance
        mean_dist = np.where(
            count_non_nan > 0, sum_dist / count_non_nan, ARB_NUM)

        if x_is_1d and y_is_1d:
            return mean_dist[0, 0]
        return mean_dist

    @staticmethod
    def _replace_neg_ones(arr: np.ndarray) -> np.ndarray:
        """Replace -1 values in the array with increasing values.

        This helper function is necessary because DBSCAN will always assign -1
        (noise) to a point that is far from any other point, regardless setting
        of min_samples=1. This function will convert every noise point to
        different clusters.
        """
        # Find the maximum value in the array
        max_val = np.max(arr)
        # Find indices where the value is -1
        indices = np.where(arr == -1)[0]
        # Create replacement values starting from max_val + 1
        replacements = np.arange(max_val + 1, max_val + 1 + len(indices))
        # Create a copy of the array to avoid modifying the original array
        new_arr = arr.copy()
        # Replace -1 values with the new replacement values
        new_arr[indices] = replacements
        return new_arr

    @staticmethod
    def _lines_agg(
        lines: np.ndarray,  # shape (N, 4)
        cluster: np.ndarray,
        func_name: str = 'mean'
    ) -> np.ndarray:
        """Aggregate the lines by the cluster.
        """
        lines_agg = (
            pd.DataFrame(lines)
            .assign(cluster=cluster)
            .groupby('cluster')
            .agg(func_name)
            .round(0)
            .astype(int)
            .to_numpy()
        )
        return lines_agg

    def dbscan(
        self,
        eps_meters: float = None,
        eps_seconds: float = None
    ) -> None:
        """Apply DBSCAN to cluster the lines.

        Args:
            eps_meters (float, optional): Epsilon in meters. Defaults to None.
            eps_seconds (float, optional): Epsilon in seconds. Defaults to
                None.
        """
        if len(self.lines) == 0:
            return  # no lines to cluster
        if (eps_meters is None) + (eps_seconds is None) != 1:
            raise ValueError('eps_meters or eps_seconds must be provided.')
        if eps_meters:
            eps = eps_meters * self.s_rate
            lines_distance_to_axis = DASAnalyzer._x_vals_lines(
                self.lines,
                np.arange(self.signal.shape[0])
            )

        else:
            eps = eps_seconds * self.t_rate
            lines_distance_to_axis = DASAnalyzer._y_vals_lines(
                self.lines,
                np.arange(self.signal.shape[1])
            )

        # apply DBSCAN
        clustering = DBSCAN(
            eps=eps,
            min_samples=1,  # each line can be a cluster on its own
            metric=DASAnalyzer._metric
        ).fit(lines_distance_to_axis)

        # replace labels -1 with increasing values
        cluster = DASAnalyzer._replace_neg_ones(clustering.labels_)
        self.cluster = cluster

        # aggregate the lines by the cluster
        lines_agg = DASAnalyzer._lines_agg(self.lines, cluster)
        self.lines = lines_agg
        self._lines_df()
