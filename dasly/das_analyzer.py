"""Provides convenient methods to advanced analyze DAS data."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-07-01'

import math
from typing import Literal
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

    def hough_transform(
        self,
        target_speed: float,
        speed_res: float,
        length_meters: float = None,
        length_seconds: float = None,
        speed_unit: Literal['km/h', 'm/s'] = 'km/h',  # 'km/h' or 'm/s',
        threshold_percent: float = 0.3,  # needs to cover 30% of the length
        min_line_length_percent: float = 1,  # needs to equal to the length
        max_line_gap_percent: float = 0.4,  # mustn't interupt > 40% the length
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

        Returns:
            Union[None, pd.DataFrame]: pd.DataFrame if inplace=False.
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
        if lines is not None:
            logger.info(f'{len(lines):,} lines are detected.')
            # self.lines = lines
            # note that the default shape of output lines is (n, 1, m), where n
            # is the number of lines and m is the coordindates of the lines.
            # The additional dimension in the middle is designed to maintain a
            # consistent multi-dimensional structure, providing flexibility and
            # compatibility with other OpenCV functions.
            # For easier manipulation, we can remove the middle dimension by
            # using np.squeeze() function.
            self.lines = np.squeeze(lines, axis=1)
        else:
            logger.info('No lines are detected.')
            self.lines = lines

    @staticmethod
    def _distance_to_horizontal(
        lines: np.ndarray,
        x_coords: np.ndarray,
    ) -> np.ndarray:
        """Calculate distances (in pixels) from lines to x-coordinates.

        Args:
            lines (np.ndarray): Lines in the format of (x1, y1, x2, y2).
            x_coords (np.ndarray): X-coordinates to calculate distances.

        Returns:
            np.ndarray: Distances from lines to x-coordinates
        """
        lines = np.squeeze(lines)  # remove the middle dimension if any
        x1 = lines[:, 0]
        y1 = lines[:, 1]
        x2 = lines[:, 2]
        y2 = lines[:, 3]

        # Initialize distances array
        dists = np.full((len(lines), len(x_coords)), np.nan)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate slopes and intercepts
            slopes = (y2 - y1) / (x2 - x1)
            intercepts = y1 - slopes * x1

        # Handle vertical lines separately
        vert_mask = (x1 == x2)
        non_vert_mask = ~vert_mask

        # Calculate y-values for all x-coordinates for non-vertical lines
        y_vals = (
            np.outer(slopes[non_vert_mask], x_coords)
            + intercepts[non_vert_mask][:, None]
        )

        # Assign the distances (absolute y-vals) for non-vertical lines
        dists[non_vert_mask] = y_vals

        # For vertical lines, directly assign the y1 value if x == x1
        vert_dists = np.where(x_coords == x1[:, None], y1[:, None], np.nan)
        dists[vert_mask] = vert_dists[vert_mask]

        # Mask distances for x-coordinates outside the range [x1, x2]
        mask = (x_coords < x1[:, None]) | (x_coords > x2[:, None])

        # Assign an arbitrary number (max int32) to distances outside the range
        # should not be NAN because DBSCAN cannot handle NAN
        dists[mask] = np.iinfo(np.int32).max

        return dists

    @staticmethod
    def _distance_to_vertial(
        lines: np.ndarray,
        y_coords: np.ndarray
    ) -> np.ndarray:
        """Calculate distances (in pixels) from lines to y-coordinates.

        Args:
            lines (np.ndarray): Lines in the format of (x1, y1, x2, y2).
            y_coords (np.ndarray): Y-coordinates to calculate distances.

        Returns:
            np.ndarray: Distances from lines to y-coordinates
        """
        lines = np.squeeze(lines)  # remove the middle dimension if any
        x1 = lines[:, 0]
        y1 = lines[:, 1]
        x2 = lines[:, 2]
        y2 = lines[:, 3]

        # Initialize distances array
        dists = np.full((len(lines), len(y_coords)), np.nan)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate slopes and intercepts
            slopes = (y2 - y1) / (x2 - x1)
            intercepts = y1 - slopes * x1

        # Handle horizontal lines separately
        horiz_mask = (y1 == y2)
        non_horiz_mask = ~horiz_mask

        # Calculate x-values for all y-coordinates for non-horizontal lines
        x_vals = (
            (y_coords - intercepts[non_horiz_mask][:, None])
            / slopes[non_horiz_mask][:, None]
        )

        # Assign the distances (absolute x-vals) for non-horizontal lines
        dists[non_horiz_mask] = x_vals

        # For horizontal lines, directly assign the x1 value if y == y1
        horiz_distances = np.where(
            y_coords == y1[:, None], np.abs(x1[:, None]), np.nan)
        dists[horiz_mask] = horiz_distances[horiz_mask]

        # Mask distances for y-coordinates outside the range [y1, y2]
        mask = (y_coords < y1[:, None]) | (y_coords > y2[:, None])

        # Assign an arbitrary number (max int32) to distances outside the range
        # should not be NAN because DBSCAN cannot handle NAN
        dists[mask] = np.iinfo(np.int32).max

        return dists

    @staticmethod
    def _metric(
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Calculate the distance between 2 points in a spatial-temporal data.
        """
        # 1. The fundamental idea is to calculate the distance between 2 points
        # in a sptial-temporal data. It's not as obvious as the distance btw 2
        # points in an ordinary 2D space (like an image). Because the unit in
        # spatial dimension is different from the unit in temporal dimension.
        # As a result, we must set the relative ratio between the spatial and
        # temporal distance. So we assume the desired speed (of the interested
        # signal) creates a 45-degree diagonal line in the spatial-temporal
        # data. Thus we can calculate the ratio.

        # 2. Imagine 2 parallel lines a and b. If a and b are long, the dist.
        # between them should be small (they are more similar). If a and b are
        # short, the distance between them should be large (they are more
        # distinct). So, if we take the sum of the distance, we will get the
        # opposite result (more distance for long lines). If we take the mean,
        # the result will, on the other hand, too harmonized, i.e. long lines
        # and short lines have the same distance. Therefore, we take the mean
        # and devided by the number of non-nan values.

        # replace the arbitrary number max int32 with nan
        arb_num = np.iinfo(np.int32).max
        x = np.where(x == arb_num, np.nan, x)
        y = np.where(y == arb_num, np.nan, y)

        abs_dist = np.abs(np.array(x) - np.array(y))
        sum_dist = np.nansum(abs_dist)  # sum of non-nan values
        if sum_dist > 0:
            # mean of non-nan values
            return sum_dist / np.sum(~np.isnan(abs_dist))
        else:
            # if all values are nan, return an arbitrary number
            return arb_num

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
            .to_numpy()
        )
        return lines_agg

    @staticmethod
    def lines_inter_edges(
        lines: np.ndarray,
        width: int,
        height: int
    ) -> pd.DataFrame:
        """"Calculate intersections of lines with all edges of the image.
        """
        lines = np.array(lines).reshape(-1, 4)
        x1 = lines[:, 0]
        y1 = lines[:, 1]
        x2 = lines[:, 2]
        y2 = lines[:, 3]

        # Calculate slopes (m) and intercepts (c)
        m = np.divide(
            (y2 - y1),
            (x2 - x1),
            out=np.zeros_like(y1, dtype=float),
            where=(x2 - x1) != 0
        )
        c = y1 - m * x1

        # Calculate intersection points with all edges
        y_left = c
        y_right = m * width + c
        x_top = -c / m
        x_bottom = (height - c) / m

        # Stack and filter valid intersections within image boundaries
        left_edges = np.stack([np.zeros_like(y_left), y_left], axis=-1)
        right_edges = np.stack(
            [np.full_like(y_right, width), y_right], axis=-1)
        top_edges = np.stack([x_top, np.zeros_like(x_top)], axis=-1)
        bottom_edges = np.stack(
            [x_bottom, np.full_like(x_bottom, height)], axis=-1)

        valid_left = (0 <= y_left) & (y_left <= height)
        valid_right = (0 <= y_right) & (y_right <= height)
        valid_top = (0 <= x_top) & (x_top <= width)
        valid_bottom = (0 <= x_bottom) & (x_bottom <= width)

        # Collect intersections in the same shape as the input lines
        intersections = np.zeros((lines.shape[0], 4, 2), dtype=int)
        intersections[:] = -1  # Initialize with -1 (invalid marker)

        intersections[valid_left, 0] = left_edges[valid_left]
        intersections[valid_right, 1] = right_edges[valid_right]
        intersections[valid_top, 2] = top_edges[valid_top]
        intersections[valid_bottom, 3] = bottom_edges[valid_bottom]

        return intersections

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
        if self.lines is None:
            return  # no lines to cluster
        if (eps_meters is None) + (eps_seconds is None) != 1:
            raise ValueError('eps_meters or eps_seconds must be provided.')
        if eps_meters:
            eps = eps_meters * self.s_rate
            lines = DASAnalyzer._distance_to_vertial(
                self.lines,
                np.arange(self.signal.shape[0])
            )

        else:
            eps = eps_seconds * self.t_rate
            lines = DASAnalyzer._distance_to_horizontal(
                self.lines,
                np.arange(self.signal.shape[1])
            )

        # apply DBSCAN
        clustering = DBSCAN(
            eps=eps,
            min_samples=1,  # each line can be a cluster on its own
            metric=DASAnalyzer._metric
        ).fit(lines)

        # replace labels -1 with increasing values
        cluster = DASAnalyzer._replace_neg_ones(clustering.labels_)
        self.cluster = cluster

        # aggregate the lines by the cluster
        # lines_agg = DASAnalyzer._lines_agg(self.lines, cluster)
        # self.lines = lines_agg

    def plot_cluster(self) -> None:
        """Plot the clustered lines."""
        pass
