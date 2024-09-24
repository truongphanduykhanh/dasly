import datetime

import numpy as np
import pandas as pd
import h5py


class DASArray(np.ndarray):
    """Custom DAS class that inherits from np.ndarray.

    This class is used to store DAS data and associated attributes.
    """
    def __new__(
        cls,
        input_array: np.ndarray | list,
        dt: float,
        ds: float,
        dx: float,
        t_start: float,
        s_start: float,
        gauge_length: float
    ) -> 'DASArray':
        """Create a new instance of the class.

        Args:
            input_array (np.ndarray | list): Input array.
            dt (float): Time step.
            ds (float): Spatial channel step.
            dx (float): Distance (in meters) between channels.
            t_start (float): Start time.
            s_start (float): Start position.
            gauge_length (float): Gauge length.

        Returns:
            DASArray: New instance of the class.
        """
        # Convert input array to ndarray and view it as the DASArray
        obj = np.asarray(input_array).view(cls)

        # Add the new attributes
        obj.dt = dt
        obj.ds = ds
        obj.dx = dx
        obj.t_start = t_start
        obj.s_start = s_start
        obj.gauge_length = gauge_length

        # Return the object
        return obj

    def __array_finalize__(self, obj: np.ndarray) -> None:
        """Finalize the array. This method ensures that the new attributes are
        maintained when NumPy methods are used.

        Args:
            obj (np.ndarray): Input array.
        """
        if obj is None:
            return
        self.dt = getattr(obj, 'dt', None)
        self.ds = getattr(obj, 'ds', None)
        self.dx = getattr(obj, 'dx', None)
        self.t_start = getattr(obj, 't_start', None)
        self.s_start = getattr(obj, 's_start', None)
        self.gauge_length = getattr(obj, 'gauge_length', None)

    def to_df(self) -> pd.DataFrame:
        """Convert the DASArray to a pandas DataFrame.

        Returns:
            pd.DataFrame: DAS data as a DataFrame.
        """

        channels = np.arange(
            self.s_start,
            self.s_start + self.shape[1] * self.ds,
            self.ds
        )
        times = gen_datetime(
            start=self.t_start,
            n=self.shape[0],
            dt=self.dt
        )

        df = pd.DataFrame(
            data=self,
            index=times,
            columns=channels
        )

        return df

    def update_attr(
        self,
        dt: float = None,
        ds: float = None,
        dx: float = None,
        t_start: float = None,
        s_start: float = None,
        gauge_length: float = None
    ) -> None:
        """Updates the attributes.

        Args:
            dt (float): Time step.
            ds (float): Spatial channel step.
            dx (float): Distance (in meters) between channels.
            t_start (float): Start time.
            s_start (float): Start position.
            gauge_length (float): Gauge length.
        """
        if dt is not None:
            self.dt = dt
        if ds is not None:
            self.ds = ds
        if dx is not None:
            self.dx = dx
        if t_start is not None:
            self.t_start = t_start
        if s_start is not None:
            self.s_start = s_start
        if gauge_length is not None:
            self.gauge_length = gauge_length


def print_hdf5_info(file_path: str) -> None:
    """Print the keys and values of an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
    """
    # Open the HDF5 file (in read mode)
    with h5py.File(file_path, 'r') as hdf_file:

        # Function to recursively print groups, datasets, and their values
        def print_hdf5_structure(
            name: str,
            obj: h5py.Group | h5py.Dataset
        ) -> None:
            """Recursively print groups, datasets, and their values.

            Args:
                name (str): Name of the group or dataset.
                obj (h5py.Group | h5py.Dataset): Group or dataset object.
            """
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"\t Dataset: {name}, Value: {obj[()]}")

        # Visit all the items in the file and print structure and values
        hdf_file.visititems(print_hdf5_structure)


def load_hdf5(
    file_paths: list[str],
    s_start_idx: int = None,
    s_end_idx: int = None,
    s_step_idx: int = None,
    t_start_idx: int = None,
    t_end_idx: int = None,
    t_step_idx: int = None,
    integrate: bool = False
):
    """Load DAS data from HDF5 files.

    Args:
        file_paths (list[str]): List of paths to the HDF5 files.
        s_start_idx (int): Start index for spatial channels. Default is None.
        s_end_idx (int): End index for spatial channels. Default is None.
        s_step_idx (int): Step index for spatial channels. Default is None.
        t_start_idx (int): Start index for temporal. Default is None.
        t_end_idx (int): End index for temporal. Default is None.
        t_step_idx (int): Step index for temporal. Default is None.
        integrate (bool): Integrate the data to get strain. Default is False.

    Returns:
        DASArray: DAS data.
    """
    # Load data
    ###########################################################################
    data = []
    # Iterate over all hdf5 files
    for n, file_path in enumerate(file_paths):
        # For each file, load the data and append to the data list
        with h5py.File(file_path, 'r') as hdf_file:
            raw_data = hdf_file['data'][
                s_start_idx:s_end_idx:s_step_idx,
                t_start_idx:t_end_idx:t_step_idx
            ]
            data.append(raw_data)

            # Get header information from the first file
            if n == 0:
                data_scale = hdf_file['header/dataScale'][()]
                sensitivity = hdf_file['header/sensitivities'][()][0][0]
                dt = hdf_file['header/dt'][()]
                dx = hdf_file['header/dx'][()]
                ds = hdf_file['demodSpec/roiDec'][()][0]
                t_start = hdf_file['header/time'][()]
                gauge_length = hdf_file['header/gaugeLength'][()]
                channels = hdf_file['header/channels'][()]

    # Combine all data into a single numpy array
    data = np.concatenate(data, axis=0)

    # Preprocess data: scale, integrate
    ###########################################################################
    data = _scale(data=data, scale_factor=data_scale / sensitivity)

    if integrate:  # Integrate data to get strain
        data = _integrate(data=data, dt=dt)

    # Create DASArray object
    ##########################################################################
    das_data = DASArray(
        input_array=data,
        dt=dt,
        ds=ds,
        dx=dx,
        t_start=t_start,
        s_start=channels[0],
        gauge_length=gauge_length,
    )

    return das_data


def _scale(
    data: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Scale data by a given factor.

    Args:
        data (np.ndarray): Data to scale.
        scale_factor (float): Factor to scale data by.

    Returns:
        np.ndarray: Scaled data.
    """

    return data * scale_factor


def _integrate(
    data: np.ndarray,
    dt: float
):
    """Integrate data.

    Args:
        data (np.ndarray): Data to integrate.
        dt (float): Time step.

    Returns:
        np.ndarray: Integrated data.
    """
    return np.cumsum(data, axis=0) * dt


def gen_datetime(
    start: float,
    n: int,
    dt: float
) -> list[datetime.datetime]:
    """Generate a list of datetime values.

    Args:
        start (float): The starting timestamp (since the Unix epoch).
        n (int): Number of datetime values to generate.
        dt (float): Time gap between datetime values.

    Returns:
        list[datetime.datetime]: List of datetime values.
    """
    # Convert the start timestamp to a datetime object
    start_datetime = datetime.datetime.fromtimestamp(start, datetime.UTC)

    # Generate a list of datetime values
    datetime_list = [
        start_datetime + datetime.timedelta(seconds=dt * i) for i in range(n)
    ]

    return datetime_list
