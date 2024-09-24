import os
from datetime import datetime, timedelta
from typing import Optional


def _parse_file_path(file_path: str) -> tuple[str, str, str, str]:
    """Parses the file path to extract exppath, yyyymmdd, hhmmss, and
    hhmmss_hdf5.

    Args:
        file_path (str): The file path to parse.

    Returns:
        tuple[str, str, str, str]: A tuple containing exppath, yyyymmdd,
            hhmmss, and hhmmss_hdf5.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    try:
        exppath, yyyymmdd, _, hhmmss_hdf5 = file_path.rsplit('/', 3)
        hhmmss = hhmmss_hdf5[:-5]  # Remove '.hdf5' extension
    except ValueError:
        raise ValueError(
            "Invalid file path format. Expected format: " +
            "<exppath>/<YYYYMMDD>/dphi/<HHMMSS>.hdf5"
        )
    return exppath, yyyymmdd, hhmmss, hhmmss_hdf5


def _get_datetime_from_strings(yyyymmdd: str, hhmmss: str) -> datetime:
    """Converts date and time strings into a datetime object.

    Args:
        yyyymmdd (str): The date string in the format 'YYYYMMDD'.
        hhmmss (str): The time string in the format 'HHMMSS'.

    Returns:
        datetime: The datetime object.
    """
    try:
        return datetime.strptime(yyyymmdd + hhmmss, '%Y%m%d%H%M%S')
    except ValueError:
        raise ValueError("Invalid date or time format in file path.")


def _get_available_dates(exppath: str) -> list[str]:
    """Retrieves a sorted list of available date directories in reverse
    chronological order.

    Args:
        exppath (str): The experiment path.

    Returns:
        list[str]: A list of available date directories.
    """
    exppath_dates = os.path.join(exppath)
    available_dates = [
        d for d in os.listdir(exppath_dates)
        if os.path.isdir(os.path.join(exppath_dates, d))
        and d.isdigit() and len(d) == 8
    ]
    return sorted(available_dates, reverse=True)


def _get_file_times_for_date(date_dir: str) -> list[str]:
    """Retrieves a sorted list of HHMMSS strings from hdf5 filenames in a given
    date directory.

    Args:
        date_dir (str): The date directory containing hdf5 files.

    Returns:
        list[str]: A list of HHMMSS strings.
    """
    try:
        file_list = os.listdir(date_dir)
    except OSError:
        return []
    file_times = [
        f[:-5] for f in file_list
        if f.endswith('.hdf5') and f[:-5].isdigit() and len(f[:-5]) == 6
    ]
    return sorted(file_times, reverse=True)


def _collect_continuous_files(
    exppath: str,
    yyyymmdd: str,
    hhmmss: str,
    num_file: int,
    available_dates: list[str],
    input_dt: datetime,
    file_path: str
) -> list[str]:
    """Collects previous hdf5 files that are continuous in time without gaps.

    Args:
        exppath (str): The experiment path.
        yyyymmdd (str): The date in the format 'YYYYMMDD'.
        hhmmss (str): The time in the format 'HHMMSS'.
        num_file (int): The number of files to collect.
        available_dates (list[str]): A list of available date directories.
        input_dt (datetime): The datetime object of the input file.
        file_path (str): The input file path.

    Returns:
        list[str]: A list of hdf5 file paths.
    """
    collected_files = [file_path]
    last_dt = input_dt
    num_collected = 1
    gap_detected = False

    # Find the index of the starting date
    if yyyymmdd in available_dates:
        start_date_index = available_dates.index(yyyymmdd)
    else:
        raise ValueError(f"Date directory {yyyymmdd} not found in {exppath}")

    # Iterate over dates starting from the input date
    for date_str in available_dates[start_date_index:]:
        date_dir = os.path.join(exppath, date_str, 'dphi')
        if not os.path.isdir(date_dir):
            continue

        file_times_sorted = _get_file_times_for_date(date_dir)
        if date_str == yyyymmdd:
            # Only include files with time less than the starting time
            file_times_filtered = [t for t in file_times_sorted if t < hhmmss]
        else:
            file_times_filtered = file_times_sorted

        for t in file_times_filtered:
            try:
                dt = datetime.strptime(date_str + t, '%Y%m%d%H%M%S')
            except ValueError:
                continue  # Skip invalid times

            time_diff = last_dt - dt
            if timedelta(0) < time_diff <= timedelta(seconds=15):
                file_candidate = os.path.join(date_dir, f"{t}.hdf5")
                if os.path.exists(file_candidate):
                    collected_files.append(file_candidate)
                    last_dt = dt
                    num_collected += 1
                    if num_collected >= num_file:
                        return collected_files[::-1]  # Chronological order
            else:
                # Gap detected
                gap_detected = True
                break

        if gap_detected or num_collected >= num_file:
            break

    return collected_files[::-1]  # Return in chronological order


def get_file_paths_n(file_path: str, n: int) -> list[str]:
    """Gets a list of hdf5 file paths including the input file_path and
    previous (num_file - 1) continuous files without time gaps (maximum time
    gap is 15 seconds).

    Args:
        file_path (str): The input file path.
        n (int): The number of files to collect.

    Returns:
        list[str]: A list of hdf5 file paths.
    """
    exppath, yyyymmdd, hhmmss, _ = _parse_file_path(file_path)
    input_dt = _get_datetime_from_strings(yyyymmdd, hhmmss)
    available_dates = _get_available_dates(exppath)
    collected_files = _collect_continuous_files(
        exppath, yyyymmdd, hhmmss, n, available_dates, input_dt,
        file_path
    )
    return collected_files


def _validate_time_range_arguments(
    start: Optional[str],
    end: Optional[str],
    duration: Optional[float]
) -> tuple[datetime, datetime]:
    """Validates and calculates the start and end datetime objects based on
    provided arguments.

    Args:
        start (Optional[str]): The start time in the format 'YYYYMMDD HHMMSS'.
        end (Optional[str]): The end time in the format 'YYYYMMDD HHMMSS'.
        duration (Optional[float]): The duration in seconds.

    Returns:
        tuple[datetime, datetime]: A tuple containing the start and end
            datetime objects.
    """
    provided_args = [start, end, duration]
    if sum(arg is not None for arg in provided_args) != 2:
        raise ValueError(
            "Exactly two of 'start', 'end', or 'duration' must be provided."
        )

    datetime_format = '%Y%m%d %H%M%S'

    if start and end:
        start_dt = datetime.strptime(start, datetime_format)
        end_dt = datetime.strptime(end, datetime_format)
    elif start and duration:
        start_dt = datetime.strptime(start, datetime_format)
        end_dt = start_dt + timedelta(seconds=duration)
    elif end and duration:
        end_dt = datetime.strptime(end, datetime_format)
        start_dt = end_dt - timedelta(seconds=duration)
    else:
        raise ValueError("Invalid combination of arguments.")

    if start_dt >= end_dt:
        raise ValueError("Start time must be before end time.")

    return start_dt, end_dt


def _collect_files_in_range(
    exppath: str,
    end_dt: datetime,
    adjusted_start_dt: datetime
) -> list[str]:
    """Collects all necessary hdf5 files from exppath to cover the data range
    defined by start_dt and end_dt.

    Args:
        exppath (str): The experiment path.
        end_dt (datetime): The end datetime.
        adjusted_start_dt (datetime): The adjusted start datetime.

    Returns:
        list[str]: A list of hdf5 file paths.
    """
    datetime_format = '%Y%m%d %H%M%S'
    collected_files = []

    # Collect dates to cover
    date_list = []
    current_date = adjusted_start_dt.date()
    end_date = end_dt.date()
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)

    for date_str in date_list:
        date_dir = os.path.join(exppath, date_str, 'dphi')
        if not os.path.isdir(date_dir):
            continue  # Skip dates without the 'dphi' directory

        file_times = _get_file_times_for_date(date_dir)

        for hhmmss in file_times:
            file_dt_str = f"{date_str} {hhmmss}"
            try:
                file_dt = datetime.strptime(file_dt_str, datetime_format)
            except ValueError:
                continue  # Skip invalid times

            # Include files whose nominal time is between adjusted_start_dt and
            # end_dt (exclusive)
            if adjusted_start_dt <= file_dt < end_dt:
                file_path = os.path.join(date_dir, f"{hhmmss}.hdf5")
                collected_files.append((file_dt, file_path))

    # Sort collected files in chronological order
    collected_files.sort(key=lambda x: x[0])

    # Extract the file paths
    return [file_path for file_dt, file_path in collected_files]


def get_file_paths(
    exppath: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    duration: Optional[float] = None
) -> list[str]:
    """Returns all necessary hdf5 files from exppath to cover the data range
    defined by start, end, and duration. Exactly two out of the three
    parameters (start, end, duration) must be provided.

    Args:
        exppath (str): The experiment path.
        start (Optional[str]): The start time in the format 'YYYYMMDD HHMMSS'.
        end (Optional[str]): The end time in the format 'YYYYMMDD HHMMSS'.
        duration (Optional[float]): The duration in seconds.

    Returns:
        list[str]: A list of hdf5 file
    """
    # Validate and calculate start_dt and end_dt
    start_dt, end_dt = _validate_time_range_arguments(start, end, duration)

    # Adjust start time backward by 11 seconds to account for files
    # that may start earlier
    adjusted_start_dt = start_dt - timedelta(seconds=11)

    # Collect files in the specified time range
    collected_files = _collect_files_in_range(
        exppath, end_dt, adjusted_start_dt
    )

    return collected_files
