import unittest
from datetime import datetime
<<<<<<< HEAD
from dasly.data_loader import Loader
=======
from dasly.data_loader import Dasly
>>>>>>> origin/main

class TestDasly(unittest.TestCase):

    def setUp(self):
<<<<<<< HEAD
        self.dasly = Loader()
=======
        self.dasly = Dasly()
>>>>>>> origin/main

    def test_load_data_with_folder_path(self):
        folder_path = "/path/to/experiment/folder"
        self.dasly.load_data(folder_path=folder_path)
        self.assertIsNotNone(self.dasly.signal_raw)
        self.assertIsNotNone(self.dasly.signal)
        self.assertIsNotNone(self.dasly.sampling_rate)
        self.assertIsNotNone(self.dasly.duration)
        self.assertIsNotNone(self.dasly.file_paths)

    def test_load_data_with_file_paths(self):
        file_paths = ["/path/to/file1.hdf5", "/path/to/file2.hdf5"]
        self.dasly.load_data(file_paths=file_paths)
        self.assertIsNotNone(self.dasly.signal_raw)
        self.assertIsNotNone(self.dasly.signal)
        self.assertIsNotNone(self.dasly.sampling_rate)
        self.assertIsNotNone(self.dasly.duration)
        self.assertEqual(self.dasly.file_paths, file_paths)

    def test_load_data_with_time_constraints(self):
        start = datetime(2024, 6, 1, 0, 0, 0)
        duration = 3600
        self.dasly.load_data(start=start, duration=duration)
        self.assertIsNotNone(self.dasly.signal_raw)
        self.assertIsNotNone(self.dasly.signal)
        self.assertIsNotNone(self.dasly.sampling_rate)
        self.assertIsNotNone(self.dasly.duration)
        self.assertIsNotNone(self.dasly.file_paths)

if __name__ == '__main__':
    unittest.main()