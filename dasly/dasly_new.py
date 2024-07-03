from dasly.data_loader import DataLoader
from dasly.data_filter import DataFilter
from dasly.data_plotter import DataPlotter


class Dasly(DataLoader, DataFilter, DataPlotter):
    def __init__(self, *args, **kwargs):
        DataLoader.__init__(self, *args, **kwargs)
        DataFilter.__init__(self)
        DataPlotter.__init__(self)
