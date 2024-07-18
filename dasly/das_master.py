from dasly.das_loader import DASLoader
from dasly.das_filter import DASFilter
from dasly.das_plotter import DASPlotter
from dasly.das_sampler import DASSampler
from dasly.das_analyzer import DASAnalyzer


class Dasly(DASLoader, DASFilter, DASPlotter, DASSampler, DASAnalyzer):
    def __init__(self):
        DASLoader.__init__(self)
        DASFilter.__init__(self)
        DASPlotter.__init__(self)
        DASSampler.__init__(self)
        DASAnalyzer.__init__(self)
