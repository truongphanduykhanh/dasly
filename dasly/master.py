from dasly.loader import DASLoader
from dasly.filter import DASFilter
from dasly.plotter import DASPlotter
from dasly.sampler import DASSampler
from dasly.analyzer import DASAnalyzer


class Dasly(DASLoader, DASFilter, DASPlotter, DASSampler, DASAnalyzer):
    def __init__(self):
        DASLoader.__init__(self)
        DASFilter.__init__(self)
        DASPlotter.__init__(self)
        DASSampler.__init__(self)
        DASAnalyzer.__init__(self)
