from dasly.das_loader import DasLoader
from dasly.das_filter import DasFilter
from dasly.das_plotter import DasPlotter
from dasly.das_sampler import DasSampler
from dasly.das_analyzer import DasAnalyzer


class Dasly(DasLoader, DasFilter, DasPlotter, DasSampler, DasAnalyzer):
    def __init__(self, *args, **kwargs):
        DasLoader.__init__(self, *args, **kwargs)
        DasFilter.__init__(self)
        DasPlotter.__init__(self)
        DasSampler.__init__(self)
        DasAnalyzer.__init__(self)
