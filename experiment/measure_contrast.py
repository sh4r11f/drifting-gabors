#!/usr/bin/env python
"""
created 2/12/2023

@author DevXI

"""
from pathlib import Path
from src.measurements import MeasureLuminance


ROOT = Path().cwd()
ruler = MeasureLuminance(ROOT)

ruler.michelson()
