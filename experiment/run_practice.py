#!usr/bin/env python
"""
Created at 2/25/22
@author: devxi

Script to run the practice for FIPS saccade experiment. It prepares the subject for correct reports
and how to make a saccade. All three parts of the experiment (1) Staircase (2) Perceptual (3) Saccade
are included in this short practice run.
"""
from run import DDDLR
from pathlib import Path

ROOT = Path().cwd()
DEBUG = True
DUMMY = True
PRACTICE = True

ddd_practice = DDDLR(DEBUG, ROOT, PRACTICE, DUMMY)
ddd_practice.go()
ddd_practice.end()
