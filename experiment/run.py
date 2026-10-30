#!usr/bin/env python
"""
Created at 4/12/22
@author: DevXI

Script to run the perceptual staircase and adjustment task.
"""
from abc import ABC

from src.tasks import DDDTasks
from pathlib import Path
from src.analysis import TaskAnalyzer
from psychopy import logging
import numpy as np


class DDDLR(DDDTasks, ABC):

    def __init__(self, debug, root, practice, dummy_tracker):
        super().__init__(debug, root, practice, dummy_tracker)

        self.ROOT = root

        if self.practice:
            # self.stair_runs = 1
            self.match_runs = 1
            # self.adjust_runs = 1
        else:
            self.stair_runs = int(self.settings["tasks"]["staircase"]["runs"])
            self.match_runs = int(self.settings["tasks"]["match"]["runs"])
            self.adjust_runs = int(self.settings["tasks"]["adjustment"]["runs"])

    def go(self):

        # Initialize counters
        stair_run = 1
        match_run = 1
        adjust_run = 1
        run_counter = 1

        t_start = self.clocks["exp"].getTime()

        # run the staircase
        # for stair in range(self.stair_runs):
        #     self.run_staircase(stair + 1, run_counter)
        #     stair_run += 1
        #     run_counter += 1

        # get the results of the staircase
        # stair_results = dict()
        # analyzer = TaskAnalyzer(self.NAME, self.ROOT)
        # analyzer.task = 'staircase'
        # for dur in self.params["DRIFT_DURATIONS"]:
        #     for dist in self.params["GABOR_DISTANCES"]:
        #         if self.practice:
        #             stair_results[f"dur-{dur}_dist-{dist}"] = -45
        #         else:
        #             fit = analyzer.fit_staircase(dur, dist, sub=self._SUBJECT["id"])
        #             stair_results[f"dur-{dur}_dist-{dist}"] = analyzer.get_pse(fit["weibull"])

        # run the match task
        for match in range(self.match_runs):
            self.run_match(match + 1, run_counter)
            match_run += 1
            run_counter += 1

        # get the results of the match task
        match_results = dict()
        analyzer = TaskAnalyzer(self.NAME, self.ROOT)
        for dur in self.params["DRIFT_DURATIONS"]:
            for dist in self.params["GABOR_DISTANCES"]:
                for side in [0, 1]:
                    if self.practice:
                        match_results[f"dur-{dur}_dist-{dist}_side-{side}"] = 45
                    else:
                        analyzer.task = 'match'
                        stat = analyzer.get_matching_stats(dur, dist, side, self._SUBJECT["id"], kind='median')
                        match_results[f"dur-{dur}_dist-{dist}_side-{side}"] = stat

        # run the adjustment task
        for adjust in range(self.adjust_runs):
            self.run_adjustment(adjust + 1, run_counter, match_results)
            adjust_run += 1
            run_counter += 1

        # Get time
        t_end = self.clocks["exp"].getTime()

        # Log and end
        exp_mins = (t_end - t_start) // 60
        exp_secs = np.round((t_end - t_start) % 60)
        logging.exp(f"Experiment duration: {exp_mins} minutes and {exp_secs} seconds.")


if __name__ == '__main__':

    # Parameters
    ROOT = Path().cwd()
    DEBUG = False
    DUMMY = True
    PR = False

    # Make the instance
    ddd_exp = DDDLR(DEBUG, ROOT, PR, DUMMY)

    # Run some checks
    if not ddd_exp.DEBUG:
        status = ddd_exp.get_system_status()
        ddd_exp.show_msg(status)

    # Run trials
    ddd_exp.go()

    # Subject completed the experiment, save their info
    ddd_exp.save_sub()

    # Finish
    ddd_exp.end()
