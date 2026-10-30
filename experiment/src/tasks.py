#!/usr/bin/env python
"""
Created at 4/3/2022

@ author Sharif Saleki

# Description
"""
from abc import ABC
from itertools import product
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from .double_drift import DoubleDrift
from psychopy import logging, data, event, core
from psychopy.tools.monitorunittools import pix2deg, deg2pix


class DDDTasks(DoubleDrift, ABC):
    """
    Contains all the methods to run three tasks for the FIPS saccade experiment:
        - Staircase
        - Perceptual Adjustment
        - Saccade

    Parameters
    ----------
    debug : bool
        Debug mode for the experiment.

    root_dir : str or Path
        Home directory of the experiment.

    practice: bool
        Practice mode

    dummy_tracker : bool
        Debug mode for the eye tracker.
    """

    def __init__(self, debug: bool, root_dir: Union[str, Path], practice: bool, dummy_tracker: bool, **kwargs):
        super().__init__(debug, root_dir, practice, dummy_tracker, **kwargs)

        # Initiate staircases here because we don't want a new block on each run
        # Staircase should be one across all runs or else it doesn't make sense
        self.stairs = None

    def run_staircase(self, run_n: int, exp_run: int):
        """
        Prepares the block and run, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for a response.
        Cleans up the trial, block, and experiment, and saves everything afterwards.

        Parameters
        ----------
        run_n : Run number in the current task

        exp_run : Run number in the entire experiment
        """
        # Set the parameters
        self.task = 'staircase'
        self.run = run_n
        self.exp_run = exp_run
        # Only make the MultistairsHandler on the first run and use the same thing throughout the staircase runs
        if run_n == 1:
            self.stairs = self.get_staircase_blocks()

        # Beginning messages
        practice_msg = \
            "On all trials, look directly at the dot in the middle of the screen.\n\n" \
            "Pay attention to the orientation of the moving Gabor.\n\n" \
            "Wait until a vertical line appears on the side of the screen opposite to the Gabor.\n\n" \
            "Using the vertical line as a reference:\n\n" \
            "(1) if the Gabor is on the tilted to the left, press the left arrow on the keyboard.\n\n" \
            "(2) if the Gabor is tilted to the right, press the right arrow on the keyboard.\n\n" \
            "Press the Spacebar to start the experiment."

        run_msg = f"===> Keyboard Run #{self.run} <===\n\n"

        # Show the message and log
        if self.practice:
            self.setup_run(practice_msg)
        else:
            self.setup_run(run_msg)

        # Number of blocks to divide into
        if self.DEBUG or self.practice:
            n_blocks = 1
            n_runs = 2
            n_trials_per_block = 2
        else:
            n_runs = self.TASKS["staircase"]["runs"]
            n_blocks = self.TASKS["staircase"]["blocks"]
            n_trials = self.TASKS["staircase"]["trials"]
            n_trials_per_block = n_trials // n_blocks // n_runs

        self.block = 1
        run_blocks = True

        # Start a block
        while run_blocks:

            # Set up the block
            self.trial = 1
            self.setup_block()

            # Get the initial values
            run_trials = self._get_next_stair()
            n_recycled = 0

            # Run trials in this block
            while run_trials is not None:

                # Unpack trial parameters
                theta, dur, dist, side = run_trials

                # Re-calibrate if many trials have failed in a row
                if n_recycled == int(self.params["RECYCLE_THRESH"]):
                    self.run_calibration()

                # Basic setup of the stimuli and eye tracker and wait for gaze
                self.setup_trial()
                # Calculate and set positions of frame, probes, and match stimuli based on side and offset
                self.set_init_positions(side, dist, dur, 'dd')

                # Start showing the stimuli and getting response
                try:

                    # Stabilization period
                    stab_dropped = self.stabilize(dur, 'dd', theta)

                    # Get the response: includes choice, number of dropped frames, and the reaction time
                    if self.DEBUG:
                        # simulate response for debugging
                        resp = 1
                        resp_dropped = 0
                        rt = 1
                        core.wait(.3)
                    else:
                        resp, resp_dropped, rt = self.get_staircase_response(dur, theta)

                    # add the dropped frames together
                    dropped = stab_dropped + resp_dropped

                    # Save the response
                    self.save_staircase_response((resp, dropped, rt), side, dur, theta)

                    # Advance the counter and get the next values
                    if self.trial < n_trials_per_block:
                        self.trial += 1
                        self.log_section("Trial", "end")
                        self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                        self.el_tracker.sendMessage(f"TRIAL_END")
                        run_trials = self._get_next_stair()
                    else:
                        run_trials = None

                    # Set number of recycled trials back to 0 since we want to re-calibrate only after
                    # a bunch of failures **in a row**
                    n_recycled = 0

                # Repeat the trial if gaze checks failed
                except RuntimeError:
                    # Don't recycle the trial if in debug mode
                    if self.DEBUG:
                        # Advance the counter and get the next values
                        if self.trial < n_trials_per_block:
                            self.trial += 1
                            self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                            self.el_tracker.sendMessage(f"TRIAL_END")
                            run_trials = self._get_next_stair()
                        else:
                            run_trials = None
                    else:
                        # Keep track of recycled trials
                        n_recycled += 1

                        # Log recycling
                        self._log_run(f"!!! Recycling trial {self.trial}.")
                        self._log_run("-" * 40)
                        self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                        self.el_tracker.sendMessage(f"TRIAL_FAIL")

                        # just wait half a second before repeating
                        core.wait(.5)

                # Clean up the screen and get ready for the next trial
                finally:
                    # Turn things off and clean up
                    self.trial_cleanup()

            # End of the block
            self.block_cleanup()

            if self.block < n_blocks:
                self.block += 1
            else:
                run_blocks = False

        # Save data only in the last run
        if not self.practice:
            if self.run == n_runs:
                self.save_staircase_data()

            # End the run
            self.run_cleanup(self.n_runs - exp_run)

    def _get_next_stair(self):
        """
        Retrieves the next conditions in a staircase

        Returns
        -------
        tuple or None
            Either returns the next offset, duration, and stimulus side, or if there is no next intensity
            returns None
        """
        # Get the next values
        try:
            theta, condition = self.stairs.next()
            dur = condition["DRIFT_DURATION"] * self.params["refresh_rate"] / 1000
            dist = deg2pix(condition["GABOR_DISTANCE"], self.monitor)
            side = np.random.choice([0, 1])
            print(theta)

            return theta, dur, dist, side

        # End of staircase
        except StopIteration:

            return None

    def get_staircase_blocks(self):
        """
        Generates a list of all blocks, each consisting of rows as trials and columns as parameters that
        either indicate features of the trial or should be populated by collected responses throughout each
        trial.
        """
        # set staircase values
        start_val = 100  # initial guess

        # staircase step sizes and reversals
        # steps = [start_val / d for d in np.repeat([2 ** i for i in range(5)], 4)]
        # steps = [start_val + d for d in np.repeat([i for i in np.arange(-20, 20, 5)], 2)]
        # steps = [start_val + d for d in np.repeat(np.arange(0, 25, 5), 2)]
        # steps = np.repeat(np.arange(5, 25, 5), 2)
        steps = [20, 20, 15, 15, 10, 10, 5, 5]

        # Subject and experiment parameters
        extra_info = {
            'RUN': self.run,
            'SESSION': self.session,
            'TASK': self.task,
            'EXP': self.NAME,
            'DATE': data.getDateStr(),
            'SUB_ID': self._SUBJECT["id"],
            'SUB_INIT': self._SUBJECT["init"]
        }

        # Staircase conditions
        combs = []
        for dist in self.params["GABOR_DISTANCES"]:
            for dur in self.params["DRIFT_DURATIONS"]:
                combs.append((f"dist{dist}_dur{dur}", dist, dur))

        stair_conditions = [
            {
                "label": label,
                'startVal': start_val,
                "GABOR_DISTANCE": dist,
                "DRIFT_DURATION": dur,
                "stepSizes": steps,
                'nUp': 1,
                'nDown': 1,
                'stepType': 'lin',
                'minVal': 90,
                'maxVal': 270,
                'extraInfo': extra_info,
                # 'startValSd': 50,
                # 'pThreshold': .63,
                # 'beta': 3.5,
                # 'grain': 1,
                # 'range': 200
            }
            for label, dist, dur in combs
        ]

        # Number of trials
        n_trials = 4 if self.DEBUG or self.practice else self.TASKS["staircase"]["trials"]

        # Staircases
        _stairs = data.MultiStairHandler(
            stairType='simple',
            # stairType='quest',
            method='fullRandom',
            conditions=stair_conditions,
            nTrials=n_trials,
            name='DDD',
            autoLog=False
        )

        # Log
        self._log_run(f"{len(_stairs.staircases)} staircases created.")

        return _stairs

    def get_staircase_response(self, duration: int, env_ori: float,) -> tuple:
        """
        Get the keyboard response from subject.
        """
        # Flush the event memory
        event.clearEvents()

        # Show the comparison circles
        self.stimuli["reference"].autoDraw = True

        # Log
        self._log_run("Response period started.")
        self.el_tracker.sendMessage("RESP_START")

        # Timing
        t_start = self.clocks["trial"].getTime()

        # Initiate vars
        resp = True
        trial_resp = None
        n_dropped = None

        # Move the frame until a response is made
        while resp:

            # Check for gaze
            gaze_check, recorded_frames = self._move_gabor(duration, 'dd', env_ori)

            # Make sure all gaze are valid
            self.validate_all_gaze(gaze_check, 'resp')

            # Make sure frame intervals were valid
            # n_dropped = self.validate_all_frames(recorded_frames)
            n_dropped = 0

            # Get response
            keys = event.getKeys()
            for key in keys:

                if key == 'left':

                    # Log
                    self._log_run("Response successful.")
                    self.el_tracker.sendMessage("RESP_END")

                    # Response
                    trial_resp = 1

                    # End report
                    resp = False

                elif key == 'right':

                    # Log
                    self._log_run("Response successful.")
                    self.el_tracker.sendMessage("RESP_END")

                    # Response
                    trial_resp = 0

                    # End report
                    resp = False

                # To be able to quit
                elif key == 'escape':

                    # Log
                    self._log_run("Staircase aborted.")
                    self.el_tracker.sendMessage("RESP_FAIL")

                    # Quit
                    self.terminate_tracker(end=True)
                    self.end()

        # Get the time to respond
        t_end = self.clocks["trial"].getTime()
        rt = t_end - t_start

        return trial_resp, n_dropped, rt

    def save_staircase_response(self, resp: Union[list, tuple], stim_side: int, duration: int, env_ori: float):
        """
        Save the response to the staircase. Also adds the stimulus side as other data.
        """
        # Unpack response
        trial_resp, n_dropped, rt = resp

        # Add info to staircase
        # a different response is a "missed" trial (0)
        # a match response is a "detected" trial (1)
        self.stairs.addResponse(trial_resp)
        self.stairs.addOtherData('DROPPED_FRAMES', n_dropped)
        self.stairs.addOtherData("RT", rt)
        self.stairs.addOtherData("SIDE", stim_side)
        self.stairs.addOtherData("DURATION_FRAMES", duration)
        self.stairs.addOtherData("ENVELOPE_ORI", env_ori)

    def save_staircase_data(self):
        """ Saves all the staircases and logs to disk."""

        # Loop all the staircases
        dfs = []
        staircases = self.stairs.staircases

        for staircase in staircases:

            # Get the probe duration parameter
            dur = staircase.condition["DRIFT_DURATION"]
            dist = staircase.condition["GABOR_DISTANCE"]

            # Get the total number of trials for making the dictionary below
            n_trials = len(staircase.intensities)

            # Make a dict that will turn into a pandas dataframe
            exp_data = {
                "DURATION_MS": np.repeat(dur, n_trials),
                "DURATION_FRAMES": np.around(staircase.otherData["DURATION_FRAMES"], self.round_err),
                "DISTANCE_DVA": np.repeat(dist, n_trials),
                "DISTANCE_PIX": np.repeat(np.round(deg2pix(dist, self.monitor), self.round_err), n_trials),
                "SIDE": staircase.otherData["SIDE"],
                "ANGLE": np.around(staircase.intensities, self.round_err),
                "RESP": staircase.data,
                "RT": np.around(staircase.otherData["RT"], self.round_err),
                "DROPPED_FRAMES": staircase.otherData["DROPPED_FRAMES"],
                "RUN": np.repeat(staircase.extraInfo["RUN"], n_trials),
                "SESSION": np.repeat(staircase.extraInfo["SESSION"], n_trials),
                "TASK": np.repeat(staircase.extraInfo["TASK"], n_trials),
                "EXP": np.repeat(staircase.extraInfo["EXP"], n_trials),
                "DATE": np.repeat(staircase.extraInfo["DATE"], n_trials),
                "SUB_ID": np.repeat(staircase.extraInfo["SUB_ID"], n_trials),
                "SUB_INIT": np.repeat(staircase.extraInfo["SUB_INIT"], n_trials)
            }

            # make it into a data frame
            df = pd.DataFrame(exp_data)

            # save it
            dfs.append(df)

            # save to pickle
            pickle_file = str(self.files["run"]) + f'_duration-{dur}_distance-{dist}'
            staircase.saveAsPickle(pickle_file)

        # Concatenate all staircases
        all_data = pd.concat(dfs, ignore_index=True)

        # Save the dataframe
        pd_file = str(self.files["run"]) + '.csv'
        all_data.to_csv(pd_file, index=False)

        # Save the log files
        logging.flush()
        logging.root.removeTarget(self.logs["task"])

    # -------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- ADJUSTMENT TASK ----------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------- #

    def run_adjustment(self, run_n: int, exp_run: int, stair_results):
        """
        Prepares the block and run, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for the saccade.
        Cleans up the trial, block, and experiment, and saves everything.
        """
        # Set the parameters
        self.task = 'adjustment'
        self.run = run_n
        self.exp_run = exp_run

        # Beginning message
        practice_msg = \
            "Maintain fixation throughout each trial.\n\n" \
            "On each trial, pay attention to the path length of the moving stimulus.\n\n" \
            "Use the mouse wheel to change the length of the line that you see at the fixation point.\n\n" \
            "Change the length until it matches the path length of the moving stimulus.\n\n " \
            "When they match, press the Spacebar to advance to the next trial.\n\n " \
            "Press the Spacebar to start the calibration procedure."

        run_msg = f"==> Mouse Run #{self.run} <==\n\n"

        # Show the message and log
        if self.practice:
            self.setup_run(practice_msg)
        else:
            self.setup_run(run_msg)

        # Get the blocks and name of columns
        exp_blocks, col_names = self.get_adjustment_blocks(stair_results)

        if self.practice or self.DEBUG:
            exp_blocks = [exp_blocks[0][:4]]

        # Start looping blocks
        for b, block in enumerate(exp_blocks):

            # Set up the block
            self.block = b + 1
            if b:
                calibrate = False
            else:
                calibrate = True
            self.setup_block(calibrate)
            n_recycled = 0

            # Make a copy of the blocks to work with because of recycling and whatnot
            trials = block.copy()
            trial_counter = 1  # trial counter

            # Looping until all trials have been successful
            while len(trials):

                # Re-calibrate if many trials have failed in a row
                if n_recycled == self.params["RECYCLE_THRESH"]:
                    self.run_calibration()

                # The current trial is always the first index
                trial = trials[0]

                # Get the parameters
                dur = int(float(trial[0]))
                dist = float(trial[3])
                side = int(float(trial[4]))
                theta = float(trial[5])
                self.trial = int(float(trial[13]))
                stim_type = trial[22]

                # Basic setup of the stimuli and eye tracker and wait for gaze
                self.setup_trial()
                self.set_init_positions(side, dist, dur, stim_type)

                # Start showing the stimuli and getting response
                try:

                    # Stabilization period
                    # stab_dropped = self.stabilize(dur, stim_type, theta)
                    stab_dropped = 0

                    # Perceptual task doesn't have stabilization period, go straight to measurement
                    if self.DEBUG:
                        # simulate response for debugging
                        resp, resp_dropped, rt = 0, 0, 0
                        core.wait(3)
                    else:
                        resp, resp_dropped, rt = self.get_adjustment_response(dur, theta, stim_type)

                    # add the dropped frames together
                    dropped = stab_dropped + resp_dropped

                    # Save the report
                    self.save_adjustment_response(block, (resp, dropped, rt))

                    # Advance the counter and pop this trial
                    trial_counter += 1
                    trials = trials[1:]
                    n_recycled = 0

                    # Log
                    self.log_section("Trial", "end")
                    self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                    self.el_tracker.sendMessage(f"TRIAL_END")

                # Catch the runtime errors sent because of gaze or saccade failure and recycle the trial
                except RuntimeError:
                    # Don't recycle the trial if in debug mode
                    if self.DEBUG:
                        # Advance the counter and pop this trial
                        trial_counter += 1
                        trials = trials[1:]
                        n_recycled = 0

                        # Log
                        self.log_section("Trial", "end")
                        self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                        self.el_tracker.sendMessage(f"TRIAL_END")
                    else:
                        # Recycle the trial
                        trials = self.recycle_trial(trials, block, 8)
                        n_recycled += 1

                        # Send info to the tracker
                        self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                        self.el_tracker.sendMessage(f"TRIAL_FAIL")

                        # Wait half a second
                        core.wait(.5)

                # Clean up for presenting the next trial anyway
                finally:

                    # Turn things off and clean up
                    self.trial_cleanup()

            # End of block
            self.block_cleanup()

        # Save data
        if not self.practice:
            self.save_run_data(exp_blocks, col_names)

            # End the run
            self.run_cleanup(self.n_runs - exp_run)

    def get_adjustment_blocks(self, pses: dict):
        """
        Generates a list of all blocks, each consisting of rows as trials and columns as parameters that
        either indicate features of the trial or should be populated by collected responses throughout each
        trial.
        """
        # Columns of experiment dataframe
        cols = [
            "DURATION_FRAMES",
            "DURATION_MS",
            "DISTANCE_DVA",
            "DISTANCE_PIX",
            "STIM_SIDE",
            "ENVELOPE_ANGLE",
            "ACTUAL_LENGTH_DVA",
            "ACTUAL_LENGTH_PIX",
            "REPORT_LENGTH_DVA",
            "REPORT_LENGTH_PIX",
            "RT",
            "DROPPED_FRAMES",
            "RECYCLED",
            "TRIAL",
            "BLOCK",
            "RUN",
            "SESSION",
            "TASK",
            "EXP",
            "DATE",
            "SUB_ID",
            "SUB_INIT",
            "STIM_TYPE"
        ]

        # Conditions
        durations = self.params["DRIFT_DURATIONS"]
        sides = [0, 1]
        dists = self.params["GABOR_DISTANCES"]
        stim_types = ['dd', 'deva']

        # Number of trials and blocks
        # if self.DEBUG or self.practice:
        #     n_trials_per_cond = 1
        #     n_blocks = 1
        # else:
        n_trials_per_cond = self.TASKS["adjustment"]["trials"]
        n_blocks = self.TASKS["adjustment"]["blocks"]

        # Initiate block list
        conditions = list(product(durations, sides, dists, stim_types))
        exp_blocks = []
        n_trials_per_block = len(conditions) * n_trials_per_cond

        # Start making blocks
        for b in range(1, n_blocks + 1):

            # Initiate the block
            rows = None

            # Loop through conditions and make as many as the specified number of trials for it
            for dur, side, dist, stim_type in conditions:

                # Get angles from the PSEs
                angle = pses[f"dur-{dur}_dist-{dist}_side-{side}"]

                # Make as many trials as specified
                for trial in range(n_trials_per_cond):

                    # Make this trial's array
                    row = np.array([
                        # 0: Move duration in frames
                        self.ms2fr(dur),
                        # 1: Move duration in ms
                        dur,
                        # 2: Distance from fixation in dva
                        dist,
                        # 3: Distance from fixation in pixels
                        np.round(deg2pix(dist, self.monitor), self.round_err),
                        # 4: Side of the frame (left or right)
                        side,
                        # 5: angle of the gabor envelope in degrees
                        np.round(angle, self.round_err),
                        # 6: Actual length of the moving gabor in dva
                        np.round(dur * self.params["ENV_SPEED"] / 1000, self.round_err),
                        # 7: Actual length of the moving gabor in pixels
                        np.round(deg2pix(dur * self.params["ENV_SPEED"] / 1000, self.monitor), self.round_err),
                        np.NaN,  # 8: Reported length in dva
                        np.NaN,  # 9: Reported length in pixels
                        np.NaN,  # 10: Trial response time
                        np.NaN,  # 11: Number of dropped frames in this trial
                        0,  # 12: Was this a recycled trial? 0 = no by default
                        np.NaN,  # 13: Trial number
                        b,  # 14: Block number
                        self.run,  # 15: Run number
                        self.session,  # 16: Session number
                        self.task,  # 17: Task name
                        self.NAME,  # 18: Experiment name
                        data.getDateStr(),  # 19: Date and time of the run
                        self._SUBJECT["id"],  # 20: Subject ID
                        self._SUBJECT["init"],  # 21: Subject's initials
                        stim_type  # 22: Double-drift or the DeValois
                    ])

                    # Stack this trial on top of all the other trials
                    if rows is None:
                        rows = row
                    else:
                        rows = np.vstack((rows, row))

            # Sanity check
            assert rows.shape == (n_trials_per_block, len(cols))

            # shuffle the trials within a block
            np.random.shuffle(rows)  # this preserves the order within row and just shuffles the rows

            # Add the block and trial labels
            rows[:, 13] = np.arange(1, n_trials_per_block + 1)

            # Append this block to the list of experiment blocks
            exp_blocks.append(rows)

        # Shuffle the order of blocks
        np.random.shuffle(exp_blocks)

        # Log the shape of the blocks and trials
        self._log_run(f"Adjustment blocks created. Shape of each block: {exp_blocks[0].shape}")

        return exp_blocks, cols

    def get_adjustment_response(self, duration: int, angle: float, stimulus: str) -> tuple:
        """
        Get the response from the participant.

        Returns
        -------
        tuple
            The final response about the distance between a probe and 0.
        """
        # Refresh everything
        event.clearEvents()

        # Log
        self._log_run("Response period started.")
        self.el_tracker.sendMessage("RESP_START")

        # Timing
        t_start = self.clocks["trial"].getTime()

        # Initiate vars
        resp = True
        trial_resp = None
        n_dropped = None
        mouse = event.Mouse(visible=False, win=self.window)

        # Move the frame until a response is made
        while resp:

            # Show the stimulus
            gaze_check, frame_durs = self._move_gabor(duration, stimulus, angle, mouse=mouse)

            # Make sure gaze is valid
            self.validate_all_gaze(gaze_check, 'resp')

            # Make sure frame intervals are valid
            # n_dropped = self.validate_all_frames(frame_durs)
            n_dropped = 0

            # Get end report and save response
            keys = event.getKeys()
            for key in keys:
                if key == 'space':

                    # The final response is the shift of the match stimuli, but since this is not always accurate,
                    # I get the mean of the shifts. That is accurate enough (a couple of pixels difference)
                    # trial_resp = self.stimuli["resp_line"].start[1] - self.stimuli["resp_line"].end[1]
                    trial_resp = self.stimuli["resp_line"].end[1]

                    # Log
                    self._log_run("Response successful.")
                    self.el_tracker.sendMessage("RESP_END")

                    # End report
                    resp = False

                elif key == 'escape':  # to be able to quit

                    # Log
                    self._log_run("Adjustment task aborted.")
                    self.el_tracker.sendMessage("RESP_FAIL")

                    # Quit
                    self.terminate_tracker(end=True)

        # Get the time to respond
        t_end = self.clocks["trial"].getTime()
        rt = t_end - t_start

        return trial_resp, n_dropped, rt

    def save_adjustment_response(self, block: np.array, resp: tuple):
        """
        Saves the participant response.

        Parameters
        ----------
        block : numpy.array
            All trials

        resp : tuple
            The report of probe positions in pixels

        """
        # Unpack the response
        length, n_dropped, rt = resp

        # Convert into degrees
        length_dva = pix2deg(length, self.monitor)

        # Save to the blocks
        block[self.trial - 1, 8] = np.round(length_dva, self.round_err)
        block[self.trial - 1, 9] = np.round(length, self.round_err)

        block[self.trial - 1, 10] = np.round(rt, self.round_err)
        block[self.trial - 1, 11] = n_dropped

    # -------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- Matching TASK ------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------- #

    def run_match(self, run_n: int, exp_run: int):
        """
        Prepares the block and run, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for the saccade.
        Cleans up the trial, block, and experiment, and saves everything.
        """
        # Set the parameters
        self.task = 'match'
        self.run = run_n
        self.exp_run = exp_run

        # Beginning message
        practice_msg = \
            "Maintain fixation throughout each trial.\n\n" \
            "Pay attention to the orientation of the moving Gabor.\n\n" \
            "Use the mouse wheel to change the orientation of the path until it looks vertical.\n\n" \
            "You can use the line on the other side of the screen as a reference.\n\n" \
            "When the path looks vertical, press the Spacebar to advance to the next trial.\n\n " \
            "Press the Spacebar to start the calibration procedure."

        run_msg = f"==> Orientation Run #{self.run} <==\n\n"

        # Show the message and log
        if self.practice:
            self.setup_run(practice_msg)
        else:
            self.setup_run(run_msg)

        # Get the blocks and name of columns
        exp_blocks, col_names = self.get_match_blocks()

        if self.practice or self.DEBUG:
            exp_blocks = [exp_blocks[0][:4]]

        # Start looping blocks
        for b, block in enumerate(exp_blocks):

            # Set up the block
            self.block = b + 1
            if b:
                calibrate = False
            else:
                calibrate = True
            self.setup_block(calibrate)
            n_recycled = 0

            # Make a copy of the blocks to work with because of recycling and whatnot
            trials = block.copy()
            trial_counter = 1  # trial counter

            # Looping until all trials have been successful
            while len(trials):

                # Re-calibrate if many trials have failed in a row
                if n_recycled == self.params["RECYCLE_THRESH"]:
                    self.run_calibration()

                # The current trial is always the first index
                trial = trials[0]

                # Get the parameters
                dur = int(float(trial[0]))
                dist = float(trial[3])
                side = int(float(trial[4]))
                self.trial = int(float(trial[9]))
                stim_type = 'dd'

                # Basic setup of the stimuli and eye tracker and wait for gaze
                self.setup_trial()
                self.set_init_positions(side, dist, dur, stim_type)

                # Start showing the stimuli and getting response
                try:

                    # Stabilization period
                    # stab_dropped = self.stabilize(dur, stim_type, theta)
                    stab_dropped = 0

                    # Perceptual task doesn't have stabilization period, go straight to measurement
                    if self.DEBUG:
                        # simulate response for debugging
                        resp, resp_dropped, rt = 0, 0, 0
                        core.wait(3)
                    else:
                        resp, resp_dropped, rt = self.get_match_response(dur)

                    # add the dropped frames together
                    dropped = stab_dropped + resp_dropped

                    # Save the report
                    self.save_match_response(block, (resp, dropped, rt))

                    # Advance the counter and pop this trial
                    trial_counter += 1
                    trials = trials[1:]
                    n_recycled = 0

                    # Log
                    self.log_section("Trial", "end")
                    self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                    self.el_tracker.sendMessage(f"TRIAL_END")

                # Catch the runtime errors sent because of gaze or saccade failure and recycle the trial
                except RuntimeError:
                    # Don't recycle the trial if in debug mode
                    if self.DEBUG:
                        # Advance the counter and pop this trial
                        trial_counter += 1
                        trials = trials[1:]
                        n_recycled = 0

                        # Log
                        self.log_section("Trial", "end")
                        self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                        self.el_tracker.sendMessage(f"TRIAL_END")
                    else:
                        # Recycle the trial
                        trials = self.recycle_trial(trials, block, 8)
                        n_recycled += 1

                        # Send info to the tracker
                        self.el_tracker.sendMessage(f"TRIALID_{self.trial}")
                        self.el_tracker.sendMessage(f"TRIAL_FAIL")

                        # Wait half a second
                        core.wait(.5)

                # Clean up for presenting the next trial anyway
                finally:

                    # Turn things off and clean up
                    self.trial_cleanup()

            # End of block
            self.block_cleanup()

        # Save data
        if not self.practice:
            self.save_run_data(exp_blocks, col_names)

            # End the run
            self.run_cleanup(self.n_runs - exp_run)

    def get_match_blocks(self):
        """
        Generates a list of all blocks, each consisting of rows as trials and columns as parameters that
        either indicate features of the trial or should be populated by collected responses throughout each
        trial.
        """
        # Columns of experiment dataframe
        cols = [
            "DURATION_FRAMES",
            "DURATION_MS",
            "DISTANCE_DVA",
            "DISTANCE_PIX",
            "STIM_SIDE",
            "ANGLE",
            "RT",
            "DROPPED_FRAMES",
            "RECYCLED",
            "TRIAL",
            "BLOCK",
            "RUN",
            "SESSION",
            "TASK",
            "EXP",
            "DATE",
            "SUB_ID",
            "SUB_INIT",
            "STIM_TYPE"
        ]

        # Conditions
        durations = self.params["DRIFT_DURATIONS"]
        stim_sides = [0, 1]
        dists = self.params["GABOR_DISTANCES"]
        stim_types = ['dd']

        # Number of trials and blocks
        # if self.DEBUG or self.practice:
        #     n_trials_per_cond = 1
        #     n_blocks = 1
        # else:
        n_trials_per_cond = self.TASKS["match"]["trials"]
        n_blocks = self.TASKS["match"]["blocks"]

        # Initiate block list
        conditions = list(product(durations, stim_sides, dists, stim_types))
        exp_blocks = []
        n_trials_per_block = len(conditions) * n_trials_per_cond

        # Start making blocks
        for b in range(1, n_blocks + 1):

            # Initiate the block
            rows = None

            # Loop through conditions and make as many as the specified number of trials for it
            for dur, stim_side, dist, stim_type in conditions:

                # Make as many trials as specified
                for trial in range(n_trials_per_cond):

                    # Make this trial's array
                    row = np.array([
                        # 0: Move duration in frames
                        self.ms2fr(dur),
                        # 1: Move duration in ms
                        dur,
                        # 2: Distance from fixation in dva
                        dist,
                        # 3: Distance from fixation in pixels
                        np.round(deg2pix(dist, self.monitor), self.round_err),
                        # 4: Side of the frame (left or right)
                        stim_side,
                        # 5: reported angle
                        np.NaN,
                        np.NaN,  # 6: Trial response time
                        np.NaN,  # 7: Number of dropped frames in this trial
                        0,  # 8: Was this a recycled trial? 0 = no by default
                        np.NaN,  # 9: Trial number
                        b,  # 10: Block number
                        self.run,  # 11: Run number
                        self.session,  # 12: Session number
                        self.task,  # 13: Task name
                        self.NAME,  # 14: Experiment name
                        data.getDateStr(),  # 15: Date and time of the run
                        self._SUBJECT["id"],  # 16: Subject ID
                        self._SUBJECT["init"],  # 17: Subject's initials
                        stim_type  # 18: Double-drift or the DeValois
                    ])

                    # Stack this trial on top of all the other trials
                    if rows is None:
                        rows = row
                    else:
                        rows = np.vstack((rows, row))

            # Sanity check
            assert rows.shape == (n_trials_per_block, len(cols))

            # shuffle the trials within a block
            np.random.shuffle(rows)  # this preserves the order within row and just shuffles the rows

            # Add the block and trial labels
            rows[:, 9] = np.arange(1, n_trials_per_block + 1)

            # Append this block to the list of experiment blocks
            exp_blocks.append(rows)

        # Shuffle the order of blocks
        np.random.shuffle(exp_blocks)

        # Log the shape of the blocks and trials
        self._log_run(f"Match blocks created. Shape of each block: {exp_blocks[0].shape}")

        return exp_blocks, cols

    def get_match_response(self, duration: int) -> tuple:
        """
        Get the response from the participant.

        Returns
        -------
        tuple
            The final response about the distance between a probe and 0.
        """
        # Refresh everything
        event.clearEvents()

        # Log
        self._log_run("Response period started.")
        self.el_tracker.sendMessage("RESP_START")

        # Timing
        t_start = self.clocks["trial"].getTime()

        # Initiate vars
        resp = True
        trial_resp = None
        n_dropped = None
        mouse = event.Mouse(visible=False, win=self.window)

        # Move the frame until a response is made
        while resp:

            # Show the stimulus
            gaze_check, frame_durs = self._move_gabor(duration, 'dd', mouse=mouse)

            # Make sure gaze is valid
            self.validate_all_gaze(gaze_check, 'resp')

            # Make sure frame intervals are valid
            # n_dropped = self.validate_all_frames(frame_durs)
            n_dropped = 0

            # Get end report and save response
            keys = event.getKeys()
            for key in keys:
                if key == 'space':

                    # Get the response
                    trial_resp = 90 - float(self.stimuli["gabor"].ori)

                    # reset
                    self.stimuli["gabor"].ori = 45

                    # Log
                    self._log_run("Response successful.")
                    self.el_tracker.sendMessage("RESP_END")

                    # End report
                    resp = False

                elif key == 'escape':  # to be able to quit

                    # Log
                    self._log_run("Match task aborted.")
                    self.el_tracker.sendMessage("RESP_FAIL")

                    # Quit
                    self.terminate_tracker(end=True)

        # Get the time to respond
        t_end = self.clocks["trial"].getTime()
        rt = t_end - t_start

        return trial_resp, n_dropped, rt

    def save_match_response(self, block: np.array, resp: tuple):
        """
        Saves the participant response.

        Parameters
        ----------
        block : numpy.array
            All trials

        resp : tuple
            The report of probe positions in pixels

        """
        # Unpack the response
        angle, n_dropped, rt = resp

        # Save to the blocks
        block[self.trial - 1, 5] = np.round(angle, self.round_err)
        block[self.trial - 1, 6] = np.round(rt, self.round_err)
        block[self.trial - 1, 7] = n_dropped
