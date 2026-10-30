#!usr/bin/env python
"""
Created 4/2/22
@author Sharif Saleki


"""
from abc import ABC

import pylink

from .experiments import EyeTracking
from typing import Union
from pathlib import Path
from psychopy import visual, event
from psychopy.tools.monitorunittools import deg2pix, pix2deg
import numpy as np


class DoubleDrift(EyeTracking, ABC):
    """

    """

    def __init__(self, debug: bool, root_dir: Union[str, Path], practice, dummy_tracker: bool, **kwargs):
        super().__init__(debug, root_dir, practice, dummy_tracker, **kwargs)

        self._make_stim()
        self.setup_stimuli()

    def setup_stimuli(self):

        # Set the parameters
        # Fixation
        self.stimuli["fix"].size = deg2pix(self.params["FIX_SIZE"], self.monitor)
        fix_pos_tracker = [self.window.size[0] / 2, (self.window.size[1] / 2)]

        # Gabor
        size_px = deg2pix(self.params["GABOR_SIZE"], self.monitor)
        # self.stimuli["gabor"].size = size_px
        # self.stimuli["gabor"].sf = self.params["GABOR_CYCLES"] / self.params["GABOR_SIZE"]

        # Dynamics
        env_speed_per_frame = deg2pix(self.params["ENV_SPEED"], self.monitor) / self.params["refresh_rate"]
        # texture speed is in Hz or cycles/sec. We only need to convert to cycles/frame.
        tex_speed_per_frame = self.params["TEX_SPEED"] / self.params["refresh_rate"]

        # durations
        # durs = [self.ms2fr(t) for t in self.params["DRIFT_DURATION"]]
        durs = list(map(self.ms2fr, self.params["DRIFT_DURATIONS"]))

        self.params["fix_pos_tracker"] = fix_pos_tracker
        self.params["v_env"] = env_speed_per_frame
        self.params["v_tex"] = tex_speed_per_frame
        self.params["drift_durations_fr"] = durs

    def _make_stim(self):

        # Fixation dot
        fix = visual.Circle(
            win=self.window,
            lineColor=0,
            fillColor=-1,
            autoLog=False
        )

        # Gabor
        size_px = deg2pix(self.params["GABOR_SIZE"], self.monitor)
        sf_px = self.params['GABOR_CYCLES'] / deg2pix(1, self.monitor)  # Convert from cycles/deg to cycles/pix
        gabor = visual.GratingStim(
            win=self.window,
            mask='gauss',
            tex='sin',
            size=size_px,
            sf=sf_px,
            ori=0,
            contrast=0.8,
            autoLog=False
        )

        # Response line
        resp_line = visual.Line(
            win=self.window,
            start=(0, 0),
            end=(0, 5),
            lineWidth=7,
            lineColor=-1,
            opacity=1,
            autoLog=False
        )

        # Reference line
        line_size_px = deg2pix(self.params["REF_LINE_LEN"], self.monitor)
        ref_line = visual.Line(
            win=self.window,
            start=(0, line_size_px / 2),
            end=(0, -line_size_px / 2),
            # size=deg2pix(self.params["REF_LINE_LEN"], self.monitor),
            lineWidth=5,
            lineColor=-1,
            opacity=1,
            autoLog=False
        )

        # Save
        self.stimuli = {
            "fix": fix,
            "gabor": gabor,
            "resp_line": resp_line,
            "reference": ref_line
        }

    def set_init_positions(self, stim_side: int, distance: float, duration: float, stim_type: str):
        """
        Calculate and set the position of the stimuli.

        Parameters
        ----------
        stim_side : int
            0 for left and 1 for rights

        distance : float
            Eccentricity of the gabor patch on this trial

        duration : float
            Duration of the drift

        stim_type : str
            dd or deva
        """
        # Get the path length of the double-drift
        path_length = self.params["v_env"] * duration

        # initiate this variable. if the stimulus is not a double drift we don't need the reference line
        # all. We actually only need it in the staircase task for the double drift.
        ref_pos = [0, 0]
        ref_margin = deg2pix(3, self.monitor)

        # The right side
        if stim_type == 'dd':
            # Get the components of the distance vector
            x, y = self.get_xy(distance, 25)
            if stim_side == 1:
                gpos = [x, y]
                ref_pos = [-x - ref_margin, y]
            else:
                gpos = [-x, y]
                ref_pos = [x + ref_margin, y]
        else:
            # Put the devalois in midway point
            # Get the components of the distance vector
            # x, y = self.get_xy(distance + path_length, 45)
            x, y = self.get_xy(distance, 25)
            if stim_side == 1:
                gpos = [x, y]
            else:
                gpos = [-x, y]

        self.stimuli["gabor"].pos = gpos
        self.stimuli["reference"].pos = ref_pos
        # self.stimuli["reference"].size = path_length

    @staticmethod
    def get_xy(dist, angle):

        coords = [np.cos(np.deg2rad(angle)) * dist, np.sin(np.deg2rad(angle)) * dist]

        return coords

    def setup_trial(self):
        """
        The first stage of a trial.
            - Checks the connection of the tracker.
            - Sets the position of stimuli.
            - Performs drift correction.
            - Starts recording.
            - Draws a fixation point and waits for gaze.
        """
        # Reset trial clock
        self.clocks["trial"].reset()

        # Check if tracker is connected
        self.check_connection()

        # put tracker in idle/offline mode before recording
        self.el_tracker.setOfflineMode()

        # Drift correction for the eye tracker
        self.drift_correction(self.params["fix_pos_tracker"])

        # Put the eye tracker offline again
        self.el_tracker.setOfflineMode()

        # Start recording
        error = self.start_recording()
        if error is not None:
            raise RuntimeError

        # Allocate some time for the tracker to cache some samples
        delay_ms = 100
        pylink.pumpDelay(100)

        # Also give the window some time to warm up
        delay_frames = int(delay_ms * self.params["refresh_rate"] / 1000)
        for fr in range(delay_frames):
            self.window.flip()

        # Get eye information
        _ = self.get_eye()

        # Draw fixation
        self.stimuli["fix"].color = -1
        self.stimuli["fix"].autoDraw = True
        self.window.flip()

        # Tell the tracker that fixation was drawn
        self.el_tracker.sendMessage('fix_onset')

        # Start recording frame intervals
        self.window.recordFrameIntervals = True

        # Wait for a gaze trigger to start the trial
        self.wait_for_gaze(self.params["fix_pos_tracker"])

        # Log
        self.log_section('Trial', 'start')
        self.el_tracker.sendMessage(f"TRIAL_START")
        self.el_tracker.sendMessage(f"TRIALID_{self.trial}")

        # Reset line size
        self.stimuli["resp_line"].end[1] = deg2pix(0.3, self.monitor)

    def stabilize(self, duration: int, stim_type: str, env_ori: float):
        """
        Runs the stabilization period for the frame effect to set in.

        Parameters
        ----------
        duration : int
            Number of screen refreshes to move

        stim_type : str
            dd or deva

        env_ori : float
            External orientation
        """
        # Log
        self._log_run("Stabilization period started.")
        self.el_tracker.sendMessage("stabilization_start")

        # Keep track of number of dropped frames
        rec_frames = []

        # Move the frame for the designated number of paths
        for stab in range(self.params["STABILIZATION"]):
            gaze_check, stab_frames = self._move_gabor(duration, stim_type, env_ori)

            # Save frame times
            rec_frames.extend(stab_frames)

            # Make sure that gazes were valid
            self.validate_all_gaze(gaze_check, 'stabilization')

        # Log again
        self._log_run("Stabilization successful.")
        self.el_tracker.sendMessage("stabilization_end")

        # Make sure frame intervals were valid
        # n_dropped = self.validate_all_frames(np.array(rec_frames))
        n_dropped = 0

        # Don't get any responses during this period, just return
        return n_dropped

    def _move_gabor(self, duration: int, stim_type: str, env_ori: float = None, mouse: event.Mouse = None) -> tuple:
        """
        Moves the double-drift gabor on the screen

        Parameters
        ----------
        duration : int
            How many frames the gabor will move for

        stim_type : str
            Either dd or deva

        env_ori : float
            Angle of the envelope from horizontal

        mouse : event.Mouse
            Whether to use the mouse for getting responses or not

        Returns
        -------

        """
        # Get the stimuli parameters
        tex_v = self.params["v_tex"]
        line = self.stimuli["resp_line"]

        # if self.task == 'adjustment':
        #     if stim_type == 'dd':
        #         tex_ori = self.params["INIT_ENV_ORI"]
        #         env_ori = tex_ori
        #     else:
        #         tex_ori = env_ori
        #
        #         env_v = self.get_xy(self.params["v_env"], env_ori)
        # self.stimuli["gabor"].ori = tex_ori

        if self.task == 'adjustment':
            if stim_type == 'dd':

                # The texture is always aligned with the envelope orientation for the double-drift
                # Psychopy thinks that vertical is 0, so I'll account for that.
                tex_ori = 90 - env_ori

                # Calculate the x and y speeds at which the double-drift stimulus should move
                env_v = self.get_xy(self.params["v_env"], env_ori)

            else:

                # The texture is always horizontal for the DeValois
                tex_ori = 90

                # There is no external drift for the DeValois
                env_v = 0

            self.stimuli["gabor"].ori = tex_ori

        else:
            tex_ori = self.stimuli["gabor"].ori
            # tex_ori = 45

        # Duration of one move
        total_dur = int(duration * 2)

        # Initiate vars
        eye_sample = None
        gaze_data = []

        # Start recording frame intervals to make sure refresh rate is fine
        self.window.recordFrameIntervals = True

        # Get moving
        for fr in range(total_dur):

            if self.task == 'match':
                self.stimuli["gabor"].ori = tex_ori
                env_v = self.get_xy(self.params["v_env"], 90 - tex_ori)

            # Change the position in the first direction: drift up
            if fr < duration:
                self.stimuli["gabor"].pos += env_v
                self.stimuli["gabor"].phase -= tex_v

            # Second part of the path: drift down
            else:
                self.stimuli["gabor"].pos -= env_v
                self.stimuli["gabor"].phase += tex_v

            # Draw the gabor
            self.stimuli["gabor"].draw()

            # Use the mouse to change the length of the response line
            if mouse is not None:
                _, wheel_dy = mouse.getWheelRel()

                if self.task == "adjustment":
                    # if line.end[1] <= 0 and line.start[1] >= 0:
                    #     line.start += [0, wheel_dy.astype(int) * 3]
                    #     line.end -= [0, wheel_dy.astype(int) * 3]
                    # else:
                    #     line.start = [0, 0]
                    #     line.end = [0, 0]
                    if line.end[1] >= 0:
                        line.end = line.end + [0, wheel_dy * 3]
                    else:
                        line.end = [0, 5]

                    line.draw()

                elif self.task == "match":
                    if tex_ori > 90:
                        tex_ori = 90
                    elif tex_ori < 0:
                        tex_ori = 0
                    else:
                        tex_ori -= wheel_dy * 3

            # Check for fixation
            eye_sample, gaze = self.check_gaze(eye_sample, self.params["fix_pos_tracker"])
            gaze_data.append(gaze)

            # Show everything
            self.window.flip()

        # Get the frame interval measurements
        self.window.recordFrameIntervals = False
        recorded_frames = self.window.frameIntervals[-total_dur:]  # Only get the latest frame timings

        return gaze_data, recorded_frames
