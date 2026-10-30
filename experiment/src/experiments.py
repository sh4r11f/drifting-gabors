#!/usr/bin/env python
"""
created 07/22/20
@author DevXI

Implements basic neuroscience/psychophysics experiments as templates
"""
from psychopy import gui, data, core, monitors, info, logging, visual, event
from psychopy.iohub.devices import Computer
from psychopy.tools.monitorunittools import deg2pix
import pylink
from .EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

from pathlib import Path
import json
from abc import abstractmethod, ABC
import platform
from typing import Dict, List, Union, Tuple

import pandas as pd
import numpy as np


class BaseExperiment:
    """
    Template experiment with basic functionality.

    Mostly deals with setting up primitive objects and settings like subject info, system checks, files and paths,
    logging, etc.

    Basically every experiment needs the following attributes/functionalities:
        - Directories and files.
        - Settings that are specific to the experiment (metadata, conditions, etc.).
            An example of this can be found in docs/examples. Settings and parameters file should live under
            config/ directory of the experiment.
        - Getting user input.
        - A monitor and a window object (from psychopy).

    Parameters
    ----------
    debug : bool
        Debug mode for the experiment.

    root_dir : str or Path
        Home directory of the experiment.

    practice: bool
        Practice mode

    """
    def __init__(self, debug: bool, root_dir: Union[str, Path], practice: bool, **kwargs):

        self.EXP_TYPE = 'psychophysics'

        # Set provided parameters
        self.HOME = root_dir if isinstance(root_dir, Path) else Path(root_dir)

        # Load parameters and settings
        self.params = self.load_config('parameters')
        self.settings = self.load_config('settings')

        # Set parameters
        self.NAME = self.settings["name"]
        self.DEBUG = debug
        self.TASKS = self.settings["tasks"]
        self.practice = practice

        # Initiate runtime parameters
        self.session = None
        self.task = None  # the task should be set for each task
        self.run = None  # the number of a run for a single task
        self.exp_run = None  # the number of the run for a whole session
        self.n_runs = 0  # number of all the runs in a session
        for nums in self.TASKS.values():
            self.n_runs += nums["runs"]
        self.block = None  # the block number should be set on each block
        self.trial = None  # the trial number should be set on each trial

        # Experiment and subject info
        exp_info = kwargs.get("exp_info")
        sub_info = kwargs.get("sub_info")

        # Initiate the experiment parameters
        if exp_info is None:

            # Debug mode parameter
            if debug or practice:
                self.session = 0

            # if not in debug mode get session number from gui
            else:
                self.session = self.get_session_gui()

        # If info is provided just load and save it
        else:
            self.session = exp_info.get("session")

        # Check if there already is a subject
        if sub_info is None:
            self._SUBJECT = dict()

            # Debug mode parameters
            if debug or practice:
                self._SUBJECT["id"] = 0
                self._SUBJECT["init"] = "GG"
                self._SUBJECT["age"] = 0
                self._SUBJECT["gender"] = 0
                self._SUBJECT["hand"] = 0
                self._SUBJECT["vision"] = 0

            # if not in debug mode
            else:
                # register a new subject for the first session
                if self.session == 1:
                    self.register_sub_gui()

                # load an old subject for other sessions
                else:
                    self.load_sub_gui()

        # If subject info is already provided just save it
        else:
            self._SUBJECT = sub_info

        # Set up directories and files
        self.paths = self._dir_setup()

        # Initialize other params
        self.stimuli = dict()

        # Get monitor and make window
        self.monitor, self.window = self.make_display()

        # Set up logging
        self.files = dict()  # to hold file names for saving
        self.logs = dict()  # to hold log levels for logging
        self._init_logging()

        # Rounding numbers to these many decimal points
        self.round_err = 8

        # Clocks to keep track of timing. These can be reset at any time, except for the experiment clock.
        self.clocks = {
            "exp": core.Clock(),  # to keep track of the experiment
            "run": core.Clock(),  # to log files during a run
            "block": core.Clock(),  # to measure block duration
            "trial": core.Clock()  # to measure trial duration
        }

    @abstractmethod
    def setup_stimuli(self):
        """
        Creates and prepares visual stimuli that are going to be presented throughout the experiment.
        """
        pass

    @abstractmethod
    def setup_run(self, *args):
        """
        Pre-run code to prepare everything, including showing some messages and logging stuff or start talking to
        equipment.
        """
        pass

    @abstractmethod
    def setup_blocks(self, *args):
        """
        Code to run before each block of the experiment.
        """
        pass

    @abstractmethod
    def setup_trial(self, *args):
        """
        Code to run at the beginning of each trial
        """
        pass

    @abstractmethod
    def go(self, *args):
        """
        Prepares the block and run, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for the saccade.
        Cleans up the trial, block, and experiment, and saves everything.
        """
        pass

    @abstractmethod
    def save_run_data(self, *args):
        """
        Saves the experiment data after everything is finished.
        """
        pass

    @property
    def subject(self) -> dict:
        """
        Subject parameters for the experiment.

        Returns
        -------
        dict
        """
        return self._SUBJECT

    @subject.setter
    def subject(self, sub):
        """
        Setter method for subject parameters. It should include:
            - ID
            - Initials
            - Age
            - Gender
            - Handedness
            - Vision
        """
        self._SUBJECT = sub

    @staticmethod
    def make_dir(dir_name: str, dir_path: Union[str, Path]):
        """
        Tries to make a directory if it doesn't exist and handles exceptions

        Parameters
        ----------
        dir_name : Name of the directory
        dir_path : object with the directory address
        """
        # Check if the path exists
        if not dir_path.is_dir():
            logging.exp(f"Creating {dir_name} directory.")

            # try creating the folder
            try:
                dir_path.mkdir()
            except Exception as e:
                logging.error(f"Error in creating {dir_name} directory: {e}")
                raise RuntimeError(f"Error in creating {dir_name} directory: {e}")

    def recycle_trial(self, trials: np.ndarray, block: np.ndarray, recycle_col: int) -> np.ndarray:
        """
        Takes one trial and adds it to the bottom of a block of trials. It is used for situations where a trial is
        considered failed and needs to be repeated, but not immediately.

        Parameters
        ----------
        trials : np.ndarray
            A block of trials, which is essentially a copy of the "block" parameter below. The difference is that I
            change this copy often, but only use the original block for saving purposes (as done here). This array
            has trials as rows and experiment parameters as columns (features).

        block : np.ndarray
            The original array of experimental block. Only used here for saving whether a specific trial is recycled
            or not.

        recycle_col : int
            Column number for specifying the recycled feature.

        Returns
        -------
        np.ndarray
            A new block of trials, with the first trial removed from the 0 index and added to the bottom of the block
            (-1 index).
        """
        # Log recycling
        self._log_run(f"!!! Recycling trial {self.trial}.")
        self._log_run("-" * 40)

        # Save that the trial is recycled
        block[self.trial - 1, recycle_col] = 1

        # Add the first row to the bottom of the array
        new_trials = np.append(trials, [trials[0]], axis=0)

        # Pop the first burnt row
        new_trials = new_trials[1:]

        return new_trials

    def register_sub_cl(self):
        """
        Get subject info from the command line when GUI is unavailable.
        This is not fancy at all.
        """
        # get experiment
        self.session = int(input("Experiment session: "))

        # get the subject parameters from command line
        self._SUBJECT["id"] = int(input("Participant ID: "))
        self._SUBJECT["init"] = input("Participant Initials: ")
        self._SUBJECT["age"] = int(input("Participant Age: "))
        self._SUBJECT["gender"] = input("Participant Gender: ")
        self._SUBJECT["hand"] = input("Participant Handedness: ")
        self._SUBJECT["vision"] = input("Participant Vision: ")

    def get_session_gui(self):
        """
        Shows a gui to get the session number from user.
        """
        # Vars
        n_sessions = self.settings["sessions"]
        info_dict = {
            "Experiment": self.NAME,
            "Date": data.getDateStr(),
            "Session Number": list(range(1, n_sessions + 1)),
        }

        # Make the gui
        sess_gui = gui.DlgFromDict(
            title="Choose session",
            dictionary=info_dict,
            order=["Experiment", "Date", "Session Number"],
            fixed=["Experiment", "Date"],
            show=False
        )

        # Show it and get the response
        _sess = 0
        sess_gui.show()
        if sess_gui.OK:
            _sess = int(info_dict["Session Number"])
        else:
            self.end(win=False)

        # log
        logging.exp(f"Running session {_sess}")

        return _sess

    def register_sub_gui(self):
        """
        Shows a GUI to register a new subject for the experiment.
        """
        # Get previous info
        # Find the participants file
        pfile = self.HOME / "data" / "participants.tsv"

        # Load all the subjects
        _df = pd.read_csv(pfile, sep='\t')
        sub_ids = _df["ID"].tolist()  # compile IDs into a list
        sub_inits = _df["INITIALS"].tolist()

        # Initiate GUI
        sub_dlg = gui.Dlg(title="New Participant", labelButtonOK="Register", labelButtonCancel="Quit")

        # Info about the experiment
        sub_dlg.addText(text="Experiment", color="blue")
        sub_dlg.addFixedField("Title", self.NAME)
        sub_dlg.addFixedField("Date", str(data.getDateStr()))

        # Info about the subject
        sub_dlg.addText(text="Participant info", color="blue")
        sub_dlg.addField("ID", choices=list(range(1, 30)))
        sub_dlg.addField("Initials", tip="Uppercase letters (e.g. GOD)")
        sub_dlg.addField("Age", choices=list(range(18, 81)))
        sub_dlg.addField("Gender", choices=["Male", "Female", "Other"])
        sub_dlg.addField("Handedness", choices=["Right", "Left"])
        sub_dlg.addField("Vision", choices=["Normal", "Corrected"])

        # Show the dialogue
        sub_info = sub_dlg.show()

        # Get button press
        if sub_dlg.OK:

            # Experiment run and session for a new subject will be 1
            self.session = 1
            self.run = 1

            # Subject
            if sub_info[2] not in sub_ids:
                self._SUBJECT["id"] = sub_info[2]
            else:
                raise ValueError("A subject with this ID already exists.")

            if sub_info[3] not in sub_inits:
                self._SUBJECT["init"] = sub_info[3]
            else:
                raise ValueError("A subject with these initials already exists. Try changing them.")

            self._SUBJECT["age"] = int(sub_info[4])
            self._SUBJECT["gender"] = sub_info[5]
            self._SUBJECT["hand"] = sub_info[6]
            self._SUBJECT["vision"] = sub_info[7]

            # Log subject number
            logging.info(f"Subject ID: {sub_info[2]}")

        # If cancelled
        else:
            self.end(win=False)

    def load_sub_gui(self):
        """
        Loads info about subject IDs from data/participants.tsv and lets the user choose a subject for the session/run.
        """
        # Find the participants file
        pfile = self.HOME / "data" / "participants.tsv"

        # Load all the subjects
        _df = pd.read_csv(pfile, sep='\t')
        sub_ids = _df["ID"].tolist()  # compile IDs into a list
        sub_ids.sort()
        print(sub_ids)

        # Make a GUI
        sub_dlg = gui.Dlg(title="Load Participant", labelButtonOK="Load", labelButtonCancel="Quit")

        # Info about the experiment
        sub_dlg.addText(text="Experiment", color="blue")
        sub_dlg.addFixedField("Title", self.NAME)
        sub_dlg.addFixedField("Date", str(data.getDateStr()))

        # Info about the subject IDs
        sub_dlg.addField("Participant ID", choices=sub_ids)

        # Show
        _info = sub_dlg.show()

        # Get button press
        if sub_dlg.OK:

            # Convert types
            _id = int(_info[-1])
            _df["ID"] = _df["ID"].astype(int)

            # Select dataframe
            sub_df = _df[_df["ID"] == _id]

            # Load info about this subject
            self._SUBJECT["id"] = _id
            self._SUBJECT["init"] = sub_df["INITIALS"]
            self._SUBJECT["age"] = sub_df["AGE"]
            self._SUBJECT["gender"] = sub_df["GENDER"]
            self._SUBJECT["hand"] = sub_df["HAND"]
            self._SUBJECT["vision"] = sub_df["VISION"]

            # Log
            logging.info(f"Subject ID #{_id} loaded.")

        # If cancel was pressed, quit
        else:
            self.end(win=False)

    def make_display(self) -> Tuple:
        """
        Makes psychopy monitor and window objects to be used in the experiment. Relies on the name of the monitor
        specified in confing/parameters.json and monitor specification found in config/monitors.json

        Returns
        -------
        tuple
            mon : psychopy.monitors.Monitor

            win : psychopy.visual.Window
        """
        # Find the monitor name and specification
        mon_name = self.params["MONITOR"]
        mon_specs = self.load_config('monitors')[mon_name]
        # refresh rate is used everywhere to convert ms to frames. it's useful to have it as a parameter
        self.params["refresh_rate"] = mon_specs["refresh_rate"]

        # Make the monitor object and set width, distance, and resolution
        mon = monitors.Monitor(
            name=mon_name,
            width=mon_specs["size_cm"][0],
            distance=mon_specs["dist"],
            autoLog=False
        )
        mon.setSizePix(mon_specs["size_px"])

        # Gamma correction
        gamma_file = self.paths["config"] / f"{mon_name}_gamma_grid.npy"
        try:
            grid = np.load(str(gamma_file))
            mon.setLineariseMethod(1)  # (a + b**xx)**gamma
            mon.setGammaGrid(grid)

        except FileNotFoundError:
            print("No gamma grid file found. Running without gamma correction.")
            mon.setGamma(None)

        # Set variables for the window object based on Debug status of the experiment
        if self.DEBUG:
            mon_size = [1600, 900]
            mon_size = mon_specs["size_px"]
            full_screen = False
            # full_screen = True
            _gui = True
        else:
            mon_size = mon_specs["size_px"]
            full_screen = True
            _gui = False

        # Make the window object
        win = visual.Window(
            name='exp_window',
            monitor=mon,
            fullscr=full_screen,
            units='deg',  # units are always degrees by default
            size=mon_size,
            allowGUI=_gui,
            waitBlanking=True,
            color=self.params["BG_COLOR"],
            screen=0,  # the internal display is used by default
            autoLog=False
        )

        return mon, win

    def validate_all_frames(self, frame_intervals: Union[list, np.array]):
        """
        Checks the duration of all the provided frame intervals and measures the number of dropped frames.
        If there are too many dropped frames, throws a runtime error.

        Parameters
        ----------
        frame_intervals : list or numpy.array
            All the frame durations
        """
        # Get parameters
        mon_name = self.params["MONITOR"]
        mon_specs = self.load_config('monitors')[mon_name]
        rf = int(mon_specs["refresh_rate"])

        # Set thresholds
        thresh = .05  # threshold is at 5% of the frames
        n_thresh = len(frame_intervals) * thresh

        # Get the stats of the frames
        dur_sd = np.std(frame_intervals)
        # rf_cutoff = (1 / rf) * 1.10  # give 10% wiggle room
        rf_cutoff = (1 / rf) + (2 * dur_sd)  # frames greater than 2 std are dropped frames

        # find the dropped frames
        n_dropped = np.count_nonzero(np.array(frame_intervals) > rf_cutoff)

        # Throw an error if too many dropped frames
        if n_dropped > n_thresh and not self.DEBUG:
            # Log it
            self._log_run(f"!!! {n_dropped} dropped frames.")

            # Change fixation color to red
            self.stimuli["fix"].color = [1, -1, -1]
            self.stimuli["fix"].draw()
            self.window.flip()

            raise RuntimeError
        else:
            # Log
            self._log_run("Refresh rates OK.")

            # Report the number of dropped frames
            return n_dropped

    def get_system_status(self) -> str:
        """
        Check the status of the system, including:
            - Standard deviation of screen refresh rate
            - Amount of free RAM
            - Any interfering processes
            - Priority of the python program running

        Returns
        -------
        str
            Message about all system status to be displayed on the screen
        """
        # Somewhere to save the warnings to
        warns = ""

        # initial system check
        _run_info = info.RunTimeInfo(
            author=self.settings["author"],
            version=self.settings["version"],
            win=self.window,
            refreshTest='grating',
            userProcsDetailed=True,
            verbose=True
        )

        # Start logging
        logging.info("------------------------------------------------")
        logging.info("SYSTEM CHECKS")
        logging.info(f"Author: {_run_info['experimentAuthor']}")
        logging.info(f"Version: {_run_info['experimentAuthVersion']}")
        warns += "SYSTEM CHECKS\n"
        warns += "================================================================================================\n\n"

        # Test the refresh rate of the monitor
        rf = self.window.getActualFrameRate(nIdentical=20, nMaxFrames=100, nWarmUpFrames=10, threshold=.5)
        mon_name = self.params["MONITOR"]
        mon_specs = self.load_config('monitors')[mon_name]
        intended_rf = int(mon_specs["refresh_rate"])

        # in case could not get a good measurement
        if rf is None:
            logging.info("No identical frame refresh times were found.")
            warns += "(✘) No identical frame refresh times were found.\n\n"

        # if the measurement was successful
        else:

            # check if the measured refresh rate is the same as the one intended
            rf = np.round(rf).astype(int)

            if rf != intended_rf:
                logging.info(f"The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}.")
                warns += f"(✘) The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}.\n\n"

            # if everything checks out
            else:
                warns += f"(✓) Monitor refresh rate checks out.\n\n"

        # Check the monitor refresh time variability
        refresh_sd = _run_info["windowRefreshTimeSD_ms"]
        thresh = 0.20  # refresh rate standard deviation threshold
        if refresh_sd > thresh:

            # Log
            refresh_sd = np.round(refresh_sd, self.round_err)
            logging.info(f"Monitor refresh rate is too unreliable. SD: {refresh_sd}.")
            warns += f"(✘) Monitor refresh rate is too unreliable. SD: {refresh_sd}.\n\n"

            # Look for processes that might be causing the high refresh SD
            flagged = _run_info['systemUserProcFlagged']
            if len(flagged):
                s = "Flagged processes: "
                warns += "\t(✘) Flagged processes: "
                # app_set = {}
                # for i in range(len(_run_info['systemUserProcFlagged']) - 1):
                #     if _run_info['systemUserProcFlagged'][i][0] not in app_set:
                #         app_set.update({_run_info['systemUserProcFlagged'][i][0]: 0})
                # while len(app_set) != 0:
                for pr in np.unique(flagged):
                    # Log
                    # pr = app_set.popitem()[0]
                    s += f"{pr}, "
                    warns += f"{pr}, "
                warns += "\n\n"
                logging.info(s)
        else:
            warns += "(✓) Refresh threshold is normal.\n\n"

        # See if we have enough RAM
        ram_thresh = 1000
        if _run_info["systemMemFreeRAM"] < ram_thresh:
            logging.info(f"Only {round(_run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.")
            warns += f"(✘) Only {round(_run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.\n\n"
        else:
            warns += "(✓) RAM is OK.\n\n"

        # Raise the priority of the experiment for CPU
        # Check if it's Mac OS X (these methods don't run on that platform)
        if platform == "darwin":
            logging.info("Cannot raise the priority because you are on Mac OS X.")
            warns += "(✘) Cannot raise the priority because you are on Mac OS X.\n\n"
        else:
            try:
                Computer.setPriority("realtime", disable_gc=True)
                warns += "(✓) Realtime processing is set.\n\n"
            except Exception as e:
                logging.info(f"Error in elevating processing priority: {e}")
                warns += f"(✘) Error in elevating processing priority: {e}.\n\n"

        return warns

    def save_sub(self):
        """
        Saves the current newly registered subject to the data/participants.tsv file.
        """
        # Make a dictionary with current subject's parameters
        sub_info = {
            "ID": self._SUBJECT["id"],
            "INITIALS": self._SUBJECT["init"],
            "GENDER": self._SUBJECT["gender"],
            "AGE": self._SUBJECT["age"],
            "HAND": self._SUBJECT["hand"],
            "VISION": self._SUBJECT["vision"],
            f"SESSION_{self.session}": data.getDateStr()
        }

        # Read the participants file
        part_file = self.HOME / "data" / "participants.tsv"

        try:
            # Read the file
            part_df = pd.read_csv(part_file, sep='\t')

            # Check if this subject already exists
            _subs = part_df["INITIALS"].tolist()

            if self._SUBJECT["init"] in _subs:

                # If it's the first session and the subject already exists then it must be a duplicate name
                if int(self.session) == 1:
                    print("A participant with these initials already exist. Try changing the initials.")
                    logging.warning("A participant with these initials already exist. Try changing the initials.")

                # If the session is above one (and the subject exists) then just update the session number
                else:
                    logging.exp("adding to subject sessions.")
                    part_df.loc[part_df["INITIALS"] == self._SUBJECT["init"], f"SESSION_{self.session}"] = data.getDateStr()

            # If this is a new subject save the information in a new row
            else:
                # Add subject dictionary to the dataframe as a row
                logging.exp("saving new subject info.")
                part_df = part_df.append(sub_info, ignore_index=True)

        # Maybe this is the first run of the experiment ever
        except FileNotFoundError:
            # Make a new dataframe
            part_df = pd.DataFrame(sub_info)

        # Save back to the file again
        try:
            part_df.to_csv(part_file, sep='\t', index=False)
        except (FileExistsError, PermissionError) as e:
            print(f"Error in saving the participants file: {e}")
            logging.error(f"Error in saving the participants file: {e}")

    def save_exp_data(self):
        """ For saving experiment-level logs and data"""

        # self.logs["exp"].write(self.files["exp_log"])
        # self.logs["error"].write(self.files["error_log"])
        pass

    def clear_screen(self):
        """ clear up the PsychoPy window"""

        # Turn off all visual stimuli
        for stim in self.stimuli.values():
            stim.autoDraw = False

        # Reset background color
        self.window.color = self.params["BG_COLOR"]

        # Flip the window
        self.window.flip()

    def show_msg(self, text, wait_for_keypress=True):
        """ Show task instructions on screen"""

        # Make a message stimulus
        msg = visual.TextStim(
            self.window,
            text,
            font="Comic Sans MS",
            color=1,
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            wrapWidth=self.window.size[0] / 2,
            autoLog=False
        )

        # Change the window color to black
        self.window.color = -1

        # Clear the screen and show the message
        self.clear_screen()
        msg.draw()
        self.window.flip()

        # wait indefinitely, terminates upon any key press
        if wait_for_keypress:
            # Ctrl + C quits the experiment. Resume otherwise.
            pressed = event.waitKeys(modifiers=True)
            if (pressed[0] == 'c') and pressed[1]['ctrl']:
                self.end()
        else:
            core.wait(3)

        # Change the window color back
        self.window.color = self.params["BG_COLOR"]

        # Clear the screen again
        self.clear_screen()

    def log_section(self, name: str, part: str):
        """
        Times and logs the start or end of some section of the experiment.

        Parameters
        ----------
        name : str
            Name of the section. Should be one of 'trial', 'block', 'run'

        part : str
            start or end of a section
        """
        # Run/block/trial params
        _name = name.lower()
        if _name == 'run':
            n = self.run
            sep = '*' * 10
        elif _name == 'block':
            n = self.block
            sep = '=' * 60
        elif _name == 'trial':
            n = self.trial
            sep = '-' * 40
        else:
            raise ValueError("Invalid name.")

        # Clear event buffer
        event.clearEvents()

        # Get the duration and log
        if part == 'start':
            self._log_run(sep)
            self._log_run(f"{name} {n} started: {self.task} task")
        elif part == 'end':
            _mins = int(self.clocks[_name].getTime() // 60)
            _secs = int(self.clocks[_name].getTime() % 60)
            self._log_run(f"{name} {n} ended.")
            self._log_run(f"Duration {_mins} minutes and {_secs} seconds.")
            self._log_run(sep)
        else:
            raise ValueError("Invalid part")

    def end(self, win: bool = True):
        """
        Closes and ends the experiment abruptly.

        Parameters
        ----------
        win : bool
            Whether there is a window to close or not.
        """
        # Log
        logging.info("Quitting the experiment.")

        # Close the window
        if win:
            self.window.close()

        # Quit psychopy and system program
        core.quit()

    def force_quit(self, key_press: str = 'escape'):
        """
        Quits the experiment during runtime if a key (default 'space') is pressed.

        Parameters
        ----------
        key_press : str
            keyboard button to press to quit the experiment.
        """
        # Get the keypress from user
        pressed = event.getKeys()

        # Check if it's the quit key
        if key_press in pressed:
            logging.warning("Force quitting.")
            self.end()

    def _dir_setup(self) -> Dict:
        """
        Sets up the directories and paths that are important in the experiment. The structure roughly follows the
        BIDS convention.

        Example: data/FIPS/sub-01/staircase/
                 data/FIPS2/sub-02/perceptual/
                 ...
        Returns
        -------
        dict
            data, sub, and task keys and the corresponding Path objects as values.
        """
        # Dictionary for all the directories
        _dirs = dict()

        # data directory
        data_dir = self.HOME / "data" / f"v{self.settings['version']}"
        self.make_dir("data", data_dir)
        _dirs["data"] = data_dir

        # subject directory
        sub_dir = data_dir / f"sub-{self._SUBJECT['id']:02}"
        self.make_dir("subject", sub_dir)
        _dirs["sub"] = sub_dir

        # task directory
        for task in self.TASKS.keys():
            task_dir = sub_dir / task
            self.make_dir(task, task_dir)
            _dirs[task] = task_dir

        # config directory
        _dirs["config"] = self.HOME / "config"

        return _dirs

    def _file_setup(self):
        """
        Sets up files that are going to be saved at the end of the experiment. Naming roughly follows the BIDS
        convention. The files are:
            - Run files: extensions are indicated at the time of saving, but the general run file name is always the
                         same.
                         Example: sub-01_ses-01_run-01_task-saccade_exp-FIPS.<extension>
        Returns
        -------
        dict
            run as keys and the corresponding Path objects as values.
        """
        # Run file name
        run_file = self.paths[self.task] / f"sub-{self._SUBJECT['id']:02d}_ses-{self.session:02d}" \
                                           f"_run-{self.run:02d}_task-{self.task}_exp-{self.NAME}" \
                                           f"_v{self.settings['version']}"
        self.files["run"] = run_file

        # Log filename and data
        log_file = str(run_file) + '.log'
        self.files["run_log"] = log_file
        log_data = logging.LogFile(log_file, filemode='w', level=99)
        self.logs["task"] = log_data

    def load_config(self, file_name: str):
        """
        Reads and saves parameters specified in a json file under the config directory, e.g. config/params.json

        Parameters
        ----------
        file_name : str
            Name of the file to be loaded.

        Returns
        -------
        dict
            The .json file is returned as a python dictionary.
        """
        # Find the parameters file
        params_file = self.HOME / "config" / f"{file_name}.json"

        # Check its status and read it
        if params_file.is_file():
            try:
                with open(str(params_file)) as pf:
                    _params = json.load(pf)
            # for corrupted files or other issues
            except IOError as e:
                logging.error(f"Unable to open the {file_name} file: {e}")
                raise f"Unable to open the {file_name} file: {e}"
        else:
            raise FileNotFoundError(f"{str(params_file)} does not exist")

        return _params

    def _init_logging(self):
        """
        Generates the log file to populate with messages throughout the experiment.

        Returns
        -------
        psychopy.logging.LogFile
        """
        # Make a clock to use for timestamps
        log_clock = core.Clock()
        logging.setDefaultClock(log_clock)  # use that clock for logging

        # Log files
        exp_file = self.paths["sub"] / \
                   f"sub-{self._SUBJECT['id']:02d}_ses-{self.session}_exp-{self.NAME}_v{self.settings['version']}-info.log"
        error_file = self.paths["sub"] / \
                     f"sub-{self._SUBJECT['id']:02d}_ses-{self.session}_exp-{self.NAME}_v{self.settings['version']}-errors.log"

        # Initialize the logfile. Setting the file to the one created with subject's ID.
        exp_log = logging.LogFile(str(exp_file), filemode='a', level=logging.INFO)
        error_log = logging.LogFile(str(error_file), filemode='a', level=logging.ERROR)

        # Control what gets printed to the console
        if self.DEBUG:
            logging.console.setLevel(logging.DEBUG)  # log everything
        else:
            logging.console.setLevel(logging.ERROR)  # log the minimum amount of information

        # Add the tasks levels and placeholder for files
        # logging.addLevel(23, "task")
        # self.logs["task"] = (23, )
        # for t, task in enumerate(self.TASKS.keys()):
        #     log_level = 41 + t
        #     logging.addLevel(log_level, task)
        #     self.logs[task] = [log_level, '']

        # Save the logs
        self.logs["exp"] = exp_log
        self.logs["error"] = error_log
        self.files["exp_log"] = exp_file
        self.files["error_log"] = error_log

        # First log
        logging.exp('************************************************************************************************')
        logging.exp("Experiment started.")
        logging.exp(f"Version: {self.settings['version']}")

    def _log_run(self, log_msg: str):
        """
        Logs a message for a specific run

        Parameters
        ----------
        log_msg : str
            Log message
        """
        logging.log(msg=log_msg, level=99, t=self.clocks["run"].getTime())

    def ms2fr(self, duration: float):
        """ Converts durations from ms to display frames"""
        return np.ceil(duration * self.params["refresh_rate"] / 1000).astype(int)


class EyeTracking(BaseExperiment, ABC):
    """
    Main template for eye tracking experiments running on the Eyelink1000 Plus. It mostly adds functionality for
    communicating and controlling the Eyelink. Other than that, it pretty much implements all its parents methods.

    Most of the code in the methods are from examples in Pylink docs.

    Parameters
    ----------
    debug : bool
        Debug mode for the experiment.

    dummy_tracker : bool
        Debug mode for the eye tracker.

    root_dir : str or Path
        Home directory of the experiment.
    """

    def __init__(
            self,
            debug: bool,
            root_dir: Union[str, Path],
            practice,
            dummy_tracker: bool,
            **kwargs
    ):
        super().__init__(debug, root_dir, practice, **kwargs)

        self.EXP_TYPE = 'eyetracking'

        # Set the dummy mode parameter for debugging the tracker
        self.DUMMY = dummy_tracker

        # Mouse instead of the tracker
        if self.DUMMY:
            self.window.mouseVisible = True

        # Eye tracking experiments always use pixel measurements.
        # Also changing the default background color to black so there's less light affecting the pupil. T
        # his can be modified anytime during actual experiments.
        # IF THIS IS NOT DONE EARLY, the eyetracker will override it and we get a grey background always!
        self.window.units = 'pix'
        self.window.color = self.params["BG_COLOR"]

        # Add Eyelink-specific .edf files' paths and names
        self.files["edf_host"] = f"{self.NAME}{self._SUBJECT['id']:02d}.edf"

        # Initialize the eye tracker
        self.el_tracker = pylink.getEYELINK()

        # Connect and configure the tracker
        self.connect_tracker()
        self.open_tracker_file()
        self.configure_tracker()
        self.setup_calibration()

    def connect_tracker(self):
        """
        Connects to an Eyelink 1000 using its python API (pylink).
        """
        # Initiate connection (only if not in dummy mode)
        if self.DUMMY:
            self.el_tracker = pylink.EyeLink(None)
        else:
            # The IP address is 100.1.1.1
            try:
                self.el_tracker = pylink.EyeLink("100.1.1.1")
                logging.exp("Tracker connected.")
            except RuntimeError as err:
                logging.error(f'Error in connecting to the Eyelink: {err}')
                self.end()

    def open_tracker_file(self):
        """
        Creates an edf file on the host Eyelink computer to record the data
        """
        # Open an EDF data file on the Host PC
        try:
            self.el_tracker.openDataFile(self.files["edf_host"])
            logging.exp("Tracker file created.")
            # Add a header text to the EDF file to identify the current experiment name
            self.el_tracker.sendCommand(f"add_file_preamble_text {self.NAME}")

        except RuntimeError as err:
            logging.error('Error in creating the EDF file:', err)

            # close the link if we have one open
            if self.el_tracker.isConnected():
                self.el_tracker.close()

            # quit the experiment
            self.end()

    def configure_tracker(self):
        """
        Configure the Eyelink 1000 connection to track the specified events and have the appropriate setup
        e.g., sample rate and calibration type.
        """
        # Put the tracker in offline mode before we change tracking parameters
        self.el_tracker.setOfflineMode()

        # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000, 5-EyeLink 1000 Plus, 6-Portable DUO
        eyelink_ver = 0  # set version to 0, in case running in Dummy mode
        if not self.DUMMY:
            vstr = self.el_tracker.getTrackerVersionString()
            eyelink_ver = int(vstr.split()[-1].split('.')[0])
            # print out some version info in the shell
            msg = f'Running experiment on {vstr}, version {eyelink_ver}'
            logging.info(msg)

        # File and Link data control
        # what eye events to save in the EDF file, include everything by default
        file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'

        # what eye events to make available over the link, include everything by default
        link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

        # what sample data to save in the EDF data file and to make available
        # over the link, include the 'HTARGET' flag to save head target sticker
        # data for supported eye trackers
        if eyelink_ver > 3:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
        else:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'

        # Send the parameters to the tracker
        self.el_tracker.sendCommand(f"file_event_filter = {file_event_flags}")
        self.el_tracker.sendCommand(f"file_sample_data = {file_sample_flags}")
        self.el_tracker.sendCommand(f"link_event_filter = {link_event_flags}")
        self.el_tracker.sendCommand(f"link_sample_data = {link_sample_flags}")

        # Optional tracking parameters
        # Sample rate, 250, 500, 1000, or 2000, check your tracker specification
        self.el_tracker.sendCommand("sample_rate 1000")

        # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
        if self.DEBUG:
            self.el_tracker.sendCommand("calibration_type = HV3")
        else:
            self.el_tracker.sendCommand("calibration_type = HV5")

        # Log
        logging.exp("Tracker configured.")

    def setup_calibration(self):
        """
        Prepares the Eyelink for calibration procedure.
        This includes setting the correct coordinates for the tracker and creating a graphics environment for
        calibration.
        """
        # get the native screen resolution used by PsychoPy
        scn_width, scn_height = self.window.size

        # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
        # see the EyeLink Installation Guide, "Customizing Screen Settings"
        el_coords = f"screen_pixel_coords = 0 0 {scn_width - 1} {scn_height - 1}"
        self.el_tracker.sendCommand(el_coords)

        # Write a DISPLAY_COORDS message to the EDF file
        # Data Viewer needs this piece of info for proper visualization, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        dv_coords = f"DISPLAY_COORDS  0 0 {scn_width - 1} {scn_height - 1}"
        self.el_tracker.sendMessage(dv_coords)

        # Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.el_tracker, self.window)
        msg = f"Version number of the CoreGraphics library: {genv}"
        logging.info(msg)

        # Set background and foreground colors for the calibration target
        # in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
        foreground_color = (1, 1, 1)
        background_color = self.params["BG_COLOR"]
        # background_color = (-1, -1, -1)
        genv.setCalibrationColors(foreground_color, background_color)

        # Set up the calibration target
        # Use the default calibration target ('circle')
        genv.setTargetType('circle')

        # Configure the size of the calibration target (in pixels)
        # this option applies only to "circle" and "spiral" targets
        genv.setTargetSize(24)

        # Beeps to play during calibration, validation and drift correction
        # parameters: target, good, error
        #     target -- sound to play when target moves
        #     good -- sound to play on successful operation
        #     error -- sound to play on failure or interruption
        # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
        genv.setCalibrationSounds('', '', '')

        # Request Pylink to use the PsychoPy window we opened above for calibration
        pylink.openGraphicsEx(genv)

    def get_eye(self) -> int:
        """
        Checks if the tracker can find an eye, and whether that eye is the same as the one specified in the
        parameters file of the experiment.

        Returns
        -------
        int
            The eye used for the experiment: 0 is left and 1 is right.
        """
        # Get the eye info from tracker
        eye_used = self.el_tracker.eyeAvailable()

        # Get the eye info from parameters file
        eye_intend = int(self.params["EYE"])

        # Check the right eye
        if eye_used == 1 and eye_used == eye_intend:
            logging.info("Tracker using the right eye.")
            self.el_tracker.sendMessage("EYE_USED 1 RIGHT")

        # Check the left eye or detecting both eyes
        elif eye_used == 0 or eye_used == 2:
            eye_used = 0  # use the left eye only
            logging.info("Tracker using the left eye.")
            self.el_tracker.sendMessage("EYE_USED 0 LEFT")

        # If there is invalid eye
        else:
            msg = 'Error in getting eye information. Recalibrate the tracker.'
            logging.warning(msg)
            self.show_msg(msg)

        return eye_used

    def check_gaze(self, old_sample, fix_pos: Union[List, Tuple]):
        """
        Function to monitor gaze on a static point, like a fixation cross.

        Parameters
        ----------
        old_sample : instance of Sample type
            Previous eye position sample

        fix_pos : list or tuple
            The position of the fixation spot on the screen that needs to be checked for gaze info

        Returns
        -------
        tuple
            eye sample: instance of Sample type
            fixating: bool value of fixation check
        """
        # Get the eye information
        eye_used = self.el_tracker.eyeAvailable()

        # Define hit region
        region = deg2pix(float(self.params["GAZE_REGION"]), self.monitor)
        fix_x, fix_y = fix_pos
        g_x = None
        g_y = None

        # Have some faith!
        fixating = True

        # Do we have a sample in the sample buffer?
        # and does it differ from the one we've seen before?
        new_sample = self.el_tracker.getNewestSample()
        if new_sample is not None:
            if old_sample is not None:
                if new_sample.getTime() != old_sample.getTime():

                    # check if the new sample has data for the eye
                    # currently being tracked; if so, we retrieve the current
                    # gaze position and PPD (how many pixels correspond to 1
                    # deg of visual angle, at the current gaze position)
                    if eye_used == 1 and new_sample.isRightSample():
                        g_x, g_y = new_sample.getRightEye().getGaze()
                    if eye_used == 0 and new_sample.isLeftSample():
                        g_x, g_y = new_sample.getLeftEye().getGaze()

                    # See if the current gaze position is in a region around the screen centered
                    # if np.fabs(g_x - fix_x) < region and np.fabs(g_y - fix_y) < region:
                    if self.get_hypot([g_x, g_y], [fix_x, fix_y]) < region:
                        fixating = True
                    else:  # gaze outside the hit region
                        fixating = False

        return new_sample, fixating

    def wait_for_gaze(self, fix_pos: Union[List, Tuple]):
        """
        Runs a while loop and keeps it running until gaze on some region is established.

        Parameters
        ----------
        fix_pos : tuple or list
            Coordinates of the fixation spot to be monitored
        """
        # Clear cached PsychoPy events
        event.clearEvents()

        # Samples
        old_sample = None

        # Trigger status
        if self.DUMMY:
            trigger_fired = True
        else:
            trigger_fired = False

        # Fire the trigger following a 500-ms gaze
        min_dur = .5

        # The position of the gaze
        gaze_start = -1

        # Get the eye information
        eye_used = self.el_tracker.eyeAvailable()

        # Define the hit region
        in_hit_region = False
        region = deg2pix(float(self.params["GAZE_REGION"]), self.monitor)
        fix_x, fix_y = fix_pos

        # Running the gaze loop
        while not trigger_fired:

            # Do we have a sample in the sample buffer?
            # and does it differ from the one we've seen before?
            new_sample = self.el_tracker.getNewestSample()

            if new_sample is not None:
                if old_sample is not None:
                    if new_sample.getTime() != old_sample.getTime():

                        # check if the new sample has data for the eye
                        # currently being tracked; if so, we retrieve the current
                        # gaze position and PPD (how many pixels correspond to 1
                        # deg of visual angle, at the current gaze position)
                        if eye_used == 1 and new_sample.isRightSample():
                            g_x, g_y = new_sample.getRightEye().getGaze()
                        if eye_used == 0 and new_sample.isLeftSample():
                            g_x, g_y = new_sample.getLeftEye().getGaze()

                        # break the while loop if the current gaze position in a region
                        # if np.fabs(g_x - fix_x) < region and np.fabs(g_y - fix_y) < region:
                        if self.get_hypot([g_x, g_y], [fix_x, fix_y]) < region:
                            # record gaze start time
                            if not in_hit_region:
                                if gaze_start == -1:
                                    gaze_start = core.getTime()
                                    in_hit_region = True

                            # check the gaze duration and fire
                            if in_hit_region:
                                gaze_dur = core.getTime() - gaze_start
                                if gaze_dur > min_dur:
                                    trigger_fired = True
                                    self._log_run("Fixation detected.")
                                    self.el_tracker.sendMessage('gaze_confirmed')

                        # gaze outside the hit region, reset variables
                        else:
                            in_hit_region = False
                            gaze_start = -1

                # update the "old_sample"
                old_sample = new_sample

    def check_saccade(self, t_onset: float, target_pos: Union[List, Tuple], t_thresh: int = 500) -> Dict:
        """
        Checks if a saccade is made to a target position.

        Parameters
        ----------
        t_onset : float
            time of the onset of the target

        target_pos : list or tuple
            Coordinates of the target (0 centered around the center of the screen)

        t_thresh : int
            Duration to wait for saccade in ms

        Returns
        -------
        dict
            Information about the saccade:
                - status: made successfully or not
                - t_start: time of initiation
                - t_end: time of landing
                - amp_x: amplitude in X direction
                - amp_y: amplitude in Y direction
                - start_x: X of start position
                - start_y: Y of start position
                - end_x: X of landing
                - end_y: Y of landing
                - rt: saccadic reaction time
                - error_x: distance from the target in X
                - error_y: distance from the target in Y
                - accuracy: whether saccade was close to the target or not

        """
        # Set up the coordinates
        target_x, target_y = target_pos
        target_x = target_x + self.window.size[0] / 2.0
        target_y = target_y + self.window.size[1] / 2.0

        # Saccade checking vars
        if self.DUMMY:
            got_sac = True
            _info = {"status": 1}
        else:
            got_sac = False
            _info = None

        # sac_start_time = -1
        # srt = -1  # initialize a variable to store saccadic reaction time (SRT)
        # land_err = -1  # landing error of the saccade
        # acc = 0  # hit the correct region or not
        acc_region = deg2pix(1, self.monitor)
        amp_thresh = 3

        # Clear all cached events if there are any
        event.clearEvents()

        # Wait for the saccade
        while not got_sac:

            # wait for a maximum of 1000 milliseconds for the saccade to occur
            if self.el_tracker.trackerTime() - t_onset >= t_thresh:
                self.el_tracker.sendMessage('saccade_time_out')
                self.el_tracker.sendMessage('SACCADE_FAIL')
                self._log_run("Saccade timeout.")
                break

            # grab the events in the buffer, for more details,
            # see the example script "link_event.py"
            eye_ev = self.el_tracker.getNextData()
            if (eye_ev is not None) and (eye_ev == pylink.ENDSACC):
                eye_dat = self.el_tracker.getFloatData()
                if eye_dat.getEye() == self.params["EYE"]:
                    sac_amp = eye_dat.getAmplitude()  # amplitude
                    sac_start_time = eye_dat.getStartTime()  # onset time
                    sac_end_time = eye_dat.getEndTime()  # offset time
                    sac_start_pos = eye_dat.getStartGaze()  # start position
                    sac_end_pos = eye_dat.getEndGaze()  # end position

                    # ignore saccades occurred before target onset
                    if sac_start_time <= t_onset:
                        sac_start_time = -1
                        pass

                    # A correct saccade was initiated
                    elif np.hypot(sac_amp[0], sac_amp[1]) > amp_thresh:

                        # Saccade status
                        got_sac = True
                        self.el_tracker.sendMessage("SACCADE_START")

                        # log a message to mark the time at which a saccadic
                        # response occurred; note that, here we are detecting a
                        # saccade END event; the saccade actually occurred some
                        # msecs ago. The following message has an additional
                        # time offset, so Data Viewer knows when exactly the
                        # "saccade_resp" event actually happened
                        t_offset = int(self.el_tracker.trackerTime() - sac_start_time)
                        sac_response_msg = f'{t_offset} saccade_resp'
                        self.el_tracker.sendMessage(sac_response_msg)
                        srt = sac_start_time - t_onset

                        # count as a correct response if the saccade lands in
                        # the square shaped hit region
                        land_err = [sac_end_pos[0] - target_x, sac_end_pos[1] - target_y]

                        # Check hit region and set accuracy
                        # if np.fabs(land_err[0]) < acc_region and np.fabs(land_err[1]) < acc_region:
                        if self.get_hypot(sac_end_pos, [target_x, target_y]) < acc_region:
                            acc = 1
                        else:
                            acc = 0

                        # Save everything
                        _info = {
                            "status": got_sac,
                            "t_start": sac_start_time,
                            "t_end": sac_end_time,
                            "amp_x": sac_amp[0],
                            "amp_y": sac_amp[1],
                            "start_x": sac_start_pos[0],
                            "start_y": sac_start_pos[1],
                            "end_x": sac_end_pos[0],
                            "end_y": sac_end_pos[1],
                            "rt": srt,
                            "error_x": land_err[0],
                            "error_y": land_err[1],
                            "accuracy": acc
                        }
        return _info

    def run_calibration(self):
        """Calibrate the Eyelink 1000"""

        # A guiding message
        msg = 'Press ENTER and C to recalibrate the tracker.\n\n' \
              'Once the calibration is done, press ENTER and O to resume the experiment.'
        self.show_msg(msg)

        # Initiate calibration
        try:
            self.el_tracker.doTrackerSetup()
            self._log_run("Tracker calibrated.")
            self.el_tracker.sendMessage('tracker_calibrated')

        except (RuntimeError, AttributeError) as err:
            self._log_run(f"Tracker not calibrated: {err}")
            self.el_tracker.exitCalibration()

    def start_recording(self):
        """Initiate recording of eye data by the tracker"""

        try:
            # arguments: sample_to_file, events_to_file, sample_over_link,
            # event_over_link (1-yes, 0-no)
            self.el_tracker.startRecording(1, 1, 1, 1)
            self._log_run("Tracker recording.")
            self.el_tracker.sendMessage("recording")

        except RuntimeError as err:
            self._log_run(f"Tracker not recording: {err}")
            self.el_tracker.sendMessage("not_recording")
            return pylink.TRIAL_ERROR

    def check_connection(self):
        """Checks if the tracker is still alive"""

        # Get the status
        error = self.el_tracker.isConnected()  # returns 1 if connected, 0 if not connected, -1 if simulated

        # Check it only in the actual running mode because simulated tracker returns -1 in isConnected()
        if not self.DUMMY:

            # For some reason TRIAL_OK is 0 in pylink
            if error is pylink.TRIAL_OK:
                # Log the disconnect
                self.el_tracker.sendMessage('tracker_disconnected')
                logging.error(f"Tracker disconnected. Error: {error}")
                self.end()

    def drift_correction(self, fix_pos):
        """
        Performs drift correction with the Eyelink 1000

        Parameters
        ----------
        fix_pos : list or tuple
            Coordinates of the fixation spot
        """
        # unpack the fixation
        fix_x, fix_y = fix_pos

        # Apply the drift correction
        self.el_tracker.sendCommand('driftcorrect_cr_disable = OFF')
        self.el_tracker.sendCommand(f'online_dcorr_refposn {int(self.window.size[0] / 2)},{int(self.window.size[1] / 2)}')
        self.el_tracker.sendCommand('online_dcorr_button = ON')
        self.el_tracker.sendCommand('normal_click_dcorr = OFF')

        # Drift correction for the eye tracker
        while not self.DUMMY:

            # terminate the task if no longer connected to the tracker or
            # user pressed Ctrl-C to terminate the task
            if (not self.el_tracker.isConnected()) or self.el_tracker.breakPressed():
                self.el_tracker.sendMessage('experiment_aborted')
                self.terminate_tracker()
                self.end()
                raise pylink.ABORT_EXPT

            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                error = self.el_tracker.doDriftCorrect(int(fix_x), int(fix_y), 1, 1)
                # break following a success drift-check
                if error is not pylink.ESC_KEY:
                    self._log_run("Tracker drift corrected.")
                    self.el_tracker.sendMessage("drift_corrected")
                    break
            except:
                self.el_tracker.sendMessage('drift_correction_failed')
                self._log_run("Tracker not drift corrected.")

    def terminate_tracker(self, end: bool = False):
        """
        Terminate the task gracefully and retrieve the EDF data file

        Parameters
        ----------
        end : whether shutdown psychopy and system program or not

        """
        # Check if there is an active connection
        if self.el_tracker.isConnected() or self.DUMMY:

            # Terminate the current trial first if the task terminated prematurely
            error = self.el_tracker.isRecording()
            if error:
                self.el_tracker.sendMessage("abort_trial")
                self.abort_trial()

            # Put tracker in Offline mode
            self.el_tracker.setOfflineMode()

            # Clear the Host PC screen and wait for 500 ms
            self.el_tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)

            # Close the edf data file on the Host
            self.el_tracker.closeDataFile()

            # Show a file transfer message on the screen
            msg = 'EDF data is transferring from EyeLink Host PC...'
            self.show_msg(msg, wait_for_keypress=False)

            # Download the EDF data file from the Host PC to a local data folder
            # parameters: source_file_on_the_host, destination_file_on_local_drive
            try:
                self.el_tracker.receiveDataFile(self.files["edf_host"], self.files["edf_local"])
            except RuntimeError as error:
                logging.error('Error in downloading the EDF file:', error)

            # Close the link to the tracker.
            if end:
                pylink.closeGraphics()
                self.el_tracker.close()

    def abort_trial(self):
        """Ends recording and recycles the trial"""

        # Stop recording
        if self.el_tracker.isRecording():
            # add 100 ms to catch final trial events
            pylink.pumpDelay(100)
            self.el_tracker.stopRecording()

        # clear the screen
        self.clear_screen()

        # Send a message to clear the Data Viewer screen
        bgcolor_RGB = (116, 116, 116)
        self.el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

        # send a message to mark trial end
        # self.el_tracker.sendMessage(f'TRIAL_RESULT {pylink.TRIAL_ERROR}')
        self.el_tracker.sendMessage(f'TRIAL_FAIL')

        # Log
        self._log_run("!!! Trial aborted.")

    def validate_all_gaze(self, gaze_arr: Union[List, np.ndarray], period: str):
        """
        Checks all gaze reports and throws an error if there is any gaze deviation.
        The runtime error should be caught later and handled by recycling the trial.

        Parameters
        ----------
        gaze_arr : list or array
            The array of True or False for each gaze time point.

        period : str
            Name of the period where the validation is being done for.
        """
        # Check all the timepoints
        if not all(gaze_arr):

            # Log
            self._log_run(f"Gaze fail in trial {self.trial}, {period} period.")
            self.el_tracker.sendMessage(f"{period.upper()}_FAIL")

            # Change fixation color to red
            self.stimuli["fix"].color = [1, -1, -1]
            self.stimuli["fix"].draw()
            self.window.flip()

            # Throw an error
            raise RuntimeError

    def setup_run(self, inst_msg: str):
        """
        Initiates clocks, shows beginning message, and logs start of a run.

        Parameters
        ----------
        inst_msg : str
            A text that has instructions for start of the experiment
        """
        # Setup files for logging and saving this run
        self._file_setup()
        self.files["edf_local"] = str(self.files["run"]) + '.edf'
        # initiate file on the tracker
        self.open_tracker_file()

        # Clock
        self.clocks["run"].reset()

        # Change the log level task name for this run
        logging.addLevel(99, self.task)

        # Log the start of the run
        self.log_section('Run', 'start')
        self.el_tracker.sendMessage(f"TASK_START")
        self.el_tracker.sendMessage(f"TASKID_{self.task}")
        self.el_tracker.sendMessage(f"RUN_START")
        self.el_tracker.sendMessage(f"RUNID_{self.exp_run}")

        # Show instruction message before starting
        self.show_msg(inst_msg)

    def setup_block(self, calib=True):
        """
        Sets up an experimental block. Shows a text message and initiates and calibrates the eye tracker.
        """
        # Timing
        self.clocks["block"].reset()

        # Tracker initialization
        self.el_tracker.setOfflineMode()
        if calib:
            self.run_calibration()

        # clear the host screen
        self.el_tracker.sendCommand('clear_screen 0')

        # log the beginning of block
        self.log_section('Block', 'start')
        self.el_tracker.sendMessage(f"BLOCK_START")
        self.el_tracker.sendMessage(f"BLOCKID_{self.block}")

    def trial_cleanup(self):
        """
        Turns off the stimuli and stop the tracker to clean up the trial.
        """
        # Turn off all stimuli
        self.clear_screen()
        # clear the host screen too
        self.el_tracker.sendCommand('clear_screen 0')

        # Stop recording frame interval
        self.window.recordFrameIntervals = False

        # Stop recording; add 100 msec to catch final events before stopping
        pylink.pumpDelay(100)
        self.el_tracker.stopRecording()

    def block_cleanup(self):
        """ Logs the end of a block and shows message about the remaining blocks"""

        # Log the end of the block
        self.log_section("Block", "end")
        self.el_tracker.sendMessage(f"BLOCKID_{self.block}")
        self.el_tracker.sendMessage(f"BLOCK_END")

        # Turn everything off
        self.clear_screen()
        # clear the host screen
        self.el_tracker.sendCommand('clear_screen 0')

        # Get the number of all blocks for this task
        n_blocks = int(self.TASKS[self.task]["blocks"])

        # Show a message between the blocks
        remain_blocks = n_blocks - self.block
        btw_blocks_msg = f"You finished block number {self.block}.\n\n" \
                         f"{remain_blocks} more block(s) remaining in this run.\n\n" \
                         "When you are ready, press the SPACEBAR to continue to calibration."
        self.show_msg(btw_blocks_msg)

    def run_cleanup(self, remain_runs):
        """ Housekeeping at the end of a run"""

        # Log
        self.log_section("Run", "end")
        self.el_tracker.sendMessage(f"RUNID_{self.exp_run}")
        self.el_tracker.sendMessage(f"RUN_END")
        self.el_tracker.sendMessage(f"TASKID_{self.task}")
        self.el_tracker.sendMessage(f"TASK_END")

        # Turn everything off
        self.clear_screen()
        self.el_tracker.sendCommand('clear_screen 0')

        # Show an info message
        btw_msg = f"\U00002705 You finished run number {self.run} of the experiment \U00002705\n\n"
        if not self.practice:
            btw_msg += f"There are {remain_runs} run remaining!\n\n"
            btw_msg += "When you are ready, press the SPACEBAR to continue to calibration!"

        end_msg = "\U0000235F\U0000235F\U0000235F \n\n"
        end_msg += "Experiment ended! Thank you for your participation :-)\n\n"
        end_msg += "\U0000235F\U0000235F\U0000235F"

        # Terminate the tracker on the last run
        if remain_runs:
            self.show_msg(btw_msg, wait_for_keypress=False)
            self.terminate_tracker(end=False)
        else:
            self.show_msg(end_msg, wait_for_keypress=False)
            self.terminate_tracker(end=True)

    def save_run_data(self, blocks: list, col_names: list):
        """
        Saves all the blocks and logs to disk.

        Parameters
        ----------
        blocks : list
            All numpy arrays that contain the experiment data

        col_names : list
            Column names for each trial block

        """
        # Compile all blocks into one giant array
        exp_data = np.concatenate(blocks)

        # Make a data frame
        pd_file = str(self.files["run"]) + '.csv'
        df = pd.DataFrame(exp_data, columns=col_names)

        # Save it
        df.to_csv(pd_file, sep=',', index=False)

        # Save the log files
        # logs are in a tuple (or list) with the first item being the log level and the second one the log file
        # self.logs["task"].write(self.files["run_log"])
        # get rid of this log file
        logging.flush()
        logging.root.removeTarget(self.logs["task"])

    @staticmethod
    def get_hypot(target, fix):

        fx, fy = np.fabs(fix)
        tx, ty = np.fabs(target)

        return np.hypot((tx - fx), (ty - fy))
