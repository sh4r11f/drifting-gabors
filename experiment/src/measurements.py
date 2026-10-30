#!usr/bin/env python
"""
Created at 3/14/22
@author: devxl

Provides classes for making measurements of the monitor and stimuli.
"""
from abc import ABC

import pandas as pd
from .experiments import BaseExperiment
from .fips import FIPS
from psychopy import event, visual, monitors
from psychopy.tools.monitorunittools import deg2pix, deg2cm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DEBUG = True
PR = False
DUMMY = True


class MeasureStimuli(FIPS, ABC):
    """
    Shows the stimuli on the screen to make measurements of luminance and size.

    Parameters
    ----------
    root_dir : str or Path
        Where the experiment is located
    """

    def __init__(self, root_dir):

        super().__init__(debug=DEBUG, root_dir=root_dir, practice=PR, dummy_tracker=DUMMY)

        self.set_init_positions(stim_side=1, offset=0)

    def go(self):
        """
        Shows a static screen with all the stimuli drawn. Waits for a quit command to exit the program.
        """
        # Write how long the measurements should be
        frame_outer = np.round(deg2cm(self.params["FRAME_SIZE"] * self.params["FRAME_FACTOR"], self.monitor), 2)
        frame_inner = np.round(deg2cm(self.params["FRAME_SIZE"], self.monitor), 2)
        probe = np.round(deg2cm(self.params["PROBE_SIZE"], self.monitor), 2)
        fix = np.round(deg2cm(self.params["FIX_SIZE"], self.monitor), 2)

        # Also make a one degree line as a sample and show how long it should be in cm on this monitor
        one_deg_cm = np.round(deg2cm(1, self.monitor), 2)
        one_deg_px = deg2pix(1, self.monitor)
        line = visual.Line(
            win=self.window,
            start=(-one_deg_px / 2, 0),
            end=(one_deg_px / 2, 0),
            lineWidth=5
        )
        line.pos = [300, self.window.size[1] / 4]

        # Write the message
        msg = visual.TextStim(
            win=self.window,
            pos=[0, self.window.size[1] / 4 + 20],
            text=f"1 deg = {one_deg_cm} cm\n" +
                 f"Fixation: {fix}\n" +
                 f"Probe: {probe}\n" +
                 f"Outer frame: {frame_outer}\n" +
                 f"Inner frame: {frame_inner}\n"
        )

        # Draw all the stimuli
        # self.window.color = 0
        self.window.flip()
        self.stimuli["fix"].draw()
        self.stimuli["frame_outer"].draw()
        self.stimuli["frame_inner"].draw()
        self.stimuli["probe_right"].draw()
        self.stimuli["probe_left"].fillColor = [-1, 1, -1]
        self.stimuli["probe_left"].draw()
        self.stimuli["match_outer"].draw()
        self.stimuli["match_inner"].draw()
        self.stimuli["match_right"].draw()
        line.draw()
        msg.draw()

        # Make the window fullscreen and show everything
        self.window.flip()

        # wait for the quit command
        event.waitKeys(keyList=['escape'])


class MeasureLuminance(BaseExperiment, ABC):
    """
    Shows color patches with different luminance to measure the monitor response

    Parameters
    ----------
    root_dir : str or Path
        Where the experiment is located
    """

    def __init__(self, root_dir):

        super().__init__(debug=DEBUG, root_dir=root_dir, practice=PR, dummy_tracker=DUMMY)

        self.lum_file = self.HOME / "data" / f"{self.params['MONITOR']}_luminance_values"

        # Set the channels that we want to calibrate
        self._guns = ['W', 'R', 'G', 'B']
        self.stimuli = self.setup_stimuli()

    def go(self):
        """
        Runs the procedure of showing the color patched and getting the responses
        """

        lum_msg.autoDraw = True
        resp.autoDraw = True
        button.autoDraw = True

        # Get the visual stimuli
        lum_stim = self.stimuli["lum"]
        resp_stim = self.stimuli["resp"]
        next_button = self.stimuli["button"]

        # Get the pre-defined measurement array
        values = self.setup_run()
        lum_stim.autoDraw = True

        for val in values:
            lum_stim.color = val[1:4]
            lum_stim.draw()
            self.window.flip()

            resp = True
            while resp:
                if next_button.isClicked:
                    lum_val = resp_stim.text
                    try:
                        val[4] = float(lum_val)
                        resp_stim.text = ""
                        resp_stim.borderColor = None
                        resp = False
                    except ValueError:
                        resp_stim.borderColor = [1, -1, -1]
                resp_stim.draw()
                lum_stim.draw()
                self.window.flip()

        # Save the measurements
        self.save_data(values)

    def michelson(self):
        """
        Shows two dark and bright patches to measure the michelson contrast
        """
        self.stimuli["dark-patch"].autoDraw = True
        self.stimuli["light-patch"].autoDraw = True

    def setup_stimuli(self):
        """
        Makes the color patch, a message, a response box to enter the luminance value in, and a button to advance to
        the next trial.

        Returns
        -------
        dict
        """

        # The patch that shows the luminance
        patch = visual.GratingStim(
            win=self.window,
            size=(1, 1),
            colorSpace='rgb255',
            units='norm',
            sf=0
        )

        lum_msg = visual.TextStim(
            win=self.window,
            text="Luminance (cd/m^2):",
            units='norm',
            pos=(-.4, -0.7),
            height=.05,
        )

        # Textbox to enter the luminance value
        resp = visual.TextBox2(
            win=self.window,
            units='norm',
            text="",
            font="Courier Prime",
            pos=(-0.1, -0.7),
            size=(0.3, 0.1),
            letterHeight=0.05,
            color=1,
            fillColor=-1,
            editable=True
        )

        # On-screen button to go to the next measurement
        button = visual.ButtonStim(
            win=self.window,
            text="Next",
            pos=(.6, -0.8),
            anchor='bottom-center',
            size=(0.1, 0.1),
            letterHeight=0.05,
            borderWidth=0.1,
            units='norm'
        )

        # Dark patch
        dark_patch = visual.Rect(
            win=self.window,
            pos=(5, 0),
            size=4,
            color=-1,
            units='deg',
            contrast=1
        )

        # Light patch
        light_patch = visual.Rect(
            win=self.window,
            pos=(-5, 0),
            size=4,
            color=1,
            units='deg',
            contrast=1
        )

        # save it to a dict
        stims = {
            "lum": patch,
            "resp": resp,
            "button": button,
            "dark-patch":dark_patch,
            "light-patch":light_patch
        }

        return stims

    def setup_run(self) -> np.ndarray:
        """
        Makes a 3D numpy array with pre-defined intensity values that has the shape
        gun x intensity values x luminance measurements

        Returns
        -------
        np.ndarray
        """
        # Testing 24 luminance values for each gun
        intensity_vals = np.arange(0, 255, 30)

        # Make an intensity level by RGB table
        w = np.repeat(intensity_vals, 3).reshape(len(intensity_vals), 3)
        # an identifier for this table (white = 0)
        cid = np.zeros((len(intensity_vals), 1))
        # a column with placeholders for later measurements
        measure_vals = np.zeros(len(intensity_vals)).reshape(-1, 1)
        # put them together
        w = np.hstack((cid, w, measure_vals, intensity_vals.reshape(-1, 1)))

        # Now the data has the shape of intensity values x 5. The first column is the id (W=0, R=1, G=2, B=3), the 2nd
        # to 4th columns are the actual RGB values, and the last column is the luminance measurement.

        # Do the same thing for other guns
        # Red = 1
        r = w.copy()
        # set G and B channels to zero
        r[:, 2:4] = 0
        # change the id to 1
        r[:, 0] = 1

        # Green = 2
        g = w.copy()
        g[:, [1, 3]] = 0
        g[:, 0] = 2

        # Blue = 3
        b = w.copy()
        b[:, 1:3] = 0
        b[:, 0] = 3

        # Now stack all of them on top of each other
        _arr = np.vstack((w, r, g, b))

        return _arr

    def plot_guns(self):
        """
        Plots the luminance measurements made for each gun
        """

        # Read the luminance measurements
        # (just let it quit if the file doesn't exist!)
        # data = np.load(str(self.lum_file) + '.npy')
        #
        # # Make a figure
        # fig = plt.figure(figsize=[12, 8])
        #
        # rgb_added = np.zeros((data[:, 0] == 0).flatnonzero()[-1])
        # for gun in range(4):
        #     idx = (data[:, 0] == gun).flatnonzero()
        #     rgb = data[np.where(data[:, 0] == gun)]
        #     if gun:
        #         rgb_added = rgb_added + data[np.where(data[:, 0] == gun)]
        #     lum = data[idx, 3]
        #     col = ['#DDDDDD', '#FF0000', '#00FF00', '#0000FF'][gun]
        #     plt.plot(rgb, lum, color=col)
        # plt.plot(rgb, rgb_added, ':', color='#000000')
        df = pd.read_csv(str(self.lum_file) + '.csv', sep=',')

        fig = plt.figure(figsize=(12, 8))
        p = sns.lineplot(
            x='INTENSITY',
            y='LUMINANCE',
            hue='CHANNEL_ID',
            palette=['#DDDDDD', '#FF0000', '#00FF00', '#0000FF'],
            data=df
        )
        p.legend_.remove()

        # Get the sum too
        sdf = df[~df["CHANNEL_ID"].isin([0])]
        ints = sdf["INTENSITY"].unique()
        vals = []
        for i in ints:
            v = 0
            for c in [1, 2, 3]:
                v += sdf.loc[(sdf["INTENSITY"] == i) & (sdf["CHANNEL_ID"] == c), "LUMINANCE"].to_numpy()[0]
            vals.append(v)

        # Plot it
        plt.plot(ints, vals, '--', color='#000000', label='Sum')
        plt.legend(["W", "R", "G", "B", "Sum"])

        self.window.close()
        plt.show()

    def save_data(self, data: np.ndarray):
        """
        Saves the gamma measurements made

        Parameters
        ----------
        data : np.ndarray
            The intensities and luminance measurements
        """

        # The dataframe will have 5 columns
        col_names = ["CHANNEL_ID", "R", "G", "B", "LUMINANCE", 'INTENSITY']

        # Make the dataframe
        df = pd.DataFrame(data, columns=col_names)

        # Save it
        df.to_csv(str(self.lum_file) + '.csv', sep=',', index=False)

        # Also save the numpy array
        np.save(str(self.lum_file) + '.npy', data)

    def get_grid_psychopy(self):
        """
        Runs gamma analysis on measurements to get the correction values.
        """
        data = np.load(str(self.lum_file) + '.npy')
        rgb = np.unique(data[:, 5])
        grid = np.zeros((4, 6))

        for gun in range(4):
            lums = data[np.where(data[:, 0] == gun)][:, 4]
            model = monitors.GammaCalculator(inputs=rgb, lums=lums, eq=2)

            # Put the values in the grid
            grid[gun, 0] = min(lums)  # minimum luminance
            grid[gun, 1] = max(lums)  # maximum luminance
            grid[gun, 2] = model.gamma  # Gamma parameter
            grid[gun, 3] = model.a  # a parameter
            grid[gun, 4] = model.b  # b parameter
            grid[gun, 5] = model.k  # k parameter

        return grid

    def save_grid(self, grid):

        grid_file = str(self.HOME / "config" / f"{self.params['MONITOR']}_gamma_grid.npy")
        np.save(grid_file, grid)


class PatrickGamma(FIPS, ABC):

    def __init__(self, root_dir):
        super().__init__(debug=DEBUG, root_dir=root_dir, practice=PR, dummy_tracker=DUMMY)

    def go(self, level, color):

        # Colors
        small_on_fill = self._set_fill(color, 1)  # get the fill color of small rectangles
        big_fill = self._set_fill(color, .5)

        # Dimensions
        n_small = 200  # both on and off
        small_height = 1  # pixels
        # width is 4 cycles/deg
        rect_width = deg2pix(1/8, self.window)

        # Stimuli
        ons = []
        offs = []
        for i in range(n_small//2):
            ons.append(
                visual.Rect(
                    win=self.window,
                    width=rect_width,
                    height=1,
                    lineWidth=0,
                    fillColor=small_on_fill,
                    autoLog=False
                )
            )
        for i in range(n_small//2):
            offs.append(
                visual.Rect(
                    win=self.window,
                    width=rect_width,
                    height=1,
                    lineWidth=0,
                    fillColor=-1,
                    autoLog=False
                )
            )

        big_rect = visual.Rect(
            win=self.window,
            width=rect_width,
            height=small_height * n_small,
            lineWidth=0,
            fillColor=big_fill,
            autoLog=False
        )
        bot_line = visual.Line(
            win=self.window,
            width=self.window.size[0] - 100,
            height=1,  # pixels
            lineWidth=0,
            fillColor=1,
            autoLog=False
        )
        top_line = visual.Line(
            win=self.window,
            width=self.window.size[0] - 100,
            height=1,  # pixels
            lineWidth=0,
            fillColor=1,
            autoLog=False
        )

        # Slider
        slider_nums = np.linspace(1, 4, 10)
        slider = visual.Slider(
            win=self.window,
            ticks=slider_nums,
            labels=slider_nums,
            units='norm',
            pos=-.8,
            size=[.7, .2],
            granularity=0,
            color=1,
            fillColor=-1,
            autoLog=False
        )

        #
        # if level == 50:

    @staticmethod
    def _set_fill(color, intensity):

        if color == 'r':
            small_fill = [intensity, 0, 0]
        elif color == 'g':
            small_fill = [0, intensity, 0]
        elif color == 'b':
            small_fill = [0, 0, intensity]
        else:
            small_fill = [intensity, intensity, intensity]

        return small_fill
