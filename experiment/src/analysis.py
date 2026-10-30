#!usr/bin/env python
"""
Created at 2/25/22
@author: Sharif Saleki

Provides the main class for analysis of different stages of the experiment.
"""
from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

from psychopy.data import FitWeibull, functionFromStaircase
from .utils import load_config, find_factors, within_subject_errorbar

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Plotting settings
custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False
}
sns.set_theme(
    font_scale=1.2,
    style='white',
    context='poster',
    font='Segoe UI',
    rc=custom_params
)


class TaskAnalyzer:

    def __init__(self, experiment, root_dir: Union[str, Path], **kwargs):

        # Set input parameters
        self.HOME = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.NAME = experiment
        self._task = None
        self._data = None
        self._mode = None
        self._fig_dir = self.HOME / 'figures'
        self._data_dir = self.HOME / 'data'

        # Get experiment conditions
        params_file = self.HOME / "config" / "parameters.json"
        settings_file = self.HOME / "config" / "settings.json"
        self.params = load_config(params_file)
        self.settings = load_config(settings_file)

        # Get the internal and external speeds
        self.ve = self.params["ENV_SPEED"]
        self.vi = self.params["TEX_SPEED"] * self.params["GABOR_SIZE"]  # convert from HZ to dva/s

        # Default angle and K values based on Heller et al. (2021)
        self.default_k = 0.74
        self.default_theta = np.rad2deg(np.arctan(self.default_k * self.vi / self.ve))

    @property
    def task(self):
        """ Getter for task attribute"""
        return self._task

    @task.setter
    def task(self, t):
        """ Setter for task attribute. Also loads the data associated with this task."""
        self._task = t
        self._data = self.load_task_data()

    @property
    def mode(self):
        """ Getter for experiment mode. Controls individual subject analysis vs. entire dataset analysis """
        return self._mode

    @mode.setter
    def mode(self, md):
        """ Setter to control what portion of the dataset gets plotted """
        assert md in ["each", "all"]
        self._mode = md

    @property
    def data(self):
        """ Getter for data in case we want to have a look"""
        return self._data

    def load_task_data(self, task=None):

        # paths and files
        # data_dir = self.HOME / 'data' / f'v{self.settings["version"]}'

        # get files from all subs
        if task is None:
            files = self._data_dir.glob(f'**/{self._task}/*.csv')
        else:
            files = self._data_dir.glob(f'**/{task}/*.csv')
        data_files = sorted(list(files))  # sort and list the files

        # load into one dataframe
        df = pd.concat(map(lambda f: pd.read_csv(f, header=0), data_files), ignore_index=True)

        return df

    def save_task_data(self):
        """
        Save the concatenated data to a csv file.

        Returns
        -------
        None
        """
        file_name = self.HOME / 'data' / f'{self.task}_task_concat.csv'
        self._data.to_csv(str(file_name))

    @staticmethod
    def _exp_func(x, b0, t, b1):
        return b0 + b1 * np.exp(t * x)

    @staticmethod
    def _linear_func(x, b0, b1):
        return b0 + b1 * x

    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    # --------------------------------------- ORIENTATION MATCHING TASK ----------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    def get_matching_stats(self, duration, distance, side, sub, kind):
        """
        Finds the mean values of reported angles in the orientation task to use in the
        length task for a subject.

        Parameters
        ----------
        duration : float

        distance : float

        side : int

        sub : int

        kind : str

        Returns
        -------
        stats : float
        """
        df = self._data.loc[
            (self._data["DURATION_MS"] == duration) & (self._data["DISTANCE_DVA"] == distance) &
            (self._data["STIM_SIDE"] == side) & (self._data["SUB_ID"] == sub)
            ]
        df = df[(np.abs(stats.zscore(df[["ANGLE"]])) < 3)]

        if kind == 'mean':
            stat = df["ANGLE"].mean()
        elif kind == 'mode':
            stat = stats.mode(df["ANGLE"])[0][0]
        elif kind == 'median':
            stat = df["ANGLE"].median()
        else:
            raise ValueError("Invalid kind of statistic.")

        return stat

    @staticmethod
    def angle_vector_combo():
        """
        Fits a logarithmic function to some pre-defined internal

        Parameters
        ----------
        vi : float
            Internal speed of the Gabor patch

        ve : float
            External speed of the Gabor patch

        Returns
        -------
        None
        """
        ks = np.array([.69, .89, .94, .93, 1.16, 1.11])
        speeds = np.array([1.2, 2.4, 3.6, 4.8, 6, 7.2])
        log_speeds = np.log(speeds)
        b, a = np.polyfit(log_speeds, ks, 1)
        vi = 9
        ve = 2.5
        k_pred = np.log(vi) * a + b
        angle_pred = np.rad2deg(np.arctan(k_pred * vi / ve))

        return angle_pred

    def angle_from_heller(self):
        """
        Calculate K value from the parameters found in Heller et al. (2021)

        Parameters
        ----------
        vi : float
            Internal speed of the Gabor

        ve : float
            External speed of the Gabor

        Returns
        -------
        theta : float
            Predicted illusory angle
        """
        # Calculate K for this external speed
        s = -0.066
        k = 1 + s * self.ve

        # Calculate angle for this K and speeds
        theta = self.calc_angle(k, self.vi, self.ve)

        return k, theta

    @staticmethod
    def calc_angle(k, vi, ve):
        """
        Calculates an angle
        Args:
            k:
            vi:
            ve:

        Returns:

        """
        return np.rad2deg(np.arctan(k * vi / ve))

    def plot_ori_report(self, save=False):
        """
        Plots the results of the matching task (adjusted orientations). The plot is only for the double-drift stimulus.
        It is deviation from vertical in degrees plotted as a function of the path duration in ms.

        There should also be a horizontal line that corresponds to the K value predicted from the vector combination
        model based on  Heller et al. (2021) for this combination of parameters of double-drift.

        Parameters
        ----------
        save : bool
        """
        # Initiate figure
        # The y-axis start from 60 or something, so to put a break in that axis I make a
        # gridspec with two top and bottom grids.
        fig = plt.figure(figsize=(16, 9))  # figure
        n = 10  # width ratio of grid
        m = 1  # height ratio of grid
        top_pos, bot_pos = gridspec.GridSpec(2, 1, height_ratios=[n, m])  # 2 by 1 grid
        ax_top = fig.add_subplot(top_pos)  # make the top figure
        ax_bottom = fig.add_subplot(bot_pos, sharex=ax_top)  # make the bottom figure
        # fig.suptitle(f"Orientation Reports (Double-Drift)", fontsize='large')  # title
        pal = 'Spectral'  # palette

        # The trick for generating within subject error bars
        self._data["ANGLE_FROM_VERTICAL"] = 90 - self._data["ANGLE"]
        new_col_name = "ANGLE_NORM"
        errorbar_df = within_subject_errorbar(
            df=self._data,
            score_col="ANGLE_FROM_VERTICAL",
            new_col=new_col_name,
            cond_cols=1,
            ci=True  # %95 by default
        )

        # Generate the final dataframe with only duration and deviation angles for plotting
        _df = errorbar_df.groupby(["SUB_ID", "DURATION_MS"], as_index=False)["ANGLE_NORM"].mean()

        # Get the prediction from Heller et al. (2021) for the stimulus parameters
        k, angle_pred = self.angle_from_heller()

        # Generate plots
        # There is a break in the middle of the axes so I need to generate two plots for top and bottom
        plot_top = sns.pointplot(
            x='DURATION_MS',
            y='ANGLE_NORM',
            # palette=pal,
            plot_kws=dict(s=30),
            # color='dimgray',
            errorbar='se',
            data=_df,
            ax=ax_top
        )
        plot_bot = sns.pointplot(
            x='DURATION_MS',
            y='ANGLE_NORM',
            # palette=pal,
            # color='dimgray',
            plot_kws=dict(s=30),
            errorbar='se',
            data=_df,
            ax=ax_bottom
        )

        # Make the K plot
        kdf = self._data.groupby(["SUB_ID", "DURATION_MS"], as_index=False)["ANGLE_FROM_VERTICAL"].mean()
        kdf["K"] = np.tan(np.deg2rad(kdf["ANGLE_FROM_VERTICAL"])) * self.ve / self.vi

        # k_ax_top = ax_top.twinx()
        # k_plot_top = sns.pointplot(
        #     x="DURATION_MS",
        #     y="K",
        #     ci=20,
        #     color='orange',
        #     data=kdf,
        #     ax=k_ax_top
        # )

        # k_ax_bot = ax_bottom.twinx()
        # k_plot_bot = sns.pointplot(
        #     x="DURATION_MS",
        #     y="K",
        #     ci=20,
        #     color='orange',
        #     data=kdf,
        #     ax=k_ax_top
        # )

        # Write k value on the plot
        # ax_top.text(2, 69, f'$K = {k}$')

        # draw the horizontal line for the calculated K
        ax_top.axhline(angle_pred, color='dimgrey', linestyle='--')

        # Change aesthetics
        ax_top.set_ylim(bottom=65)  # change y range of the top axis
        ax_bottom.set_ylim(0, 1)  # short y range for the bottom axis
        # k_ax_bot.set_ylim(0, 0.7)  # short y range for the bottom axis
        plt.setp(ax_top.get_xticklabels(), visible=False)  # no labels for the top axis
        sns.despine(ax=ax_top, bottom=True)  # remove top axis' bottom spine
        # sns.despine(ax=k_ax_top, bottom=True, right=False)  # remove top axis' bottom spine
        sns.despine(ax=ax_bottom, top=True)  # remove bottom axis' top spine

        # Change labels
        plot_top.set(
            xlabel=None,
            ylabel='Deviation from Vertical (°)',
        )
        plot_bot.set(
            xlabel='Path duration (ms)',
            ylabel=None,
            yticks=[0],
        )

        # k_plot_top.set(
        #     ylabel='K (a.u.)'
        # )
        # k_plot_bot.set(
        #     ylabel=None,
        #     yticks=[0]
        # )

        # Draw
        plt.show()

        # For saving directly
        if save:
            fig.savefig(str(self._fig_dir / "orientation_plot.png"))

    def plot_k_values(self, save=False):
        """
        Plots mean K value for each duration.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(16, 9))  # figure
        # fig.suptitle(r"$k$ values", size='large')  # title

        # Make the K plot
        kdf = self._data.groupby(["SUB_ID", "DURATION_MS"], as_index=False)["ANGLE_FROM_VERTICAL"].mean()
        kdf["K"] = np.tan(np.deg2rad(kdf["ANGLE_FROM_VERTICAL"])) * self.ve / self.vi

        plot = sns.pointplot(
            x="DURATION_MS",
            y="K",
            errorbar='se',
            color='purple',
            data=kdf,
            ax=ax
        )

        plot.set(
            xlabel='Path duration (ms)',
            ylabel="K"
        )
        plot.yaxis.label.set_rotation(0)
        plot.yaxis.set_label_coords(-0.1, 0.5)

        # For saving directly
        if save:
            fig.savefig(str(self._fig_dir / "k_plot.png"))

    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- LENGTH ADJUSTMENT TASK ------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    def plot_len_reports(self, save=False):
        """
        Plots reported path length in dva against path duration (ms), and uses color to differentiate
        single- from double-drift stimulus

        Parameters
        ----------
        save : bool

        Returns
        -------
        None
        """
        # Figure and titles
        fig, ax = plt.subplots(figsize=(16, 9))
        # fig.suptitle("Length Reports", fontsize='large')
        pal = 'Spectral'

        # Generate within-subject error bars
        new_col_name = "LENGTH_NORM"
        errorbar_df = within_subject_errorbar(
            df=self._data,
            score_col="REPORT_LENGTH_DVA",
            new_col=new_col_name,
            cond_cols=2,
            ci=True  # %95 by default
        )

        # Generate the final dataframe with only duration and deviation angles for plotting
        _df = errorbar_df.groupby(["SUB_ID", "STIM_TYPE", "DURATION_MS"], as_index=False)["LENGTH_NORM"].mean()

        # Generate plots
        length_plot = sns.pointplot(
            x='DURATION_MS',
            y='LENGTH_NORM',
            hue='STIM_TYPE',
            errorbar='se',
            # palette=pal,
            data=_df,
            ax=ax
        )

        length_plot.set(
           xlabel='Path duration (ms)',
            ylabel='Path Length (dva)',
        )

        # Change legend
        legend_title = None
        length_plot.legend_.set_title(legend_title)
        # replace labels
        new_labels = ['Double-drift', 'Single-drift']
        for t, l in zip(length_plot.legend_.texts, new_labels):
            t.set_text(l)

        if save:
            fig.savefig(str(self._fig_dir / "length_plot.png"))

    def get_ratios(self):
        """
        Calculates the ratios of DD and SD.

        Returns:
            pandas.Dataframe

        """
        # Get the vector combo weights
        df = self.df_with_weights()

        # DD ratio is we * ve * T * cos(theta) / reported length
        dd_df = df[df.STIM_TYPE == "dd"]
        dd_df["RATIO"] = (dd_df["DURATION_MS"] / 1000) * self.ve * np.cos(np.deg2rad(dd_df["ANGLE_FROM_VERTICAL"])) \
                         / dd_df["REPORT_LENGTH_DVA"]
        dd_df["RATIO_RAW"] = (dd_df["DURATION_MS"] / 1000) / dd_df["REPORT_LENGTH_DVA"]

        # # DeValois ratio is wi * vi * T / reported length
        # deva_df = df[df.STIM_TYPE == 'deva']
        # deva_df["RATIO"] = deva_df["INTERNAL_WEIGHT"] * (deva_df["DURATION_MS"] / 1000) * self.ve \
        #                    / deva_df["REPORT_LENGTH_DVA"]

        # DeValois ratio is just w_sd / w_i
        deva_df = df[df.STIM_TYPE == 'deva']
        deva_df["RATIO"] = deva_df.INTERNAL_WEIGHT / dd_df.INTERNAL_WEIGHT

        df = pd.concat([dd_df, deva_df])

        return df

    def plot_dd_ratio(self, save=False, inverse=False):
        """

        Args:
            save:

        Returns:

        """
        # Get the dataframe with ratios
        df = self.get_ratios()
        df["INV_RATIO"] = 1 / df["RATIO"]

        # Make the figure
        fig, ax = plt.subplots(figsize=(16, 9))

        # Make the plot
        plot = sns.pointplot(
            x="DURATION_MS",
            y="INV_RATIO" if inverse else "RATIO",
            data=df[df.STIM_TYPE == "dd"]
        )

        # Change aesthetics
        plot.set(
            xlabel="Path Duration (ms)",
            ylabel="Perceived to physical ratio" if inverse else "Physical to perceived ratio",
            ylim=[0, 12] if inverse else None
        )

        if save:
            fig.savefig(str(self._fig_dir / "dd_ratio_plot.png"))

        return df

    def plot_dd_raw_ratio(self, save=False, inverse=False):
        """

        Args:
            save:

        Returns:

        """
        # Get the dataframe with ratios
        df = self.get_ratios()
        df["INV_RATIO_RAW"] = 1 / df["RATIO_RAW"]

        # Make the figure
        fig, ax = plt.subplots(figsize=(16, 9))

        plot = sns.pointplot(
            x="DURATION_MS",
            y="INV_RATIO_RAW" if inverse else "RATIO_RAW",
            color='red',
            data=df[df.STIM_TYPE == "dd"]
        )

        # Change aesthetics
        plot.set(
            xlabel="Path Duration (ms)",
            ylabel="Perceived to physical ratio" if inverse else "Physical to perceived ratio",
            ylim=[0, 7] if inverse else None
        )

        if save:
            fig.savefig(str(self._fig_dir / "dd_ratio_raw_plot.png"))

        return df

    def plot_deva_ratio(self, save=False):
        """

        Args:
            save:

        Returns:

        """
        # Get the dataframe with ratios
        df = self.get_ratios()

        # Make the figure
        fig, ax = plt.subplots(figsize=(16, 9))

        # Make the plot
        plot = sns.pointplot(
            x="DURATION_MS",
            y="RATIO",
            color='#F97306',
            data=df[df.STIM_TYPE == "deva"]
        )

        # Change aesthetics
        plot.set(
            xlabel="Path Duration (ms)",
            # ylabel=r"$\frac{W_{sd}}{W_i}$"
        )
        plot.set_ylabel(ylabel=r"$\frac{W_{sd}}{W_i}$", fontsize=36)
        plot.yaxis.label.set_rotation(0)
        plot.yaxis.set_label_coords(-0.12, 0.5)

        if save:
            fig.savefig(str(self._fig_dir / "deva_ratio_plot.png"))

    def plot_ratio(self, save=False):
        """
        Plots reported path length in dva against path duration (ms), and uses color to differentiate
        single- from double-drift stimulus

        Args:
            save: bool

        Returns:
            None
        """
        # Get the dataframe with ratios
        df = self.get_ratios()

        # Make the figure
        fig, ax = plt.subplots(figsize=(16, 9))
        # fig.suptitle("Ratio", fontsize=30)

        # Make the plot
        plot = sns.pointplot(
            x="DURATION_MS",
            y="RATIO",
            hue="STIM_TYPE",
            data=df
        )

        # Change aesthetics
        plot.set(
            xlabel="Path Duration (ms)",
            ylabel="Weighted path / Reported path"
        )
        # Change legend
        legend_title = None
        plot.legend_.set_title(legend_title)
        # replace labels
        new_labels = ['Double-drift', 'Single-drift']
        for t, l in zip(plot.legend_.texts, new_labels):
            t.set_text(l)

        if save:
            fig.savefig(str(self._fig_dir / "ratio_plot.png"))

    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- VECTOR COMBINATION ----------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    def plot_weights(self, save=False):
        """

        Parameters
        ----------
        save

        Returns
        -------

        """
        # Make figure
        fig, ax = plt.subplots(figsize=(16, 9))
        # fig.suptitle("Vector combination weights", fontsize=30)

        # Find weights
        df = self.df_with_weights()

        # Compute Ks
        dd_df = df[df.STIM_TYPE == 'dd']
        mean_df = dd_df.groupby(["DURATION_MS"], as_index=False)["INTERNAL_WEIGHT", "EXTERNAL_WEIGHT"].mean()

        # Plot all the weights
        weights_plot = sns.scatterplot(
            x="EXTERNAL_WEIGHT",
            y="INTERNAL_WEIGHT",
            hue="DURATION_MS",
            palette='Spectral',
            s=200,
            alpha=0.5,
            ax=ax,
            legend=False,
            # data=dd_df.loc[dd_df["EXTERNAL_WEIGHT"] < 0.2]
            data=dd_df
        )

        # Plot the means
        mean_plot = sns.scatterplot(
            x="EXTERNAL_WEIGHT",
            y="INTERNAL_WEIGHT",
            hue="DURATION_MS",
            palette='Spectral',
            s=1000,
            ax=ax,
            data=mean_df
        )

        # Plot the regression line
        regression_plot = sns.regplot(
            x="EXTERNAL_WEIGHT",
            y="INTERNAL_WEIGHT",
            scatter=False,
            # scatter_kws=dict(cmap='Spectral'),
            # palette='Spectral',
            color="black",
            data=mean_df
        )

        # Change aesthetics
        weights_plot.set(
            xlabel=r"External weight ($w_e$)",
            ylabel=r"Internal weight ($w_i$)",
        )
        # ax.set_xlim(right=0.20)

        legend_title = "Duration (ms)"
        mean_plot.legend(fontsize=18)
        mean_plot.legend_.set_title(legend_title)
        # # replace labels
        # new_labels = [
        #     rf"{dur}, $\bark$={round(kdf.loc[kdf.DURATION_MS == dur, 'K'].values[0], 2)}" for dur in df.DURATION_MS.unique()
        # ]
        # new_labels = [f"{t}ms" for t in df.DURATION_MS.unique()]
        new_labels = list(df.DURATION_MS.unique())
        for t, l in zip(mean_plot.legend_.texts, new_labels):
            t.set_text(l)
        # ax.setp(weights_plot.get_legend().get_texts(), fontsize='12')
        # weights_plot.legend_.remove()

        # save
        if save:
            fig.savefig(str(self._fig_dir / "weights_plot.png"))

        return df

    def plot_weights_all(self, save=False):
        """

        Parameters
        ----------
        save

        Returns
        -------

        """
        # Make figure
        fig, ax = plt.subplots(figsize=(16, 9))
        # fig.suptitle("Vector combination weights", fontsize='large')

        # Find weights
        df = self.df_with_weights()

        # Compute Ks
        dd_df = df[df.STIM_TYPE == 'dd']
        mean_df = dd_df.groupby(["DURATION_MS"], as_index=False)["INTERNAL_WEIGHT", "EXTERNAL_WEIGHT"].mean()

        # Plot the means
        mean_plot = sns.scatterplot(
            x="EXTERNAL_WEIGHT",
            y="INTERNAL_WEIGHT",
            hue="DURATION_MS",
            palette='Spectral',
            s=2000,
            alpha=1,
            ax=ax,
            data=mean_df
        )

        # Plot all the weights
        weights_plot = sns.scatterplot(
            x="EXTERNAL_WEIGHT",
            y="INTERNAL_WEIGHT",
            hue="DURATION_MS",
            palette='Spectral',
            s=400,
            alpha=0.5,
            ax=ax,
            legend=False,
            data=dd_df
        )

        # Plot the regression line
        regression_plot = sns.regplot(
            x="EXTERNAL_WEIGHT",
            y="INTERNAL_WEIGHT",
            scatter=False,
            color="black",
            data=dd_df
        )

        # Change aesthetics
        weights_plot.set(
            xlabel=r"External weight ($w_e$)",
            ylabel=r"Internal weight ($w_i$)",
        )
        # ax.set_xlim(right=0.20)

        legend_title = "Duration (ms)"
        weights_plot.legend(fontsize=18)
        weights_plot.legend_.set_title(legend_title)

        # replace labels
        new_labels = list(df.DURATION_MS.unique())
        for t, l in zip(weights_plot.legend_.texts, new_labels):
            t.set_text(l)

        # save
        if save:
            fig.savefig(str(self._fig_dir / "weights_plot_all.png"))

        return df

    @staticmethod
    def vector_combo_weights(vi: float, ve: float, length: float, theta: float, tau: float):
        """

        Parameters
        ----------
        vi : float
            Internal motion velocity

        ve : float
            External motion velocity

        length : float
            Perceived length

        theta : float
            Illusory angle in degrees

        tau : float
            Duration of motion

        Returns
        -------
        tuple (we, wi)
            External and internal weights
        """
        wi = length * np.sin(np.deg2rad(theta)) / (vi * tau / 1000)
        we = length * np.cos(np.deg2rad(theta)) / (ve * tau / 1000)

        return wi, we

    def df_with_weights(self):
        """

        Returns:
            pandas.Dataframe that includes internal and external weights for the double drift
        """

        # Find weights
        self.task = "adjustment"
        self.mode = "all"
        df = self._data.copy()

        dd_df = df[df.STIM_TYPE == 'dd'].groupby(
            ["SUB_ID", "DURATION_MS"], as_index=False
        )["REPORT_LENGTH_DVA", "ENVELOPE_ANGLE"].mean()
        dd_df["ANGLE_FROM_VERTICAL"] = 90 - dd_df["ENVELOPE_ANGLE"]

        w_i, w_e = self.vector_combo_weights(
            self.vi, self.ve, dd_df["REPORT_LENGTH_DVA"], dd_df["ANGLE_FROM_VERTICAL"], dd_df["DURATION_MS"]
        )

        dd_df["EXTERNAL_WEIGHT"] = w_e
        dd_df["INTERNAL_WEIGHT"] = w_i
        dd_df["STIM_TYPE"] = 'dd'

        # Also add the internal weights for the single drift
        deva_df = df[df.STIM_TYPE == 'deva'].groupby(
            ["SUB_ID", "DURATION_MS"], as_index=False
        )["REPORT_LENGTH_DVA"].mean()
        deva_df["STIM_TYPE"] = 'deva'

        deva_df["INTERNAL_WEIGHT"] = deva_df.REPORT_LENGTH_DVA / (self.vi * deva_df.DURATION_MS / 1000)
        #
        # for sub in df.SUB_ID.unique():
        #     for dur in df.DURATION_MS.unique():
        #         deva_df.loc[
        #             (deva_df.SUB_ID == sub) & (deva_df.DURATION_MS == dur), "INTERNAL_WEIGHT"
        #         ] = dd_df.loc[(dd_df.SUB_ID == sub) & (dd_df.DURATION_MS == dur), "INTERNAL_WEIGHT"]

        df = pd.concat([deva_df, dd_df])

        return df
    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    # -------------------------------------------- OTHER METHODS ------------------------------------------ #
    # ----------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    def _round_dur(self):
        """ Rounds the duration of the probe because sometimes they are floats with many digits"""
        self._data["ROUND_DUR"] = np.around(self._data["PROBE_DUR"])

    def _prepare_plotting(self):
        """
        Sets up plotting based on the mode of the analysis

        Returns
        -------
        tuple : figure and axes of matplotlib.pyplot
        """
        # Parameters
        if self._mode == 'all':
            subs = [None]
            n_rows = 1
            n_cols = 1
        else:
            subs = self._data["SUB_ID"].unique()
            n = len(subs)

            # find a good set of factors
            factors = find_factors(n)
            if len(factors) > 2:
                n_cols = factors[1]  # set the smallest factor as the number of columns
                n_rows = n // n_cols
            else:
                # For prime numbers
                n_cols = 1
                n_rows = n

        fig_size = (16, 9 * len(subs))

        # Set up plotting
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

        return fig, axes


class MeasurementAnalyzer:

    def __init__(self, experiment, root_dir: Union[str, Path], **kwargs):

        # Set input parameters
        self.HOME = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.NAME = experiment

        # Get experiment conditions
        params_file = self.HOME / "config" / "parameters.json"
        settings_file = self.HOME / "config" / "settings.json"
        self.params = load_config(params_file)
        self.settings = load_config(settings_file)

    def get_gamma(self):
        pass
