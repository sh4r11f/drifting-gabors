#!usr/bin/env python
"""
Created at 3/18/22
@author: devxl

Utility functions that help with running neuroscience experiments.
"""
# System imports
import json
from itertools import product

# Scientific imports
import pandas as pd
import numpy as np


def load_config(file_path):
    """
    Reads and saves parameters specified in a json file under the config directory, e.g. config/params.json

    Parameters
    ----------
    file_path : str or pathlib.Path
        The file to be loaded.

    Returns
    -------
    dict
        The .json file is returned as a python dictionary.
    """
    # Check its status and read it
    if file_path.is_file():
        try:
            with open(str(file_path)) as pf:
                params = json.load(pf)
        # for corrupted files or other issues
        except IOError as e:
            raise f"Unable to open {file_path}: {e}"
    else:
        raise FileNotFoundError(f"{str(file_path)} does not exist")

    return params


def find_factors(num):
    """
    Finds all the factors of a natural number

    Parameters
    ----------
    num : int

    Returns
    -------
    list : factors of the input number
    """
    # Make sure we're dealing with a positive integer
    if not (isinstance(num, int) and num > 0):
        raise ValueError("Input must be a natural number.")

    # Find the factors
    factors = []
    for i in range(1, num + 1):
        if not num % i:
            factors.append(i)

    return factors


def within_subject_errorbar(df: pd.DataFrame, score_col: str, new_col: str, cond_cols: int, ci: bool):
    """
    Changes the values of a dependent variable such that when plotting, errorbars would
    reflect within-subject standard error (or %95 CI, if desired). Based on Cousineau & O'Brien (2012).

    For each condition, it takes each subject's average and replaces it with the grand average (by subtracting
    subject average from each datapoint and adding the grand average). This allows the variation
    in the data to reflect within-subject deviation from the mean.

    Then, it applies a correction factor (sqrt(J/J-1) for each condition, where J is the number of conditions,
    to change the spread and finishes by undoing the centering. This allows the mean of the standardized scores to be
    the same as the original data, but the spread is modified.

    For CIs that reflect the significance of a test (e.g., means are not different at a decision threshold of .05
    if the CI in one condition contains the mean of another condition), there is an additional correction factor
    of 1/sqrt(n) that can be applied.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to use for calculating averages

    score_col : str
        Name of the column of the dependent variable

    new_col : str
        Name of the new column that represents the adjusted values

    cond_cols : int
        Number of condition columns to use

    ci : bool
        Whether to apply the additional correction factor for intrepretable CIs.

    Returns
    -------
    pd.DataFrame
    """
    # Don't modify the original data
    new_df = df.copy()
    new_df["Y_score"] = np.zeros(len(df))
    new_df[new_col] = np.zeros(len(df))

    # Get the conditions and how many there are
    if cond_cols == 1:
        durations = df.DURATION_MS.unique()
        n_conditions = len(durations)

    elif cond_cols == 2:
        stim_types = df.STIM_TYPE.unique()
        durations = df.DURATION_MS.unique()
        n_conditions = len(list(product(stim_types, durations)))

    else:
        raise ValueError

    # Get a list of subjects, assuming subject column name is SUB_ID
    subs = new_df["SUB_ID"].unique()

    # Level 1 normalization: Y_{sj} = X_{sj} - \bar{X}_{s.} + \bar{X}_{..}
    # Take the average of all subjects across all conditions i.e. \bar{X}_{..}
    grand_avg = new_df[score_col].mean()

    # Calculate the individual average in each condition
    for sub in subs:
        # select subject's data
        sub_df = new_df[new_df["SUB_ID"] == sub]

        # take the mean of subject across all conditions i.e. \bar{X}_{s.}
        sub_avg = sub_df[score_col].mean()

        # replace subject average with grand average for each datapoint
        new_df.loc[(new_df["SUB_ID"] == sub), "Y_score"] = sub_df[score_col] - sub_avg + grand_avg

    # Level 2 normalization: Z_{sj} = sqrt(J/J-1) * (Y_{sj} - \bar{Y}_{.j}) + \bar{Y}_{.j}
    # Calculate the correction factor. If %95 CI is desired, there is an additional 1/sqrt(2) factor that needs to be
    # multiplied. That means that for 2 factors, this step does not change the data at all.
    corr_factor = np.sqrt(n_conditions / (2 * (n_conditions - 1))) if ci else np.sqrt(n_conditions / (n_conditions - 1))

    # This time go through conditions first
    for duration in durations:

        if cond_cols == 1:

            # select condition data
            cond_df = new_df[(new_df.DURATION_MS == duration)]

            # get the average in this condition i.e. \bar{Y}_{.j}
            # these are Y scores, so use the temp column that was made in the 1st step
            cond_avg = cond_df["Y_score"].mean()

            # go through subject
            for sub in subs:
                # select the subject data
                sub_df = cond_df[cond_df["SUB_ID"] == sub]

                # get the mean of this subject i.e. Y_{sj}
                sub_avg = sub_df["Y_score"].mean()

                # apply correction
                new_df.loc[(new_df["SUB_ID"] == sub) & (new_df.DURATION_MS == duration), new_col] = \
                    corr_factor * (sub_avg - cond_avg) + cond_avg

        elif cond_cols == 2:

            for stim_type in stim_types:

                # select condition data
                cond_df = new_df[(new_df.STIM_TYPE == stim_type) & (new_df.DURATION_MS == duration)]

                # get the average in this condition i.e. \bar{Y}_{.j}
                # these are Y scores, so use the temp column that was made in the 1st step
                cond_avg = cond_df["Y_score"].mean()

                # go through subject
                for sub in subs:

                    # select the subject data
                    sub_df = cond_df[cond_df["SUB_ID"] == sub]

                    # get the mean of this subject i.e. Y_{sj}
                    sub_avg = sub_df["Y_score"].mean()

                    # apply correction
                    new_df.loc[(new_df["SUB_ID"] == sub) & (new_df.STIM_TYPE == stim_type) &
                               (new_df.DURATION_MS == duration), new_col] = \
                        corr_factor * (sub_avg - cond_avg) + cond_avg

    return new_df
