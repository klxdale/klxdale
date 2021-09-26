import numpy as np

# This function takes a matrix of size m x n and normalises every column (of size m x 1) to values between 0 and 1
def normalise_0_1(input_matrix):

    # Get shape of matrix
    dimensions_matrix = input_matrix.shape

    rows = dimensions_matrix[0]
    cols = dimensions_matrix[1]

    for i in range(cols):
        feature = input_matrix.iloc[:,i]
        normalised = (feature-min(feature))/(max(feature)-min(feature))
        input_matrix.iloc[:,i] = normalised

    return input_matrix



def categorise_margin(Y):

    # Column Name
    col = 'Margin'

    # Conditions
    conditions = [Y[col] < -60,
                  (Y[col] >= -60) & (Y[col] < -30),
                  (Y[col] >= -30) & (Y[col] < -12),
                  (Y[col] >= -12) & (Y[col] < -0),
                  (Y[col] >= 0) & (Y[col] < 12),
                  (Y[col] >= 12) & (Y[col] < 30),
                  (Y[col] >= 30) & (Y[col] < 60),
                  Y[col] >= 60]

    choices = ['Very heavy home loss',
               'Heavy home loss',
               'Moderate home loss',
               'Close home loss',
               'Close home win',
               'Moderate home win',
               'Heavy home win',
               'Very heavy home win']

    Y['Margin'] = np.select(conditions, choices, default=np.nan)

    return Y


def category_to_win_loss(Y):

    col = 'Margin'

    conditions = [Y[col] == "Very heavy home loss",
                  Y[col] == "Heavy home loss",
                  Y[col] == "Moderate home loss",
                  Y[col] == "Close home loss",
                  Y[col] == "Close home win",
                  Y[col] == "Moderate home win",
                  Y[col] == "Heavy home win",
                  Y[col] == "Very heavy home win"]

    choices = [0,
               0,
               0,
               0,
               1,
               1,
               1,
               1]

    Y['Margin'] = np.select(conditions, choices, default=np.nan)

    return Y


def category_to_blowout_winloss(Y):

    col = 'Margin'

    conditions = [Y[col] == "Very heavy home loss",
                  Y[col] == "Heavy home loss",
                  Y[col] == "Moderate home loss",
                  Y[col] == "Close home loss",
                  Y[col] == "Close home win",
                  Y[col] == "Moderate home win",
                  Y[col] == "Heavy home win",
                  Y[col] == "Very heavy home win"]

    choices = [1,
               0,
               0,
               0,
               0,
               0,
               0,
               1]

    Y['Margin'] = np.select(conditions, choices, default=np.nan)

    return Y



def category_to_win_loss_TF(Y):

    col = 'Margin'

    conditions = [Y[col] == "Very heavy home loss",
                  Y[col] == "Heavy home loss",
                  Y[col] == "Moderate home loss",
                  Y[col] == "Close home loss",
                  Y[col] == "Close home win",
                  Y[col] == "Moderate home win",
                  Y[col] == "Heavy home win",
                  Y[col] == "Very heavy home win"]

    choices = [False,
               False,
               False,
               False,
               True,
               True,
               True,
               True]

    Y['Margin'] = np.select(conditions, choices, default=np.nan)

    return Y