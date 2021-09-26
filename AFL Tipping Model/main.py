### This file makes a retrospective AFL tipping model for the 2019 season ###
# Author: Kieran Dale

# It takes a range of inputs based on a team's form, rolling averages for disposals, tackles, marks and other recorded statistics.
# Different machine learning algorithms are used to train models which predict the likelihood of the home team winning.
# This exercise is based on the fascinating and well-explained articles detailing AFL statistical models here: https://www.aflgains.com/
# NOTE: A key variation on the above model is the effect of each team's form. This will be considered over the last M games (3, 5, 7 in this script, to be implemented).

### Script Layout ###
# 0 Import packages and functions
# 1 Webscrape, clean and prepare data as inputs
# 2 Test/Tune several models on 2018 data
# 3 Validate machine learning algorithms using 2019 data
# 4 Apply model to betting


#######################################
### 0 Import packages and functions ###
#######################################

import numpy as np
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Lasso
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

os.chdir('C:\\Users\\Kieran Dale\\Documents\\Python Projects\\AFL Tipping')

execfile('ladder.py')
execfile('get_afl_data.py')
execfile('normalise.py')
execfile('model performance functions.py')

#####################################################
### 1 Webscrape, clean and prepare data as inputs ###
#####################################################

# Rolling Window
r = 5

# Get raw scores
scores_2018 = get_afl_scores("https://fixturedownload.com/download/afl-2018-AUSEasternStandardTime.csv")
scores_2019 = get_afl_scores("https://fixturedownload.com/download/afl-2019-AUSEasternStandardTime.csv")

# Get raw data
data_2018 = get_afl_stats("https://afltables.com/afl/stats/2018t.html")
data_2019 = get_afl_stats("https://afltables.com/afl/stats/2019t.html")

# Turn all that data into differentials
diff_2018 = data_2018[:]
diff_2019 = data_2019[:]

for index, column in enumerate(
        diff_2018.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], 0):
    for row in range(396):
        diff_2018[column][row] = eval(diff_2018[column][row].partition('-')[0])

for index, column in enumerate(
        diff_2019.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
        0):
    for row in range(396):
        diff_2019[column][row] = eval(diff_2019[column][row].partition('-')[0])

roll_2018 = diff_2018[:]
roll_2019 = diff_2019[:]

# Now turn this into rolling averages
# But first, shift the statistics so we only see the weeks leading up to the game
for index, column in enumerate(
        roll_2018.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], 0):
    roll_2018[column] = roll_2018.groupby('Team', as_index=False)[column].transform(lambda x: x.shift(1))
    roll_2018[column] = roll_2018.groupby('Team', as_index=False)[column].rolling(r, min_periods=1).mean().reset_index(
        0, drop=True).iloc[:, 1]

for index, column in enumerate(
        roll_2018.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], 0):
    roll_2019[column] = roll_2019.groupby('Team', as_index=False)[column].transform(lambda x: x.shift(1))
    roll_2019[column] = roll_2019.groupby('Team', as_index=False)[column].rolling(r, min_periods=1).mean().reset_index(
        0, drop=True).iloc[:, 1]

# Merge scores and stats and from now we will only consider individual games with reference to the home team
home_data_2018 = pd.merge(scores_2018, roll_2018, how='left', left_on=['Round', 'Home Team'],
                          right_on=['Round', 'Team'])
away_data_2018 = pd.merge(scores_2018, roll_2018, how='left', left_on=['Round', 'Away Team'],
                          right_on=['Round', 'Team'])
merged_data_2018 = home_data_2018[:]
for index, column in enumerate(
        roll_2018.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], 0):
    merged_data_2018[column] = home_data_2018[column] - away_data_2018[column]

home_data_2019 = pd.merge(scores_2019, roll_2019, how='left', left_on=['Round', 'Home Team'],
                          right_on=['Round', 'Team'])
away_data_2019 = pd.merge(scores_2019, roll_2019, how='left', left_on=['Round', 'Away Team'],
                          right_on=['Round', 'Team'])
merged_data_2019 = home_data_2019[:]
for index, column in enumerate(
        roll_2019.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], 0):
    merged_data_2019[column] = home_data_2019[column] - away_data_2019[column]

# We need to add the following columns for wins, form and ladder position
merged_data_2018['Home Win'] = np.where(merged_data_2018['Margin'] >= 0, 1, 0)
merged_data_2019['Home Win'] = np.where(merged_data_2019['Margin'] >= 0, 1, 0)

# This is the ladder positions for every round in the year
ladder_2018 = ladder_position(merged_data_2018)
ladder_2019 = ladder_position(merged_data_2019)

# Merge the away team
merge_away_team = data_2018[['Team', 'Opponent', 'Round']]
merge_away_team.loc[merge_away_team['Round'].str.contains('R'), 'Round'] = merge_away_team['Round'].str.replace('R',
                                                                                                                '').astype(
    int)

ladder_2018 = pd.merge(ladder_2018,
                       merge_away_team,
                       how='left',
                       left_on=['Round Number', 'Team'], right_on=['Round', 'Team'])

merge_away_team = data_2019[['Team', 'Opponent', 'Round']]
merge_away_team.loc[merge_away_team['Round'].str.contains('R'), 'Round'] = merge_away_team['Round'].str.replace('R',
                                                                                                                '').astype(
    int)

ladder_2019 = pd.merge(ladder_2019,
                       merge_away_team,
                       how='left',
                       left_on=['Round Number', 'Team'], right_on=['Round', 'Team'])

# Again, shift by one week so we are looking at retrospective data
for index, column in enumerate(ladder_2018.columns[[9, 10, 11, 12]], 0):
    ladder_2018[column] = ladder_2018.groupby('Team', as_index=False)[column].transform(lambda x: x.shift(1))
    # ladder_2018[column] = ladder_2018.groupby('Team', as_index = False)[column].rolling(5, min_periods=1).mean().reset_index(0, drop = True)

for index, column in enumerate(ladder_2019.columns[[9, 10, 11, 12]], 0):
    ladder_2019[column] = ladder_2019.groupby('Team', as_index=False)[column].transform(lambda x: x.shift(1))
    # ladder_2019[column] = ladder_2019.groupby('Team', as_index = False)[column].rolling(5, min_periods=1).mean().reset_index(0, drop = True)

merged_data_2018['Round Number'] = merged_data_2018['Round Number'].astype(int)
merged_data_2019['Round Number'] = merged_data_2019['Round Number'].astype(int)

merge_data_home = pd.merge(merged_data_2018,
                           ladder_2018[['Round Number', 'Team', 'Ladder Position', 'Form', 'Points For Roll',
                                        'Points Against Roll']],
                           how='left',
                           left_on=['Round Number', 'Home Team'], right_on=['Round Number', 'Team']).rename(
    columns={"Points For Roll": "Pt_For_Home",
             'Points Against Roll': 'Pt_Against_Home',
             'Form': 'Form_Home',
             'Ladder Position': 'Ladder_Home'})

merge_data_all_2018 = pd.merge(merge_data_home,
                               ladder_2018[['Round Number', 'Team', 'Ladder Position', 'Form', 'Points For Roll',
                                            'Points Against Roll']],
                               how='left',
                               left_on=['Round Number', 'Away Team'], right_on=['Round Number', 'Team']).rename(
    columns={'Points For Roll': 'Pt_For_Away',
             'Points Against Roll': 'Pt_Against_Away',
             'Form': 'Form_Away',
             'Ladder Position': 'Ladder_Away'})

merge_data_all_2018['Ladder Diff'] = merge_data_all_2018['Ladder_Home'] - merge_data_all_2018['Ladder_Away']
merge_data_all_2018['Rolling Point Diff'] = merge_data_all_2018['Pt_For_Away'] - merge_data_all_2018['Pt_Against_Away']
merge_data_all_2018['Form Diff'] = merge_data_all_2018['Form_Home'] - merge_data_all_2018['Form_Away']

merge_data_home = pd.merge(merged_data_2019,
                           ladder_2019[['Round Number', 'Team', 'Ladder Position', 'Form', 'Points For Roll',
                                        'Points Against Roll']],
                           how='left',
                           left_on=['Round Number', 'Home Team'], right_on=['Round Number', 'Team']).rename(
    columns={'Points For Roll': 'Pt_For_Home',
             'Points Against Roll': 'Pt_Against_Home',
             'Form': 'Form_Home',
             'Ladder Position': 'Ladder_Home'})

merge_data_all_2019 = pd.merge(merge_data_home,
                               ladder_2019[['Round Number', 'Team', 'Ladder Position', 'Form', 'Points For Roll',
                                            'Points Against Roll']],
                               how='left',
                               left_on=['Round Number', 'Away Team'], right_on=['Round Number', 'Team']).rename(
    columns={'Points For Roll': 'Pt_For_Away',
             'Points Against Roll': 'Pt_Against_Away',
             'Form': 'Form_Away',
             'Ladder Position': 'Ladder_Away'})

merge_data_all_2019['Ladder Diff'] = merge_data_all_2019['Ladder_Home'] - merge_data_all_2019['Ladder_Away']
merge_data_all_2019['Rolling Point Diff'] = merge_data_all_2019['Pt_For_Away'] - merge_data_all_2019['Pt_Against_Away']
merge_data_all_2019['Form Diff'] = merge_data_all_2019['Form_Home'] - merge_data_all_2019['Form_Away']

############################################################
### 2 Test/tune machine learning algorithms on 2018 data ###
############################################################


# Let's train and validate the data on 2018 data
# normalise inputs and outputs
X_Train = normalise_0_1(
    merge_data_all_2018[['KI', 'MK', 'HB', 'TK', 'CL', 'IF',
                         'CP', 'UP', 'CM', 'MI',
                         'Ladder Diff', 'Rolling Point Diff', 'Form Diff']].dropna()
)

# Output matrix contents, put into BINS
Y_Train = categorise_margin(
    pd.DataFrame(merge_data_all_2018[['Margin']][9:198].dropna())
)

###########################################################################
### 3 Validate and evaluate statistical and machine learning algorithms ###
###########################################################################

X = X_Train
y = category_to_win_loss(categorise_margin(pd.DataFrame(merge_data_all_2018[['Margin']][9:198].dropna())))
# split into train/test sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5, random_state=2)

# fit a model
# Logistic
model_lr = LogisticRegression(solver='lbfgs')
model_lr.fit(trainX, np.ravel(trainY))
ns_fpr, ns_tpr, lr_fpr, lr_tpr = roc_compare(trainX, testX, trainY, testY, model_lr)
roc_auc(testX, testY, model_lr)

# NN
model_nn = MLPClassifier(hidden_layer_sizes=(4, 3, 6, 6, 4), learning_rate='adaptive', max_iter=10000)
model_nn.fit(trainX, np.ravel(trainY))
ns_fpr, ns_tpr, nn_fpr, nn_tpr = roc_compare(trainX, testX, trainY, testY, model_nn)
roc_auc(testX, testY, model_nn)

# set this flag to true if analysis on neural network nodes/layers should be run
nn_analysis = False
if nn_analysis:
    exec("nn_layer_analysis.py")
else:
    nn_layer_node = pd.read_csv("nn_layer_node/nn_output.csv")
    nn_layer_node['total nodes'] = nn_layer_node['Layer_1'] + nn_layer_node['Layer_2'] + nn_layer_node['Layer_3']


    # plot boxplots in pairs
    def set_box_color(bp, color):
        pyplot.setp(bp['boxes'], color=color)
        pyplot.setp(bp['whiskers'], color=color)
        pyplot.setp(bp['caps'], color=color)
        pyplot.setp(bp['medians'], color=color)


    tick_loc = []

    for count, value in enumerate(np.unique(nn_layer_node['total nodes'])):
        tick_loc.append(count * 3 + 1.5)
        set_train = nn_layer_node['Accuracy_Train_WL'].where(nn_layer_node['total nodes'] == value).dropna() * 100
        set_validation = nn_layer_node['Accuracy_Validation_WL'].where(
            nn_layer_node['total nodes'] == value).dropna() * 100
        bp = pyplot.boxplot(set_train, positions=[count * 3 + 1], widths=0.6)
        set_box_color(bp, '#2C7BB6')
        bp = pyplot.boxplot(set_validation, positions=[count * 3 + 2], widths=0.6)
        set_box_color(bp, '#D7191C')

    ticks = np.repeat(['9', '12', '15', '18', '21', '24', '27', '30', '33', '36', '39', '42', '45'], 1)
    pyplot.xticks(np.asarray(tick_loc), ticks)
    pyplot.xlim(0, 39)
    pyplot.ylim(0, 100)
    # draw temporary red and blue lines and use them to create a legend
    hB, = pyplot.plot([100, 100], 'b-')
    hR, = pyplot.plot([100, 100], 'r-')
    pyplot.legend((hB, hR), ('Test Set', 'Validation Set'))
    hB.set_visible(False)
    hR.set_visible(False)
    pyplot.xlabel('Total Nodes')
    pyplot.ylabel('% Correct Win/Loss')
    pyplot.plot([0,39],[50,50],'k--')
    pyplot.title('Neural Network Nodes vs Training/Validation Set Performance')

    pyplot.savefig('figs/boxcompare.png')
    show()



# Nearest Neighbours
n_nb = 30
model_nb_uni = neighbors.KNeighborsClassifier(n_nb, weights='uniform').fit(trainX, np.ravel(trainY))
ns_fpr, ns_tpr, nb_uni_fpr, nb_uni_tpr = roc_compare(trainX, testX, trainY, testY, model_nb_uni)
roc_auc(testX, testY, model_nb_uni)

# Nearest Neighbours V2
model_nb_dist = neighbors.KNeighborsClassifier(n_nb, weights='distance').fit(trainX, np.ravel(trainY))
ns_fpr, ns_tpr, nb_dist_fpr, nb_dist_tpr = roc_compare(trainX, testX, trainY, testY, model_nb_dist)
roc_auc(testX, testY, model_nb_dist)

# Lasso
model_rdmfst = RandomForestClassifier().fit(trainX, np.ravel(trainY))
ns_fpr, ns_tpr, rdmfst_fpr, rdmfst_tpr = roc_compare(trainX, testX, trainY, testY, model_rdmfst)
roc_auc(testX, testY, model_rdmfst)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.plot(nn_fpr, nn_tpr, marker='.', label='NN')
pyplot.plot(nb_uni_fpr, nb_uni_tpr, marker='.', label='Nearest Neighbours (uniform)')
pyplot.plot(nb_dist_fpr, nb_dist_tpr, marker='.', label='Nearest Neighbours (distance)')
pyplot.plot(rdmfst_fpr, rdmfst_tpr, marker='.', label='Random Forest')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# It would appear as though the LR performs the best
pyplot.savefig("figs/roc.png")

#######################################
### 4 Apply model to betting market ###
#######################################

# It would appear as though the LR performs the best
# This should be used to identify when there is a discrepancy between the probability of a team winning per the model
# And the probability that a team wins
# We will define this as 1/Odds. Uncertainty is already built into this equation, given that teams with equal odds will pay out <2:1
