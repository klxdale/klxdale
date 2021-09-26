import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import numpy as np
import re

# This function scrapes data from a website

def get_afl_stats(url_link):

    # this is the data from the website, in HTML form
    url = requests.get(url_link)
    soup = bs(url.content, 'html.parser')

    # this extracts all of the individual tables
    divs = soup.select("div")

    for ii in np.linspace(0,51,18):

        # create data frame to store data
        data = pd.DataFrame(np.zeros((22, 25)),
                            columns=['Round', 'Team', 'Opponent', 'KI', 'MK', 'HB', 'DI', 'GL', 'BH', 'HO', 'TK', 'RB',
                                     'IF',
                                     'CL', 'CG', 'FF', 'FA', 'BR', 'CP', 'UP', 'CM', 'MI', '1P', 'BO', 'GA'])

        ii = int(ii)

        # extract current team data
        team = divs[ii]

        # This determines which is the name of the team we are extracting data for
        raw_data_header = []
        for tr in team.find_all("tr"):
            if tr.contents:
                raw_data_header.append(tr.find_next().text.split("\n"))

        raw_data = list()

        for idx, td in enumerate(team.find_all("td"), 1):
            if td.contents:
                raw_data.append(td.find_next().text.split("\n"))

        # Only include list items of length 1
        # We need to split into table 1 & 2
        table_1 = raw_data[0:307]
        if raw_data[350] == ['GF']:
            table_2 = raw_data[364:627]
        elif raw_data[336] == ['PF'] or raw_data[336] == ['GF']:
            table_2 = raw_data[350:613]
        elif raw_data[322] == ['SF'] or raw_data[322] == ['PF']:
            table_2 = raw_data[336:599]
        elif raw_data[308] == ['QF'] or raw_data[308] == ['EF']:
            table_2 = raw_data[322:585]
        else:
            table_2 = raw_data[308:571]


        # We need to add round 1 opponents
        first_team = re.search(">(.*?)<", str(team.find_all("td")[1]))[1]
        table_1.insert(0, list([first_team]))
        table_2.insert(0, list([first_team]))

        table_1 = [x[0] if len(x) == 1 and len(x[0]) < 20 else re.sub("\d", "", x[0])[1:len(x[0])] for x in table_1]
        table_2 = [x[0] if len(x) == 1 and len(x[0]) < 20 else re.sub("[0-9]", "", re.search("\d(.*?)\d", str(table_2[12]))[0])
                   for x in table_2]

        # reshape tables so that they can be shoehorned into the main table
        table_1_rs = np.reshape(np.array(table_1), (-1, 14))
        table_2_rs = np.reshape(np.array(table_2), (-1, 12))

        # Go through tables 1 and 2, and append the correct data to the correct index in the data table
        data[['Opponent', 'Round', 'KI', 'MK', 'HB', 'DI', 'GL', 'BH', 'HO', 'TK', 'RB', 'IF', 'CL', 'CG']] = np.asmatrix(
            table_1_rs[:, 0:14])
        data[['FF', 'FA', 'BR', 'CP', 'UP', 'CM', 'MI', '1P', 'BO', 'GA']] = table_2_rs[:, 2:12]

        for index, column in enumerate(data.columns[[2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], 0):
            data[column] = pd.Series(table_1_rs[:, index])

        for index, column in enumerate(data.columns[[15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], 2):
            data[column] = pd.Series(table_2_rs[:, index])

        data['Team'] = re.sub("\['", "", str(raw_data_header[0]).split(" ")[0])

        if ii == 0:
            data_all = data
        else:
            data_all = data_all.append(pd.DataFrame(data=data), ignore_index=True)


    # Fix team names
    def f(x):
        return {
            'St': 'St Kilda',
            'Gold': 'Gold Coast',
            'Greater': 'Greater Western Sydney',
            'North': 'North Melbourne',
            'Port': 'Port Adelaide',
            'West': 'West Coast',
            'Western': 'Western Bulldogs'
        }.get(x,x)

    # Apply this to the whole dataframe
    data_all['Team'] = data_all['Team'].apply(f,'columns')

    return data_all


def get_afl_scores(url_link):

    data = pd.read_csv(url_link)

    # Fix team names
    def f(x):
        return {
            'Brisbane Lions': 'Brisbane',
            'Gold Coast Suns': 'Gold Coast',
            'GWS Giants': 'Greater Western Sydney',
            'West Coast Eagles': 'West Coast',
            'Geelong Cats': 'Geelong',
            'Sydney Swans': 'Sydney',
            'Adelaide Crows': 'Adelaide'
        }.get(x, x)

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # Apply this to the whole dataframe
    data['Away Team'] = data['Away Team'].apply(f, 'columns')
    data['Home Team'] = data['Home Team'].apply(f, 'columns')

    # Filter to home and away games
    data = data[(data['Round Number'] != "Finals W1") & (data['Round Number'] != "Semi Finals") & (data['Round Number'] != "Prelim Finals") & (data['Round Number'] != "Grand Final")]

    # Add score margin and home and away scores
    data = data.assign(Round = lambda x: "R" + x['Round Number'],
                       home_score = lambda x: x['Result'].str.split('-', expand = True)[0].astype(int),
                       away_score = lambda x: x['Result'].str.rsplit('-', expand = True)[1].astype(int))

    data['Margin'] = np.zeros(198)
    for ii in range(198):
        data['Margin'][ii] = eval(str(data['Result'][ii]))

    return data
