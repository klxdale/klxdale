import numpy as np
import pandas as pd

def ladder_position(data):

    # Create dataframe, unsorted
    data_home = data[['Round Number', 'Home Team', 'Home Win', 'home_score', 'away_score']].rename(columns = {"Home Team" : "Team", "Home Win": "Win", "home_score" : "Points For", "away_score" : "Points Against"})
    data_away = data[['Round Number', 'Away Team', 'Home Win', 'away_score', 'home_score']].rename(columns = {"Away Team" : "Team", "Home Win": "Win", "away_score" : "Points For", "home_score" : "Points Against"})
    data_away['Win'] = np.where(data_away['Win'] == 0, 1, 0)

    data_all = pd.concat([data_home, data_away], ignore_index = True)
    data_all['Round Number'] = data_all['Round Number'].astype(int)
    data_all = data_all.sort_values(['Round Number'])

    data = data_all.reset_index(0, drop = True)

    # Determine cumulative points for and against and percentage
    data['Cumulative Wins'] = data.groupby(['Team'])['Win'].cumsum().astype(int)
    data['Cumulative Points For'] = data.groupby('Team')['Points For'].cumsum().astype(int)
    data['Cumulative Points Against'] = data.groupby('Team')['Points Against'].cumsum().astype(int)
    data['Percentage'] = data['Cumulative Points For']/data['Cumulative Points Against'].astype(float)
    data['Round Number'] = data['Round Number'].astype(int)

    # Ladder
    ladder = data.sort_values(['Round Number','Cumulative Wins','Percentage'],ascending = True, kind = 'mergesort')
    ladder['Ladder Position'] = np.resize(np.flip(np.r_[1:19]),18*22)

    ladder['Form'] = ladder.groupby('Team')['Win'].rolling(5, min_periods=1).sum().reset_index(0,drop = True)
    ladder['Points For Roll'] = ladder.groupby('Team')['Points For'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    ladder['Points Against Roll'] = ladder.groupby('Team')['Points Against'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

    return ladder


