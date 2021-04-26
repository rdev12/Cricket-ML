import numpy as np
import pandas as pd
from imutils import paths
from sklearn.model_selection import train_test_split
import csv

def train_data(input):
    dataPaths = list(paths.list_files('.\ipl_csv2'))
    dataPaths.pop(len(dataPaths) - 2)
    dataPaths.pop(len(dataPaths) - 1)

    p_team = str(input.iloc[0]['batting_team']) 
    opp_team = str(input.iloc[0]['bowling_team'])
        
    def match_history(p_team, opp_team, num):
        req_df = []
        
        for match_file in dataPaths:
            try:
                df = pd.read_csv(match_file, index_col='match_id')
                df = preprocess(df)

                inn1 = df[df['innings'] == 1]
                inn2 = df[df['innings'] == 2]


                if (not inn1.empty and inn1.iloc[0]['batting_team'] == p_team or p_team == -1):
                    if (inn1.iloc[0]['bowling_team'] == opp_team or opp_team == -1):
                        if len(req_df) < num or num == -1:
                            req_df.append(df)
                elif (not inn2.empty and inn2.iloc[0]['batting_team'] == p_team or p_team == -1):
                    if (inn2.iloc[0]['bowling_team'] == opp_team or opp_team == -1):
                        if len(req_df) < num or num == -1:
                            req_df.append(df)

            except Exception as e:
                print(e)

        return req_df


    def preprocess(df):
        pp_df = df[df['ball'] < 6.1]
        pp_df = pp_df[pp_df['innings'] <= 2]
        pp_df['total_runs'] = pp_df['runs_off_bat'] + pp_df['extras']

        return pp_df


    df_list = match_history(p_team, -1, -1)


    #read the rating files
    def read_files(fileName, n_features):
        dictionary = {}
        with open(fileName, mode='r') as infile:
            reader = csv.reader(infile)

            for row in reader:
                for i in range(1, n_features + 1):
                    dictionary.setdefault(row[0], []).append(row[i])
        
        return dictionary

    stad_rating = read_files('stadium_rating.csv', 1)

    for k in stad_rating:
        stad_rating[k] = float(stad_rating[k][0])

    batsmen_record = read_files('batsmen_record.csv', 2)

    for k in batsmen_record:
        for (i, p) in enumerate(batsmen_record[k]):
            batsmen_record[k][i] = [int(s) for s in p[1:-1].split(',')]

    bowlers_record = read_files('bowlers_record.csv', 4)

    for k in bowlers_record:
        for (i, p) in enumerate(bowlers_record[k]):
            bowlers_record[k][i] = [int(s) for s in p[1:-1].split(',')]

    #creates feature values based on ratings
    def get_venue_score(venue):
        return stad_rating[venue]

    def get_batsmen_scores(u_bat):
        rating = []

        for batsman in u_bat:
            rating.append(np.mean(batsmen_record[batsman][0]))
        
        return rating
        
    def get_bowlers_scores(u_bowlers):

        rating = []

        for bowler in u_bowlers:
            try:
                rating.append(np.mean(bowlers_record[bowler][0]))
            except Exception as e:
                rating.append(30)
        
        return rating

    def get_wickets(u_bat):
        return len(u_bat) - 2
        

    #create training set by looping through the narrowed down matches
    X = pd.DataFrame()
    Y = pd.DataFrame()

    for df in df_list:
        match_id = df.index[0]

        #innings for which we are going to use for training (when the team is batting)
        inn = df.loc[(df['batting_team'] == p_team)]['innings'].iloc[0] 
        df_train_inn = df.loc[(df['innings'] == inn)]

        runs_x = df_train_inn['total_runs'].sum() #runs of training innings
        #runs_target = df.loc[(df['innings'] == 3 - inn)]['total_runs'].sum() #runs of other innings

        #convert the df into usable training data with required features
        # to add team rating: 'team_rating':[get_team_rating(df_train_inn)], 
        #venue = df_train_inn.iloc[1]['venue']
        u_bat = list(set(df_train_inn['striker'].unique()) | set(df_train_inn['non_striker'].unique()))
        u_bowlers = list(df_train_inn['bowler'].unique())

        temp = pd.DataFrame({'batsmen_rating': [np.sum(np.array(get_batsmen_scores(u_bat)))], 'bowler_rating': [np.sum(np.array(get_bowlers_scores(u_bowlers)))], 'wickets': [get_wickets(u_bat)]})
        
        X = X.append(temp)
        Y = Y.append(pd.DataFrame({'runs': [runs_x]}))

    
    #venue = df.iloc[0]['venue']
    u_bat = list(input.iloc[0]['batsmen'].split(','))
    u_bowlers = list(input.iloc[0]['bowlers'].split(','))
    #p_team = str(df.iloc[0]['batting_team'])
    #opp_team = str(df.iloc[0]['bowling_team'])

    temp = pd.DataFrame({'batsmen_rating': [np.sum(np.array(get_batsmen_scores(u_bat)))], 'bowler_rating': [np.sum(np.array(get_bowlers_scores(u_bowlers)))], 'wickets': [get_wickets(u_bat)]})

    return X, Y, temp