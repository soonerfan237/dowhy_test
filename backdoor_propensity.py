import pygraphviz as pgv
print(pgv.__version__)

import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

gamelogs = pd.read_pickle("./nfl_gamelogs_stats_bkp.pkl")
gamelogs['result'] = (gamelogs['tm_score'] > gamelogs['opp_score']).astype(int)
gamelogs['turnover_result'] = (gamelogs['tm_turnovers'] > gamelogs['opp_turnovers']).astype(int)

temp_lower_bound = gamelogs['temperature'].quantile(0.05)
temp_upper_bound = gamelogs['temperature'].quantile(0.95)

gamelogs['extreme_temperature'] = gamelogs['temperature'].apply(lambda x: 1 if x <= temp_lower_bound or x >= temp_upper_bound else 0).astype(int)

print(gamelogs['extreme_temperature'].unique())
print(gamelogs['extreme_temperature'].dtype)

gamelogs = gamelogs[['tm_score','opp_score','extreme_temperature',
    'tm_rush_att', 'tm_rush_yds',
    'tm_rush_tds', 'tm_pass_att',
    'tm_pass_yds', 'tm_pass_tds',
    'tm_turnovers', 'tm_fumbles',
    'tm_fumbles_lost', 'tm_first_downs',
    'opp_rush_att', 'opp_rush_yds',
    'opp_rush_tds', 'opp_pass_att',
    'opp_pass_yds', 'opp_pass_tds',
    'opp_turnovers', 'opp_fumbles',
    'opp_fumbles_lost', 'opp_first_downs',
    'result','turnover_result'
    ]]

gamelogs = gamelogs.dropna()
#gamelogs = gamelogs.fillna(gamelogs.mean())

gamelogs.info()
pd.set_option('display.max_columns', None)
#print(gamelogs.describe())
gamelogs.head(5)

dag = """
digraph {

extreme_temperature;

tm_rush_att;
tm_rush_yds;
tm_rush_tds;
tm_pass_att;
tm_pass_yds;
tm_pass_tds;
tm_turnovers;
tm_score;

opp_rush_att;
opp_rush_yds;
opp_rush_tds;
opp_pass_att;
opp_pass_yds;
opp_pass_tds;
opp_turnovers;
opp_score;

turnover_result;
result;

extreme_temperature -> tm_rush_att;
extreme_temperature -> opp_rush_att;
extreme_temperature -> tm_score;
extreme_temperature -> opp_score;

tm_rush_att -> tm_rush_yds;
tm_rush_yds -> tm_rush_tds;
tm_rush_tds -> tm_score;
opp_rush_att -> opp_rush_yds;
opp_rush_yds -> opp_rush_tds;
opp_rush_tds -> opp_score;

tm_pass_att -> tm_pass_yds;
tm_pass_yds -> tm_pass_tds;
tm_pass_tds -> tm_score;
opp_pass_att -> opp_pass_yds;
opp_pass_yds -> opp_pass_tds;
opp_pass_tds -> opp_score;

tm_rush_yds -> tm_first_downs;
tm_pass_yds -> tm_first_downs;
tm_first_downs -> tm_score
opp_rush_yds -> opp_first_downs;
opp_pass_yds -> opp_first_downs;
opp_first_downs -> opp_score

tm_rush_att -> tm_fumbles;
tm_fumbles -> tm_fumbles_lost;
tm_fumbles_lost -> tm_turnovers;
tm_turnovers -> tm_score;
opp_rush_att -> opp_fumbles;
opp_fumbles -> opp_fumbles_lost;
opp_fumbles_lost -> opp_turnovers;
opp_turnovers -> opp_score;

tm_turnovers -> turnover_result;
opp_turnovers -> turnover_result;

turnover_result -> tm_score;

tm_score -> result;
opp_score -> result;

}
"""

model=CausalModel(
    data = gamelogs,
    treatment='turnover_result',
    outcome='tm_score',
    graph=dag
)

identified_estimand = model.identify_effect()
print(identified_estimand)

desired_effect = "ate"

estimate = model.estimate_effect(
    identified_estimand,
    #method_name="backdoor.linear_regression"
    method_name="backdoor.propensity_score_matching"
    #,target_units=desired_effect
    #,method_params={"weighting_scheme":"ips_weight"}
)

print("Causal Estimate is " + str(estimate.value))

print("done")