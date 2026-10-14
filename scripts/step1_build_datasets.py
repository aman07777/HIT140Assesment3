"""
STEP 1 - Build the two regression datasets.

Inputs  (data/):  matches_raw.csv  (104 match results)
                  team_info.csv    (FIFA rank, confederation, host, World Cup titles)
Outputs (data/):  matches_104.csv       one row per match      -> Task 2.1 (goal difference)
                  team_matches_208.csv  one row per team-match -> Task 2.2 (goals scored)

Every explanatory variable is known BEFORE kick-off:
  - FIFA rank, confederation, host status, World Cup titles  (fixed facts)
  - stage of the tournament and days of rest (from the schedule)
  - form = goals scored / conceded per game, and points won, in the team's EARLIER matches only

The brief allows at most 4 explanatory variables to be shared between Task 2.1 and Task 2.2.
We share exactly 4:  ranking gap, host nation, same confederation, tournament stage.
The other 4 variables in each task are different (see the lists in step 2 and step 3).

Squad age and squad size were NOT used: FBref calculates them over the whole
tournament, so they would leak information from the matches we want to predict.
"""
import pandas as pd

matches = pd.read_csv("../data/matches_raw.csv", parse_dates=["date"])
teams = pd.read_csv("../data/team_info.csv").set_index("team")

STAGE_NUMBER = {"Group": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "Bronze": 5, "Final": 5}
PRIOR_GOALS = 1.3     # typical World Cup goals per team per match, used before a team has played
PRIOR_WEIGHT = 1      # the prior counts as one pretend game
MAX_REST = 7          # rest days are capped at 7 (also used for a team's first match)

# Average FIFA rank of each confederation (lower number = stronger confederation)
confederation_rank = teams.groupby("confederation")["fifa_rank"].mean()

# Running record for each team: goals for, goals against, points, games played, date of last game
record = {team: {"gf": 0, "ga": 0, "points": 0, "games": 0, "last_date": None} for team in teams.index}


def team_features_before_match(team, match_date):
    """Rest days and form for a team, using only matches played before this one."""
    r = record[team]
    rest = MAX_REST if r["last_date"] is None else min((match_date - r["last_date"]).days, MAX_REST)
    gf_form = (PRIOR_GOALS * PRIOR_WEIGHT + r["gf"]) / (PRIOR_WEIGHT + r["games"])
    ga_form = (PRIOR_GOALS * PRIOR_WEIGHT + r["ga"]) / (PRIOR_WEIGHT + r["games"])
    return {"rest": rest, "gf_form": gf_form, "ga_form": ga_form, "points": r["points"]}


match_rows = []      # dataset for Task 2.1
team_match_rows = [] # dataset for Task 2.2

for match_id, m in enumerate(matches.itertuples(), start=1):
    a, b = teams.loc[m.team_a], teams.loc[m.team_b]
    fa = team_features_before_match(m.team_a, m.date)   # features are read BEFORE the record is updated
    fb = team_features_before_match(m.team_b, m.date)
    stage = STAGE_NUMBER[m.stage]
    same_conf = int(a.confederation == b.confederation)

    match_rows.append({
        "match_id": match_id, "date": m.date.date(), "team_a": m.team_a, "team_b": m.team_b,
        "score_a": m.score_a, "score_b": m.score_b, "goal_diff": m.score_a - m.score_b,
        # --- 4 variables shared with Task 2.2 ---
        "rank_diff": b.fifa_rank - a.fifa_rank,            # positive = team A is ranked higher
        "host_advantage": a.is_host - b.is_host,
        "same_confederation": same_conf,
        "stage_num": stage,
        # --- 4 variables used only in Task 2.1 ---
        "titles_diff": a.prior_wc_titles - b.prior_wc_titles,
        "rest_diff": fa["rest"] - fb["rest"],
        "points_diff": fa["points"] - fb["points"],        # tournament points before the match
        "conf_strength_diff": confederation_rank[b.confederation] - confederation_rank[a.confederation],
    })

    for team, opp, own, opp_own, goals in ((m.team_a, m.team_b, fa, fb, m.score_a),
                                           (m.team_b, m.team_a, fb, fa, m.score_b)):
        t, o = teams.loc[team], teams.loc[opp]
        team_match_rows.append({
            "match_id": match_id, "date": m.date.date(), "team": team, "opponent": opp,
            "goals_scored": goals,
            # --- 4 variables shared with Task 2.1 ---
            "rank_advantage": o.fifa_rank - t.fifa_rank,      # positive = this team is ranked higher
            "is_host": t.is_host,
            "same_confederation": same_conf,
            "stage_num": stage,
            # --- 4 variables used only in Task 2.2 ---
            "own_gf_form": own["gf_form"],        # how many goals this team has been scoring
            "opp_ga_form": opp_own["ga_form"],    # how many goals the opponent has been conceding
            "team_is_european": int(t.confederation == "UEFA"),
            "opp_is_european": int(o.confederation == "UEFA"),
        })

    # Only now add this match to each team's record
    for team, scored, conceded in ((m.team_a, m.score_a, m.score_b), (m.team_b, m.score_b, m.score_a)):
        record[team]["points"] += 3 if scored > conceded else (1 if scored == conceded else 0)
        record[team]["gf"] += scored
        record[team]["ga"] += conceded
        record[team]["games"] += 1
        record[team]["last_date"] = m.date

data_21 = pd.DataFrame(match_rows)
data_22 = pd.DataFrame(team_match_rows)

# Quick checks against the assignment brief
assert len(data_21) == 104 and len(data_22) == 208
assert data_21.shape[1] - 7 == 8      # 7 id/target columns + 8 explanatory variables
assert data_22.shape[1] - 5 == 8      # 5 id/target columns + 8 explanatory variables
assert not data_21.isna().any().any() and not data_22.isna().any().any()

data_21.to_csv("../data/matches_104.csv", index=False)
data_22.to_csv("../data/team_matches_208.csv", index=False)
print("Saved matches_104.csv", data_21.shape, "and team_matches_208.csv", data_22.shape)
