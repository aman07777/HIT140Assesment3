# FIFA World Cup 2026 – Objective 2: Linear Regression (Assessment 3)

Two linear regression models built from pre-match information only:

## Folder layout
```
data/      matches_raw.csv, team_info.csv        (source data we compiled)
           matches_104.csv, team_matches_208.csv (datasets built by step 1)
scripts/   step1_build_datasets.py               build the two datasets (no future information used)
           step2_goal_difference_model.py        Task 2.1 model + model comparison
           step3_goals_scored_model.py           Task 2.2 model + model comparison
           step4_make_charts.py                  the two figures used in the report
           model_helpers.py                      small helper functions used by steps 2 and 3
outputs/   fig1_ranking_story.png, fig2_model_comparison.png, results/*.csv
```

## How to run
```
pip install -r requirements.txt
cd scripts
python step1_build_datasets.py
python step2_goal_difference_model.py
python step3_goals_scored_model.py
python step4_make_charts.py
```
