"""
STEP 4 - Make the two figures used in the report.

Figure 1 (the story):  how much does the FIFA ranking gap matter?
    (a) Task 2.1  scatter plot of ranking gap vs goal difference, with the regression line
    (b) Task 2.2  average goals scored for five bands of ranking advantage
Figure 2 (the comparison): cross-validated error of every model we tried (lower = better)

Colour idea: one blue for the model/story we care about, grey for everything else.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

BLUE, DARK_BLUE, GREY = "#2a78d6", "#184f95", "#a9a8a2"
TEXT, LIGHT_TEXT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "axes.edgecolor": GRID,
                     "axes.labelcolor": LIGHT_TEXT, "xtick.color": LIGHT_TEXT, "ytick.color": LIGHT_TEXT})


def tidy(ax):
    """Light grid, no top/right border."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


match_data = pd.read_csv("../data/matches_104.csv")
team_data = pd.read_csv("../data/team_matches_208.csv")

# ================= Figure 1 =================
fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) scatter + regression line with 95% confidence band
x, y = match_data["rank_diff"], match_data["goal_diff"]
line = sm.OLS(y, sm.add_constant(x)).fit()
x_line = np.linspace(-85, 85, 100)
band = line.get_prediction(sm.add_constant(x_line)).summary_frame()
left.fill_between(x_line, band["mean_ci_lower"], band["mean_ci_upper"], color=BLUE, alpha=0.12, linewidth=0)
left.plot(x_line, band["mean"], color=DARK_BLUE, linewidth=2)
left.scatter(x, y, s=34, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.9, zorder=3)

# label three surprising matches: (team_a, team_b) -> where to put the text
labels = {("Germany", "Curacao"): (62, 7.0, "right"),
          ("Ecuador", "Curacao"): (86, -2.8, "right"),
          ("Tunisia", "Japan"): (-62, -5.5, "left")}
for m in match_data.itertuples():
    if (m.team_a, m.team_b) in labels:
        tx, ty, align = labels[(m.team_a, m.team_b)]
        text = f"{m.team_a} {m.score_a}–{m.score_b} {m.team_b}".replace("Curacao", "Curaçao")
        left.annotate(text, (m.rank_diff, m.goal_diff), xytext=(tx, ty), ha=align, va="center", fontsize=7,
                      color=TEXT, arrowprops=dict(arrowstyle="-", color=LIGHT_TEXT, linewidth=0.6))

decisive = match_data[match_data["goal_diff"] != 0]
upsets = (np.sign(decisive["rank_diff"]) != np.sign(decisive["goal_diff"])).sum()
left.text(84, -5.6, f"upset: lower-ranked\nside won ({upsets} of {len(decisive)})", ha="right", va="center",
          fontsize=7, color=LIGHT_TEXT, style="italic")
left.set_xlim(-88, 88)
left.set_ylim(-6.4, 7.6)
left.set_xlabel("Ranking gap: Team B rank − Team A rank (places)")
left.set_ylabel("Goal difference (Team A − Team B)")
left.set_title("(a) 2.1  Higher rank, bigger winning margin",
               loc="left", fontsize=8.2, fontweight="bold", color=TEXT)
tidy(left)

# (b) average goals by band of ranking advantage
bands = pd.cut(team_data["rank_advantage"], [-100, -30, -10, 10, 30, 100], labels=range(5))
summary = team_data.groupby(bands, observed=True)["goals_scored"].agg(["mean", "std", "count"])
summary["ci"] = stats.t.ppf(0.975, summary["count"] - 1) * summary["std"] / np.sqrt(summary["count"])
positions = np.arange(5)
overall = team_data["goals_scored"].mean()

right.axhline(overall, color=GREY, linewidth=0.9)
right.text(4.45, overall + 0.06, f"all team-matches: {overall:.2f}", ha="right", fontsize=7, color=LIGHT_TEXT)
right.plot(positions, summary["mean"], color=BLUE, linewidth=2)
right.errorbar(positions, summary["mean"], yerr=summary["ci"], fmt="none", ecolor=DARK_BLUE, capsize=3, elinewidth=1)
right.scatter(positions, summary["mean"], s=46, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
for i, count in enumerate(summary["count"]):
    right.text(i, 0.05, f"n={count}", ha="center", fontsize=7, color=LIGHT_TEXT)
for i in (0, 4):   # print the value at both ends of the line
    right.text(i, summary["mean"].iloc[i] + summary["ci"].iloc[i] + 0.07, f"{summary['mean'].iloc[i]:.2f}",
               ha="center", fontsize=7.5, color=TEXT)
right.set_xticks(positions)
right.set_xticklabels(["≤ −30", "−29 to\n−10", "−9 to 9", "10 to 29", "≥ 30"], fontsize=7.2)
right.set_xlim(-0.5, 4.5)
right.set_ylim(0, 3.1)
right.set_xlabel("Ranking advantage: opp. rank − own rank")
right.set_ylabel("Mean goals scored (95% CI)")
right.set_title("(b) 2.2  Advantage pays – then levels off", loc="left", fontsize=8.2, fontweight="bold", color=TEXT)
tidy(right)

fig.tight_layout(w_pad=1.6)
fig.savefig("../outputs/fig1_ranking_story.png")
plt.close(fig)

# ================= Figure 2 =================
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35))
panels = [("task21_model_comparison.csv", "(a) 2.1  Goal difference"),
          ("task22_model_comparison.csv", "(b) 2.2  Goals scored")]
for ax, (file, title) in zip(axes, panels):
    results = pd.read_csv("../outputs/results/" + file).sort_values("RMSE", ascending=False).reset_index(drop=True)
    best = results["RMSE"].idxmin()
    for i, row in results.iterrows():
        is_required_model = row["model"] == "OLS (8 variables)"
        colour = BLUE if is_required_model else GREY
        ax.plot([row["RMSE"] - row["RMSE_sd"], row["RMSE"] + row["RMSE_sd"]], [i, i], color=colour, linewidth=1.4)
        ax.scatter(row["RMSE"], i, s=44, color=colour, edgecolor="white", linewidth=1.2, zorder=3)
        if is_required_model or i == best:
            ax.text(row["RMSE"] + row["RMSE_sd"] + 0.012, i, f"{row['RMSE']:.2f}", va="center", fontsize=7.5, color=TEXT)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(results["model"], fontsize=7.5)
    for label in ax.get_yticklabels():
        if label.get_text() == "OLS (8 variables)":
            label.set_color(DARK_BLUE)
            label.set_fontweight("bold")
    ax.set_xlim(results["RMSE"].min() - 0.09, results["RMSE"].max() + 0.12)
    ax.set_ylim(-0.6, len(results) - 0.4)
    ax.set_xlabel("CV RMSE, goals (±1 SD, 30 repeats)")
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", color=TEXT)
    tidy(ax)
    ax.grid(axis="y", visible=False)

fig.tight_layout(w_pad=1.0)
fig.savefig("../outputs/fig2_model_comparison.png")
plt.close(fig)

# ================= Figure 3 (exploratory analysis) =================
# Correlation of every explanatory variable with the target. This tells us where the signal is
# before we fit any model, and motivates the ranking-only and capped-ranking models.
features_21 = ["rank_diff", "host_advantage", "same_confederation", "stage_num",
               "titles_diff", "rest_diff", "points_diff", "conf_strength_diff"]
features_22 = ["rank_advantage", "is_host", "same_confederation", "stage_num",
               "own_gf_form", "opp_ga_form", "team_is_european", "opp_is_european"]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3))
for ax, (data, features, target, title) in zip(axes, [(match_data, features_21, "goal_diff", "(a) 2.1  Goal difference"),
                                                      (team_data, features_22, "goals_scored", "(b) 2.2  Goals scored")]):
    corr = data[features].corrwith(data[target]).sort_values()
    colours = [BLUE if name in ("rank_diff", "rank_advantage") else GREY for name in corr.index]
    ax.barh(range(len(corr)), corr.values, color=colours, height=0.6)
    for i, value in enumerate(corr.values):
        ax.text(value + (0.012 if value >= 0 else -0.012), i, f"{value:.2f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=7, color=TEXT)
    ax.axvline(0, color=LIGHT_TEXT, linewidth=0.8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=7.2)
    ax.set_xlim(-0.34, 0.75)
    ax.set_xlabel("Correlation with the target")
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", color=TEXT)
    tidy(ax)
    ax.grid(axis="y", visible=False)
fig.tight_layout(w_pad=1.0)
fig.savefig("../outputs/fig3_correlations.png")
plt.close(fig)
print("Saved fig1_ranking_story.png, fig2_model_comparison.png and fig3_correlations.png")
