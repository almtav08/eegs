# --- 6. VISUALIZACIÓN DIFERENCIAS: CONSISTENTS PASS vs FAIL ---
print("\n--- 6. Visualizing Consistents: Pass vs Fail Differences ---")

readable_metrics = {
    "total_clicks": "Total Clicks",
    "total_sessions": "Total Sessions",
    "avg_delta_t_sec": "Avg. Time\nBtw Clicks (s)",  # <--- Dos líneas
    "avg_clicks_per_item": "Avg. Clicks\nper Resource",  # <--- Dos líneas
    "std_dev_daily_clicks": "Daily Clicks\nStd. Dev.",  # <--- Dos líneas
    "percentile_80_delta_t": "80th Percentile\nTime (s)",  # <--- Dos líneas
    "activity_span_days": "Activity Span\n(Days)",
    "num_active_days": "Active Days",
    "regularity_ratio": "Regularity Ratio",
    "max_consecutive_days": "Max Consecutive\nDays",
    "avg_inactive_days": "Avg. Inactive\nDays",
    "std_dev_hour_session_start": "Session Start\nHour SD",
    "ratio_short_breaks": "Short Breaks\nRatio",
    "first_week_ratio": "First Week Ratio",
    "procrastination_ratio": "Procrastination\nRatio",
    "weekend_ratio": "Weekend Ratio",
    "night_owl_ratio": "Night Owl Ratio",
    "mid_course_ratio": "Mid-Course Ratio",
    "activity_slope": "Activity Trend\n(Slope)",
    "num_unique_items": "Unique Resources\nVisited",
    "zapping_ratio": "Zapping Ratio\n(<1min)",
    "fixation_ratio": "Fixation Ratio",
    "coverage_ratio": "Course Coverage\nRatio",
    "progress_ratio": "Linear Progress\nRatio",
    "backward_prereq_ratio": "Backward Prereq.\nRatio",
    "revisit_ratio": "Revisit Ratio",
}

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Para guardar figuras sin mostrar

# Datos JSON proporcionados
with open("models/pct_diff.json", "r") as f:
    diff_data = json.load(f)

# Crear DataFrame
df_diff = pd.DataFrame(list(diff_data.items()), columns=["Metric_Raw", "Pct_Diff"])

# Mapear nombres técnicos a nombres legibles
df_diff["Metric_Readable"] = df_diff["Metric_Raw"].map(readable_metrics)
df_diff["Metric_Readable"] = df_diff["Metric_Readable"].fillna(df_diff["Metric_Raw"])

# Ordenar por diferencia
df_diff = df_diff.sort_values("Pct_Diff", ascending=True).reset_index(drop=True)

# Colores
colors = ["#d62728" if x < 0 else "#2ca02c" for x in df_diff["Pct_Diff"]]

# Índice del valor máximo (First Week Ratio)
max_val_idx = df_diff["Pct_Diff"].idxmax()

# --- FIGURA CON EJE PARTIDO ---
fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    sharey=True,
    figsize=(16, 15),
    gridspec_kw={"width_ratios": [4, 1]},
)

y_pos = np.arange(len(df_diff))

# --- EJE IZQUIERDO (RANGO PRINCIPAL) ---
ax1.barh(
    y_pos,
    df_diff["Pct_Diff"],
    color=colors,
    edgecolor="black",
    alpha=0.8,
)
ax1.set_xlim(-20, 50)
ax1.axvline(0, color="black", linewidth=1.2)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_diff["Metric_Readable"], fontsize=16)
ax1.set_xlabel(
    "% Difference (Positive = Higher in Passing Students)",
    fontsize=16,
    fontweight="bold",
)
ax1.grid(axis="x", linestyle="--", alpha=0.6)

# --- EJE DERECHO (OUTLIER) ---
ax2.barh(
    y_pos,
    df_diff["Pct_Diff"],
    color=colors,
    edgecolor="black",
    alpha=0.8,
)
ax2.set_xlim(300, 400)
ax2.set_xticks([300, 350, 400])
ax2.set_xlabel("Outlier scale", fontsize=14)
ax2.tick_params(labelleft=False)

# Ocultar spines entre ejes
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

""" # Marcas diagonales del eje partido
d = 0.015
kwargs = dict(transform=ax1.transAxes, color="black", clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) """

# --- ETIQUETAS DE TEXTO ---
for i, val in enumerate(df_diff["Pct_Diff"]):
    # Caso especial: First Week Ratio (outlier)
    if i == max_val_idx:
        ax2.text(
            val - 5,
            i,
            f"{val:+.1f}%",
            va="center",
            ha="right",
            fontsize=16,
            fontweight="bold",
            color="white",
        )
    else:
        if val >= 0:
            ax1.text(
                val + 1.5,
                i,
                f"{val:+.1f}%",
                va="center",
                ha="left",
                fontsize=14,
                color="black",
                fontweight="bold",
            )
        else:
            ax1.text(
                1.5,
                i,
                f"{val:+.1f}%",
                va="center",
                ha="left",
                fontsize=14,
                color="black",
                fontweight="bold",
            )

# Título general
fig.suptitle(
    "Consistents Archetype: Differentiating Factors (Pass vs. Fail)",
    fontsize=18,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig("figs/cons_diff.eps", format="eps")
plt.close()
