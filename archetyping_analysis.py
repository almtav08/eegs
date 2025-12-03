import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import json
from collections import Counter
import os
import joblib

matplotlib.use("Agg")  # Para guardar figuras sin mostrar

# Definiciones de Ritmo (Delta T en segundos)
ZAPPING_THRESHOLD = 60  # 1 minuto
SESSION_THRESHOLD = 1800  # 30 minutos
PAUSA_ESTUDIO_MIN = 60  # 1 minuto
PAUSA_ESTUDIO_MAX = 900  # 15 minutos

# --- 1. Carga y Preparación de tus Datos Reales ---
print("--- 1. Loading your real data ---")

archivo_de_logs = "database/logs.csv"
archivo_de_notas = "database/grades.csv"

try:
    df_clicks_raw = pd.read_csv(archivo_de_logs)
    df_outcomes_raw = pd.read_csv(archivo_de_notas)
except FileNotFoundError:
    print(f"Error: Files '{archivo_de_logs}' or '{archivo_de_notas}' not found.")
    exit()

# --- Adaptación de Nombres de Columnas ---
print("Adapting column names...")
df_clicks = df_clicks_raw.rename(
    columns={"user": "student_id", "item": "resource_id", "time": "timestamp"}
)

df_outcomes = df_outcomes_raw.rename(columns={"user": "student_id", "grade": "outcome"})

# --- Conversión de Tipo de Datos ---
print("Converting 'timestamp' to datetime format...")
try:
    df_clicks["timestamp"] = pd.to_datetime(df_clicks["timestamp"])
except Exception as e:
    print(f"Error converting 'timestamp'. Details: {e}")
    exit()

# Filtrado de usuarios comunes
usuarios_logs = set(df_clicks["student_id"].unique())
usuarios_notas = set(df_outcomes["student_id"].unique())
usuarios_comunes = usuarios_logs.intersection(usuarios_notas)

print(f"Analyzing {len(usuarios_comunes)} students present in both files.")
df_clicks = df_clicks[df_clicks["student_id"].isin(usuarios_comunes)]
df_outcomes = df_outcomes[df_outcomes["student_id"].isin(usuarios_comunes)]

# Carga de Grafos
try:
    with open("database/forward_graph.json", "r") as f:
        prereq_map = {int(k): v for k, v in json.load(f).items()}
    with open("database/remedial_graph.json", "r") as f:
        revisit_map = {int(k): v for k, v in json.load(f).items()}
    print("Graphs loaded.")
except FileNotFoundError:
    print("WARNING: Graph files not found in database/ folder.")
    prereq_map = {}
    revisit_map = {}

# --- Cálculo de Métricas Globales del Curso ---
total_items_set = set()
for m in [prereq_map, revisit_map]:
    for k, v in m.items():
        total_items_set.add(int(k))
        for val in v:
            total_items_set.add(int(val))
total_unique_items_in_course = len(total_items_set)

# --- 2. Finding the Course "Pulse" ---
print("\n--- 2. Finding the Course Pulse ---")
all_clicks_by_day = df_clicks.set_index("timestamp").resample("D").size()
peak_dates = all_clicks_by_day.nlargest(2).index
print(f"Peak activity dates inferred: {[d.date() for d in peak_dates]}")

plt.figure(figsize=(15, 6))
all_clicks_by_day.plot(kind="line")
plt.title("Course Pulse (Aggregated Activity)")
plt.savefig("figs/pulso_curso.png")

# --- 3. Feature Engineering ---
print("\n--- 3. Creating Behavioral Features ---")
course_start_date = df_clicks["timestamp"].min()
course_end_date = df_clicks["timestamp"].max()
course_duration_days = (course_end_date - course_start_date).days


def calculate_features_per_student(group):
    group = group.sort_values("timestamp")
    total_clicks = len(group)

    delta_t = group["timestamp"].diff().dt.total_seconds()
    total_sessions = (delta_t > SESSION_THRESHOLD).sum() + 1
    active_deltas = delta_t[delta_t <= SESSION_THRESHOLD]
    avg_delta_t = active_deltas.mean() if not active_deltas.empty else 0
    zapping_ratio = (
        (active_deltas <= ZAPPING_THRESHOLD).mean() if not active_deltas.empty else 0
    )
    percentil_80_delta_t = active_deltas.quantile(0.8) if not active_deltas.empty else 0

    days_from_start = (group["timestamp"] - course_start_date).dt.days
    activity_span_days = days_from_start.max() - days_from_start.min()

    clicks_near_peak = 0
    for peak in peak_dates:
        peak_start = peak - timedelta(days=2)
        clicks_near_peak += group[
            (group["timestamp"] >= peak_start) & (group["timestamp"] <= peak)
        ].shape[0]
    procrastination_ratio = clicks_near_peak / total_clicks if total_clicks > 0 else 0

    first_week_ratio = (
        group[group["timestamp"] <= (course_start_date + timedelta(days=7))].shape[0]
        / total_clicks
        if total_clicks > 0
        else 0
    )

    num_items_unicos = group["resource_id"].nunique()
    num_active_days = group["timestamp"].dt.date.nunique()
    regularity_ratio = (
        num_active_days / activity_span_days if activity_span_days > 0 else 0
    )

    hour = group["timestamp"].dt.hour
    night_owl_ratio = ((hour >= 0) & (hour < 5)).mean()

    avg_clicks_per_item = total_clicks / num_items_unicos if num_items_unicos > 0 else 0
    fixation_ratio = (
        (group["resource_id"].value_counts().max() / total_clicks)
        if total_clicks > 0
        else 0
    )

    daily_clicks = group.groupby(group["timestamp"].dt.date).size()
    std_dev_daily_clicks = daily_clicks.std() if len(daily_clicks) > 1 else 0
    weekend_ratio = (group["timestamp"].dt.weekday >= 5).mean()

    mid_date = course_start_date + timedelta(days=course_duration_days / 2)
    c1 = group[group["timestamp"] < mid_date].shape[0]
    c2 = group[group["timestamp"] >= mid_date].shape[0]
    mid_course_ratio = (c2 / c1) if c1 > 0 else (999 if c2 > 0 else 0)

    slope_actividad = 0
    if num_active_days > 1:
        X_reg = (
            (daily_clicks.index - course_start_date.date())
            .map(lambda x: x.days)
            .values.reshape(-1, 1)
        )
        y_reg = daily_clicks.values
        try:
            slope_actividad = LinearRegression().fit(X_reg, y_reg).coef_[0]
        except:
            pass

    session_starts = group[(delta_t > SESSION_THRESHOLD) | delta_t.isnull()][
        "timestamp"
    ]
    std_dev_hour_session_start = (
        (session_starts.dt.hour + session_starts.dt.minute / 60).std()
        if len(session_starts) > 1
        else 0
    )

    ratio_short_breaks = (
        ((delta_t > PAUSA_ESTUDIO_MIN) & (delta_t <= PAUSA_ESTUDIO_MAX)).sum()
        / total_clicks
        if total_clicks > 0
        else 0
    )

    max_consecutive = 0
    avg_inactive = 0
    if num_active_days > 1:
        dates = sorted(group["timestamp"].dt.date.unique())
        diffs = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        curr, m_streak = 1, 1
        for d in diffs:
            if d == 1:
                curr += 1
            else:
                m_streak = max(m_streak, curr)
                curr = 1
        max_consecutive = max(m_streak, curr)
        breaks = [d for d in diffs if d > 1]
        avg_inactive = np.mean(breaks) if breaks else 0

    coverage_ratio = (
        num_items_unicos / total_unique_items_in_course
        if total_unique_items_in_course > 0
        else 0
    )

    items = group["resource_id"].astype(str).values
    deltas_arr = delta_t.values
    fwd, bwd, rev, sess_moves = 0, 0, 0, 0

    for i in range(1, len(items)):
        if deltas_arr[i] <= SESSION_THRESHOLD:
            sess_moves += 1
            src, dst = int(items[i - 1]), int(items[i])
            if dst in prereq_map.get(src, []):
                fwd += 1
            elif src in prereq_map.get(dst, []):
                bwd += 1
            elif dst in revisit_map.get(src, []):
                rev += 1

    progress_ratio = fwd / sess_moves if sess_moves > 0 else 0
    backward_prereq_ratio = bwd / sess_moves if sess_moves > 0 else 0
    revisit_ratio = rev / sess_moves if sess_moves > 0 else 0

    return pd.Series(
        {
            "total_clicks": total_clicks,
            "total_sessions": total_sessions,
            "avg_delta_t_sec": avg_delta_t,
            "zapping_ratio": zapping_ratio,
            "activity_span_days": activity_span_days,
            "first_week_ratio": first_week_ratio,
            "procrastination_ratio": procrastination_ratio,
            "num_unique_items": num_items_unicos,
            "num_active_days": num_active_days,
            "regularity_ratio": regularity_ratio,
            "night_owl_ratio": night_owl_ratio,
            "avg_clicks_per_item": avg_clicks_per_item,
            "fixation_ratio": fixation_ratio,
            "std_dev_daily_clicks": std_dev_daily_clicks,
            "weekend_ratio": weekend_ratio,
            "mid_course_ratio": mid_course_ratio,
            "activity_slope": slope_actividad,
            "std_dev_hour_session_start": std_dev_hour_session_start,
            "ratio_short_breaks": ratio_short_breaks,
            "percentile_80_delta_t": percentil_80_delta_t,
            "max_consecutive_days": max_consecutive,
            "avg_inactive_days": avg_inactive,
            "coverage_ratio": coverage_ratio,
            "progress_ratio": progress_ratio,
            "backward_prereq_ratio": backward_prereq_ratio,
            "revisit_ratio": revisit_ratio,
        }
    )


df_features = (
    df_clicks.groupby("student_id").apply(calculate_features_per_student).fillna(0)
)
df_features = df_features.merge(
    df_outcomes.set_index("student_id"), left_index=True, right_index=True
)

# --- 4. Clustering ---
print("\n--- 4. Finding Archetypes with K-Means ---")
features_for_clustering = [
    "total_clicks",
    "total_sessions",
    "avg_delta_t_sec",
    "zapping_ratio",
    "activity_span_days",
    "first_week_ratio",
    "procrastination_ratio",
    "num_unique_items",
    "num_active_days",
    "regularity_ratio",
    "night_owl_ratio",
    "avg_clicks_per_item",
    "fixation_ratio",
    "std_dev_daily_clicks",
    "weekend_ratio",
    "mid_course_ratio",
    "activity_slope",
    "std_dev_hour_session_start",
    "ratio_short_breaks",
    "percentile_80_delta_t",
    "max_consecutive_days",
    "avg_inactive_days",
    "coverage_ratio",
    "progress_ratio",
    "backward_prereq_ratio",
    "revisit_ratio",
]

X = df_features[features_for_clustering]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

K_OPTIMO = 3
kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10).fit(X_scaled)

df_features["archetype"] = kmeans.labels_
archetypes_names = {0: "Dropouts", 1: "Consistents", 2: "Erratics"}
df_features["archetype_name"] = df_features["archetype"].map(archetypes_names)


# --- 5. VISUALIZACIÓN MEJORADA (CON NOMBRES LIMPIOS) ---
print("\n--- 5. Analyzing the Archetypes (Improved Visualization) ---")

# --- A) DICCIONARIO DE TRADUCCIÓN (Para que se vea bonito) ---
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

# --- B) Definición de Categorías ---
metric_categories_raw = {
    "Volume & Pacing": [
        "total_clicks",
        "total_sessions",
        "avg_delta_t_sec",
        "avg_clicks_per_item",
        "std_dev_daily_clicks",
        "percentile_80_delta_t",
    ],
    "Consistency & Timing": [
        "activity_span_days",
        "num_active_days",
        "regularity_ratio",
        "max_consecutive_days",
        "avg_inactive_days",
        "std_dev_hour_session_start",
        "ratio_short_breaks",
    ],
    "Habitual & Phasic": [
        "first_week_ratio",
        "procrastination_ratio",
        "weekend_ratio",
        "night_owl_ratio",
        "mid_course_ratio",
        "activity_slope",
    ],
    "Content Interaction": [
        "num_unique_items",
        "zapping_ratio",
        "fixation_ratio",
        "coverage_ratio",
        "progress_ratio",
        "backward_prereq_ratio",
        "revisit_ratio",
    ],
}

# --- C) Preparación de Datos ---
df_plot = pd.DataFrame(X_scaled, columns=features_for_clustering)
df_plot["archetype_name"] = df_features["archetype_name"].values

# 'Melt' para formato largo
df_melted = df_plot.melt(
    id_vars="archetype_name", var_name="Metric_Raw", value_name="Z-Score"
)

# Aplicar nombres legibles
df_melted["Metric"] = df_melted["Metric_Raw"].map(readable_metrics)

# Ordenar features por categoría usando los nombres bonitos
ordered_features_readable = []
metric_counts_per_cat = {}

for cat, raws in metric_categories_raw.items():
    valid_names = [readable_metrics[r] for r in raws if r in features_for_clustering]
    ordered_features_readable.extend(valid_names)
    metric_counts_per_cat[cat] = len(valid_names)

# --- D) Configuración de la Figura ---
fig = plt.figure(figsize=(20, 14))  # Un poco más alta para que quepan los nombres
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# --- E) Panel Izquierdo: Dot Plot ---
sns.pointplot(
    data=df_melted,
    x="Z-Score",
    y="Metric",
    hue="archetype_name",
    join=False,
    dodge=0.6,
    order=ordered_features_readable,  # Usar el orden "bonito"
    palette="deep",
    errorbar=("ci", 95),
    capsize=0.3,
    markers=["o", "s", "D"],
    ax=ax1,
)

ax1.set_title("Behavioral Profile by Archetype (Standardized Mean)", fontsize=18)
ax1.set_xlabel("Z-Score (Deviation from Mean)", fontsize=18)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=18)
ax1.set_ylabel("")  # Quitamos etiqueta "Metric" porque es redundante
ax1.tick_params(axis="y", labelsize=18)  # AUMENTAR TAMAÑO DE FUENTE
ax1.grid(True, axis="x", linestyle="--")
ax1.axvline(0, color="black", linewidth=1.5, alpha=0.5)
ax1.set_xlim(-3, 3)
ax1.legend(
    title="Archetype", loc="center left", fontsize=18, title_fontsize=18
)

# --- F) Panel Derecho: Stacked Bar Chart ---
outcome_by_archetype = (
    df_features.groupby("archetype_name")["outcome"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)
cols_order = [c for c in ["Fail", "Pass"] if c in outcome_by_archetype.columns]
outcome_by_archetype = outcome_by_archetype[cols_order]

colors_outcome = {"Fail": "#d62728", "Pass": "#2ca02c"}
actual_colors = [colors_outcome.get(c, "gray") for c in outcome_by_archetype.columns]

outcome_by_archetype.plot(
    kind="bar", stacked=True, ax=ax2, color=actual_colors, edgecolor="black", width=0.7
)

for c in ax2.containers:
    labels = [f"{v.get_height():.1%}" if v.get_height() > 0.01 else "" for v in c]
    ax2.bar_label(
        c,
        labels=labels,
        label_type="center",
        color="white",
        fontweight="bold",
        fontsize=18,
        rotation=90
    )

ax2.set_title("Academic Outcome", fontsize=18)
ax2.set_xlabel("Archetype", fontsize=18)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=18)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=18)
ax2.legend(
    title="Outcome", loc="upper center", fontsize=18, title_fontsize=18
)

plt.tight_layout()
plt.savefig("figs/analisis_arquetipos.eps", format="eps")
print("Figure saved with READABLE metric names.")