import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

matplotlib.use("Agg")  # Para guardar figuras sin mostrar
from sklearn.linear_model import LinearRegression
import json
from collections import Counter
import os
import joblib

# Definiciones de Ritmo (Delta T en segundos)
ZAPPING_THRESHOLD = 60  # 1 minuto
SESSION_THRESHOLD = 1800  # 30 minutos
PAUSA_ESTUDIO_MIN = 60  # 1 minuto
PAUSA_ESTUDIO_MAX = 900  # 15 minutos (más que esto, es una pausa larga)

# --- 1. Carga y Preparación de tus Datos Reales ---
print("--- 1. Loading your real data ---")

# <-- ¡CAMBIO IMPORTANTE!
# Especifica los nombres correctos de tus archivos
archivo_de_logs = "database/logs.csv"  # Tu archivo con: user, item, time
archivo_de_notas = (
    "database/grades.csv"  # Tu archivo con: user, resultado (Aprueba/Suspende)
)

try:
    # Cargar los datos de clics
    df_clicks_raw = pd.read_csv(archivo_de_logs)

    # Cargar los datos de resultados
    df_outcomes_raw = pd.read_csv(archivo_de_notas)

except FileNotFoundError:
    print(f"Error: Files '{archivo_de_logs}' or '{archivo_de_notas}' not found.")
    print(
        "Make sure the files are in the same folder as the script or provide the full path."
    )
    # Si da error, detenemos el script
    exit()

# --- Adaptación de Nombres de Columnas ---
# Renombramos tus columnas a las que el script espera
print("Adapting column names...")
df_clicks = df_clicks_raw.rename(
    columns={"user": "student_id", "item": "resource_id", "time": "timestamp"}
)

# <-- ¡CAMBIO IMPORTANTE!
# Ajusta 'columna_resultado' al nombre real de tu columna de notas (ej. 'nota', 'final', 'outcome')
df_outcomes = df_outcomes_raw.rename(
    columns={
        "user": "student_id",
        "grade": "outcome",  # CAMBIA 'columna_resultado'
    }
)

# --- Conversión de Tipo de Datos ---
# Es CRÍTICO que la columna 'timestamp' sea un objeto datetime
print("Converting 'timestamp' to datetime format...")
try:
    df_clicks["timestamp"] = pd.to_datetime(df_clicks["timestamp"])
except Exception as e:
    print(
        f"Error converting 'timestamp' column. Make sure the format is correct (e.g. '2023-09-05 15:27:38')."
    )
    print(f"Detailed error: {e}")
    exit()

# Asegurarnos de que tenemos los datos de outcome para los usuarios
# Esto es un chequeo de sanidad
usuarios_logs = set(df_clicks["student_id"].unique())
usuarios_notas = set(df_outcomes["student_id"].unique())
usuarios_comunes = usuarios_logs.intersection(usuarios_notas)

print(f"Found {len(usuarios_logs)} students in logs.")
print(f"Found {len(usuarios_notas)} students in grades.")
print(f"Analyzing {len(usuarios_comunes)} students present in both files.")

# Filtramos para quedarnos solo con los estudiantes que tienen ambos datos
df_clicks = df_clicks[df_clicks["student_id"].isin(usuarios_comunes)]
df_outcomes = df_outcomes[df_outcomes["student_id"].isin(usuarios_comunes)]

print("\n--- Sample of Loaded and Adapted Clicks ---")
print(df_clicks.head())
print("\n--- Sample of Loaded and Adapted Grades ---")
print(df_outcomes.head())

try:
    with open("database/forward_graph.json", "r") as f:
        prereq_map = json.load(f)
        prereq_map = {int(k): v for k, v in prereq_map.items()}
    print("Prerequisite map loaded.")
except FileNotFoundError:
    print("ERROR: 'database/forward_graph.json' not found.")
    prereq_map = {}

try:
    with open("database/remedial_graph.json", "r") as f:
        revisit_map = json.load(f)
        revisit_map = {int(k): v for k, v in revisit_map.items()}
    print("Revisit map loaded.")
except FileNotFoundError:
    print("ERROR: 'database/remedial_graph.json' not found.")
    revisit_map = {}


# --- Cálculo de Métricas Globales del Curso ---
print("Calculating global course metrics...")
total_items_set = set()


# Iteramos sobre los mapas para encontrar todos los items únicos
def add_items_to_set(item_map, item_set):
    for key, values in item_map.items():
        item_set.add(int(key))
        for val in values:
            item_set.add(int(val))


add_items_to_set(prereq_map, total_items_set)
add_items_to_set(revisit_map, total_items_set)

total_unique_items_in_course = len(total_items_set)

if total_unique_items_in_course == 0:
    print("WARNING! No items loaded from the maps. 'coverage_ratio' will be 0.")
else:
    print(f"Found {total_unique_items_in_course} unique items in total in the course.")


# --- 2. Finding the Course "Pulse" ---
print("\n--- 2. Finding the Course Pulse ---")

# Group all clicks by day
all_clicks_by_day = df_clicks.set_index("timestamp").resample("D").size()
all_clicks_by_day.rename("total_clicks", inplace=True)

# Find the 3 days with the most activity
# You can change nlargest(3) to 5 if your course is longer
peak_dates = all_clicks_by_day.nlargest(2).index
print(f"Peak activity dates inferred: \n{peak_dates.date}")

# Visualizar el pulso
plt.figure(figsize=(15, 6))
all_clicks_by_day.plot(kind="line", label="Daily Total Activity")
plt.title("Course Pulse (Aggregated Activity)")
plt.ylabel("Total Clicks")
plt.xlabel("Date")

# Mark the found peaks
for peak in peak_dates:
    plt.axvline(peak, color="red", linestyle="--", lw=2, label=f"Peak: {peak.date()}")

plt.legend()
plt.grid(True, linestyle=":")
plt.savefig("figs/pulso_curso.eps", format="eps")


# --- 3. Feature Engineering (Frequency, Rhythm, and Timing) ---
print("\n--- 3. Creating Behavioral Features ---")

# Course start and end dates (inferred from your data)
course_start_date = df_clicks["timestamp"].min()
course_end_date = df_clicks["timestamp"].max()
course_duration_days = (course_end_date - course_start_date).days
print(
    f"Course duration: {course_duration_days} days (from {course_start_date.date()} to {course_end_date.date()})"
)


def calculate_features_per_student(group):
    # Ordenar clics para calcular deltas
    group = group.sort_values("timestamp")

    # --- Feature 1: Frecuencia ---
    total_clicks = len(group)

    # --- Feature 2: Ritmo (Delta T) ---
    delta_t = group["timestamp"].diff().dt.total_seconds()
    total_sessions = (delta_t > SESSION_THRESHOLD).sum() + 1
    active_deltas = delta_t[delta_t <= SESSION_THRESHOLD]
    avg_delta_t = active_deltas.mean()
    zapping_ratio = (active_deltas <= ZAPPING_THRESHOLD).mean()

    # --- Feature 3: Timing ---
    first_click_day = (group["timestamp"].min() - course_start_date).days
    last_click_day = (group["timestamp"].max() - course_start_date).days
    activity_span_days = last_click_day - first_click_day

    # Procrastinación: % de clics cerca de los picos
    clicks_near_peak = 0
    peak_window_days = 2  # Consideramos "cerca" 2 días antes del pico
    for peak in peak_dates:
        peak_start = peak - timedelta(days=peak_window_days)
        clicks_near_peak += group[
            (group["timestamp"] >= peak_start) & (group["timestamp"] <= peak)
        ].shape[0]
    procrastination_ratio = clicks_near_peak / total_clicks if total_clicks > 0 else 0

    # % de clics en la primera semana
    first_week_end = course_start_date + timedelta(days=7)
    first_week_clicks = group[group["timestamp"] <= first_week_end].shape[0]
    first_week_ratio = first_week_clicks / total_clicks if total_clicks > 0 else 0

    # Diversidad
    num_items_unicos = group["resource_id"].nunique()
    group_dates_only = group["timestamp"].dt.date

    # Regularidad
    num_active_days = group["timestamp"].dt.date.nunique()
    if activity_span_days > 0:
        regularity_ratio = num_active_days / activity_span_days
    else:
        # Si solo ha entrado un día, su span es 0.
        regularity_ratio = 0

    # Micro-Timing
    hour_of_click = group["timestamp"].dt.hour
    night_owl_clicks = (hour_of_click >= 0) & (hour_of_click < 5)
    night_owl_ratio = night_owl_clicks.mean()

    # Profundidad
    if num_items_unicos > 0:
        promedio_clics_por_item = total_clicks / num_items_unicos
    else:
        promedio_clics_por_item = 0

    # Ratio de "fijación" en el recurso más usado
    if total_clicks > 0:
        clicks_item_mas_frecuente = group["resource_id"].value_counts().max()
        ratio_de_fijacion = clicks_item_mas_frecuente / total_clicks
    else:
        clicks_item_mas_frecuente = 0
        ratio_de_fijacion = 0

    # Consistencia
    clicks_per_active_day = group.groupby(group_dates_only).size()
    std_dev_clics_diarios = clicks_per_active_day.std()

    weekday = group["timestamp"].dt.weekday
    weekend_clicks = (weekday >= 5).sum()
    if total_clicks > 0:
        ratio_fin_de_semana = weekend_clicks / total_clicks
    else:
        ratio_fin_de_semana = 0

    # Mitad curso
    if course_duration_days > 0:
        mid_point_date = course_start_date + timedelta(days=course_duration_days / 2)

        clicks_primera_mitad = group[group["timestamp"] < mid_point_date].shape[0]
        clicks_segunda_mitad = group[group["timestamp"] >= mid_point_date].shape[0]

        if clicks_primera_mitad > 0:
            ratio_mitad_curso = clicks_segunda_mitad / clicks_primera_mitad
        elif clicks_segunda_mitad > 0:
            # Si solo trabajó en la 2a mitad, ratio muy alto (ej. 999)
            ratio_mitad_curso = 999
        else:
            ratio_mitad_curso = 0
    else:
        ratio_mitad_curso = 0

    # Slope
    slope_actividad = 0
    # Solo calculamos si hay más de un día activo
    if num_active_days > 1:
        # Convertimos fechas a "días desde el inicio" para la regresión
        dias_desde_inicio = (
            clicks_per_active_day.index - course_start_date.date()
        ).map(lambda d: d.days)

        # Preparamos X (días) e y (clics)
        X = dias_desde_inicio.to_numpy().reshape(-1, 1)
        y = clicks_per_active_day.to_numpy()

        try:
            model = LinearRegression()
            model.fit(X, y)
            slope_actividad = model.coef_[0]  # La pendiente
        except Exception:
            slope_actividad = 0  # Si falla la regresión

    # Inicio Sesión
    std_dev_hora_inicio_sesion = 0
    # Identificamos los inicios de sesión (primer clic O clic tras >30min)
    inicios_de_sesion = (delta_t > SESSION_THRESHOLD) | delta_t.isnull()
    timestamps_inicio_sesion = group[inicios_de_sesion]["timestamp"]

    if len(timestamps_inicio_sesion) > 1:
        # Extraemos la hora (con decimales, ej. 9.5 para 9:30)
        horas_inicio = timestamps_inicio_sesion.dt.hour + (
            timestamps_inicio_sesion.dt.minute / 60
        )
        std_dev_hora_inicio_sesion = horas_inicio.std()

    # Pausas de Estudio
    pausas_de_estudio = (delta_t > PAUSA_ESTUDIO_MIN) & (delta_t <= PAUSA_ESTUDIO_MAX)
    if total_clicks > 0:
        ratio_pausas_cortas = pausas_de_estudio.sum() / total_clicks
    else:
        ratio_pausas_cortas = 0

    # Percentil 80 del tiempo entre clics (para ignorar el zapping)
    if active_deltas.empty:
        percentil_80_delta_t = 0
    else:
        percentil_80_delta_t = active_deltas.quantile(0.8)

    # Hábito
    dias_consecutivos_max = 0
    promedio_dias_inactivos = 0

    if num_active_days > 1:
        unique_dates = group_dates_only.unique()
        unique_dates.sort()

        # Diferencias en días entre días activos
        deltas_dias = unique_dates[1:] - unique_dates[:-1]
        deltas_dias_num = [d.days for d in deltas_dias]

        # Calcular racha máxima
        current_streak = 1
        max_streak = 1
        for d in deltas_dias_num:
            if d == 1:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 1
        dias_consecutivos_max = max(
            max_streak, current_streak
        )  # Comprobar la racha final

        # Calcular descanso promedio
        breaks = [d for d in deltas_dias_num if d > 1]
        if breaks:
            promedio_dias_inactivos = np.mean(breaks)

    # coverage_ratio
    if total_unique_items_in_course > 0:
        coverage_ratio = num_items_unicos / total_unique_items_in_course
    else:
        coverage_ratio = 0

    # Ratios de Movimiento

    # Extraemos los arrays para iterar rápido
    items = group["resource_id"].astype(str).values  # Convertir a string
    deltas = delta_t.values

    forward_moves = 0
    backward_prereq_moves = 0
    revisit_moves = 0
    total_session_moves = 0

    # Iteramos desde el segundo clic
    for i in range(1, len(items)):
        # Solo contamos movimientos DENTRO de una sesión
        if deltas[i] <= SESSION_THRESHOLD:
            total_session_moves += 1

            item_A = int(items[i - 1])  # Clic anterior
            item_B = int(items[i])  # Clic actual

            # Comprobamos movimiento de PROGRESO (item_A es prerequisito de item_B)
            if item_B in prereq_map.get(item_A, []):
                forward_moves += 1

            # Comprobamos movimiento de RETROCESO CONFUSO (item_B es prerequisito de item_A)
            elif item_A in prereq_map.get(item_B, []):
                backward_prereq_moves += 1

            # Comprobamos movimiento de REVISITA (item_B está en la lista de revisita de item_A)
            elif item_B in revisit_map.get(item_A, []):
                revisit_moves += 1

    # Calcular ratios finales
    if total_session_moves > 0:
        progress_ratio = forward_moves / total_session_moves
        backward_prereq_ratio = backward_prereq_moves / total_session_moves
        revisit_ratio = revisit_moves / total_session_moves
    else:
        progress_ratio = 0
        backward_prereq_ratio = 0
        revisit_ratio = 0

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
            "avg_clicks_per_item": promedio_clics_por_item,
            "fixation_ratio": ratio_de_fijacion,
            "std_dev_daily_clicks": std_dev_clics_diarios,
            "weekend_ratio": ratio_fin_de_semana,
            "mid_course_ratio": ratio_mitad_curso,
            "activity_slope": slope_actividad,
            "std_dev_hour_session_start": std_dev_hora_inicio_sesion,
            "ratio_short_breaks": ratio_pausas_cortas,
            "percentile_80_delta_t": percentil_80_delta_t,
            "max_consecutive_days": dias_consecutivos_max,
            "avg_inactive_days": promedio_dias_inactivos,
            "coverage_ratio": coverage_ratio,
            "progress_ratio": progress_ratio,
            "backward_prereq_ratio": backward_prereq_ratio,
            "revisit_ratio": revisit_ratio,
        }
    )


# Aplicar la función a cada grupo de estudiante
df_features = df_clicks.groupby("student_id").apply(calculate_features_per_student)

# Limpiar NaNs (p.ej., avg_delta_t si un estudiante solo tuvo 1 clic)
df_features.fillna(0, inplace=True)

# Añadir los resultados (Aprueba/Suspende) al dataframe de features
# Esto ahora funciona porque cargamos 'df_outcomes' en el paso 1
df_features = df_features.merge(
    df_outcomes.set_index("student_id"), left_index=True, right_index=True
)

print("\n--- Sample of Features Created ---")
print(df_features.head())


# --- 4. Clustering (K-Means) para Encontrar Arquetipos ---
print("\n--- 4. Finding Archetypes with K-Means ---")

# Seleccionar las features que usaremos para el clustering
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

# --- PASO CRÍTICO: Escalar los datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Encontrar el número óptimo de clusters (k) con el "Método del Codo" ---
inertia = []
K_range = range(1, 9)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Finding Optimal k")
plt.grid(True, linestyle=":")
plt.savefig("figs/metodo_del_codo.png", format="png")

# --- Ejecutar el Clustering final ---
K_OPTIMO = 3
print(f"Optimal choosen k: {K_OPTIMO}")

kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Añadir la etiqueta del arquetipo a nuestro dataframe
df_features["archetype"] = kmeans.labels_
archetypes_names = {0: "The Dropouts", 1: "The Consistents", 2: "The Erratics"}
df_features["archetype_name"] = df_features["archetype"].map(archetypes_names)

print("\n--- Sample of Features with Assigned Archetype ---")
print(df_features.head())


# --- 5. Análisis de los Arquetipos ---
print("\n--- 5. Analyzing the Archetypes ---")

# 1. ¿Cuál es el perfil de cada arquetipo?
# Calculamos la media de las features para cada grupo
archetype_profile = df_features.groupby("archetype")[features_for_clustering].mean()

print("\n--- Profile of Each Archetype ---")
# .T transpone para que sea más fácil de leer
print(archetype_profile.T.round(4))

# 2. ¿Cómo se relaciona cada arquetipo con el resultado (Aprobado/Suspenso)?
# Usamos la columna 'outcome' que cargamos en el paso 1
outcome_by_archetype = (
    df_features.groupby("archetype")["outcome"].value_counts(normalize=True).unstack()
)
outcome_by_archetype = outcome_by_archetype.fillna(0)

print("\n--- Pass/Fail Rate by Archetype ---")
print(outcome_by_archetype)

# Visualizar los resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico 1: Perfiles (usando datos escalados para comparar)
# Creamos un DataFrame con los centros de los clusters
profile_scaled = pd.DataFrame(
    scaler.transform(archetype_profile), columns=features_for_clustering
)
profile_scaled["archetype"] = archetype_profile.index
profile_scaled.set_index("archetype").T.plot(kind="bar", ax=ax1)
ax1.set_title("Profile of Features by Archetype (Standardized Values)")
ax1.set_ylabel("Standardized Value (mean=0, std=1)")
ax1.grid(True, linestyle=":")
ax1.legend(title="Archetype", labels=[archetypes_names[i] for i in archetype_profile.index])

# Gráfico 2: Tasa de Aprobado/Suspenso
outcome_by_archetype.plot(kind="bar", stacked=True, ax=ax2, colormap="RdYlGn")
ax2.set_title("Outcome (Pass/Fail) by Archetype")
ax2.set_ylabel("% of Students")
ax2.yaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
ax2.grid(True, linestyle=":")
ax2.set_xticklabels(
    [archetypes_names[i] for i in outcome_by_archetype.index], rotation=0
)
ax2.set_xlabel("Archetype")

plt.tight_layout()
plt.savefig("figs/analisis_arquetipos.eps", format="eps")

print("\n--- 6. Analyzing Contrasts (Zoom-In) ---")

# Especifica el número del arquetipo que tiene resultados mixtos (Pass y Fail)
ARCHETYPE_A_ANALIZAR = 1

# Filtramos el DataFrame para quedarnos solo con ese arquetipo
df_zoom_in = df_features[df_features["archetype"] == ARCHETYPE_A_ANALIZAR].copy()

if df_zoom_in.empty:
    print(
        f"Error: The Archetype {ARCHETYPE_A_ANALIZAR} does not exist or has no students."
    )
elif (
    "Pass" not in df_zoom_in["outcome"].values
    or "Fail" not in df_zoom_in["outcome"].values
):
    print(
        f"Error: The Archetype {ARCHETYPE_A_ANALIZAR} is not mixed. All pass or all fail."
    )
else:
    print(f"Zooming In on Archetype {ARCHETYPE_A_ANALIZAR}...")

    # Calculamos la media de todas las métricas, agrupando por 'outcome'
    df_comparison = df_zoom_in.groupby("outcome")[features_for_clustering].mean()

    print("\n--- Comparative Analysis (Pass vs Fail) within the Archetype ---")
    print(df_comparison.T.round(4))

    # --- Calcular la Diferencia Porcentual (El dato clave) ---
    profile_pass = df_comparison.loc["Pass"]
    profile_fail = df_comparison.loc["Fail"]

    # (Pass - Fail) / Fail * 100
    # Nos dice cuánto % "mejor" (o peor) es el perfil de Aprobado
    pct_diff = ((profile_pass - profile_fail) / profile_fail.abs()) * 100
    pct_diff = pct_diff.sort_values(ascending=False)

    print("\n--- Percentual Difference (Pass vs Fail) ---")
    print(pct_diff.round(1))

    # --- Visualización de las diferencias clave ---
    plt.figure(figsize=(15, 8))
    # Tomamos las 10 métricas con más diferencia (positiva y negativa)
    top_diffs = pd.concat([pct_diff.head(5), pct_diff.tail(5)])

    colors = ["g" if x > 0 else "r" for x in top_diffs.values]

    top_diffs.plot(kind="barh", color=colors)
    plt.title(f"Key Differences (Pass vs Fail) in Archetype {ARCHETYPE_A_ANALIZAR}")
    plt.xlabel("Difference % (Positive = Pass has more)")
    plt.ylabel("Metric")
    plt.grid(True, linestyle=":", axis="x")
    plt.axvline(0, color="black", lw=1)
    plt.tight_layout()
    plt.savefig("figs/diferencias_clave_arquetipo_1.eps", format="eps")

# --- 7. Guardar Artefactos para el Recomendador ---
print("\n--- 7. Saving artifacts for recommender ---")

# Crear el directorio 'models' si no existe
os.makedirs("models", exist_ok=True)

try:
    # 1. Guardar el Scaler
    joblib.dump(scaler, "models/scaler.joblib")
    print("Scaler saved to models/scaler.joblib")

    # 2. Guardar el Modelo K-Means
    joblib.dump(kmeans, "models/kmeans.joblib")
    print("K-Means model saved to models/kmeans.joblib")

    # 3. --- MODIFICADO ---
    # Guardar SOLO el perfil "ideal" (Pass)
    if "df_comparison" in locals() and "Pass" in df_comparison.index:
        # Seleccionamos solo la fila 'Pass'
        ideal_profile = df_comparison.loc["Pass"]

        # La guardamos en su propio archivo
        ideal_profile.to_json("models/ideal_profile.json", indent=4)
        print("Ideal profile (Pass) saved to models/ideal_profile.json")
    else:
        print(
            "WARNING: 'df_comparison' or 'Pass' profile not found. 'ideal_profile.json' not saved."
        )

    # Guardar las diferencias % (sigue siendo útil para análisis)
    if "pct_diff" in locals():
        pct_diff.to_json("models/pct_diff.json", indent=4)
        print("Percentage differences saved to models/pct_diff.json")
    else:
        print("WARNING: 'pct_diff' not found.")

    # 4. Guardar metadatos del curso
    course_metadata = {
        "course_start_date": course_start_date.isoformat(),
        "course_end_date": course_end_date.isoformat(),
        "peak_dates": [d.isoformat() for d in peak_dates],
        "total_unique_items_in_course": total_unique_items_in_course,
        "features_for_clustering": features_for_clustering,
        "ARCHETYPE_A_ANALIZAR": (
            ARCHETYPE_A_ANALIZAR if "ARCHETYPE_A_ANALIZAR" in locals() else -1
        ),
    }
    with open("models/course_metadata.json", "w") as f:
        json.dump(course_metadata, f, indent=4)
    print("Course metadata saved to models/course_metadata.json")

    print("\n--- Artifact saving complete. ---")

except Exception as e:
    print(f"\n--- Error saving artifacts ---")
    print(f"Details: {e}")
