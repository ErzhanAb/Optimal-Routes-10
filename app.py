import json
import warnings
from datetime import datetime, timedelta

import folium
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor


warnings.filterwarnings('ignore')
print("✓ Библиотеки импортированы.")

print("--- Загрузка артефактов... ---")
canonical_routes = {}
try:
    catboost_model = CatBoostRegressor()
    catboost_model.load_model('catboost_eta_predictor.cbm')

    xgb_artifacts = joblib.load(
        'xgboost_traffic_model_tuned_no_weather.joblib'
    )
    xgboost_model = xgb_artifacts['model']
    le_geom_color = xgb_artifacts['label_encoder']

    with open('canonical_routes.json', 'r', encoding='utf-8') as f:
        canonical_routes = json.load(f)

    print("✓ Все модели и данные успешно загружены.")
    print(f"✓ Найдено маршрутов: {len(canonical_routes)}")

except Exception as e:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА при загрузке артефактов: {e}")

XGB_TRAINING_COLUMNS = [
    "start_lon", "start_lat", "end_lon", "end_lat", "tod_sin", "tod_cos",
    "dow_sin", "dow_cos", "route_name_12_mkr-TSUM",
    "route_name_Ak_Orgo-Hyatt_Regency", "route_name_Ala_Too_Sq-Asanbai",
    "route_name_Alamedin1-Orion", "route_name_Cosmopark-Vostok_5",
    "route_name_Dordoi-Togolok_Moldo", "route_name_Dzhal-Osh_Bazar",
    "route_name_PVT-Globus", "route_name_Tunguch-Philharmonia",
    "route_name_Yuzhnye_Vorota-Dordoi", "variant_rank_0", "variant_rank_1"
]
CATBOOST_TRAINING_COLUMNS = [
    "route_name", "variant_rank", "total_distance_m", "maneuvers_count",
    "turns_left_count", "turns_right_count", "color_share_fast",
    "color_share_normal", "color_share_slow", "color_share_very_slow",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "turn_coefficient"
]
print("✓ Списки колонок для моделей (без погоды) воссозданы.")


def prepare_time_features(calc_time):
    minutes_of_day = calc_time.hour * 60 + calc_time.minute
    day_of_week = calc_time.weekday()
    return {
        "tod_sin": np.sin(2 * np.pi * minutes_of_day / 1440),
        "tod_cos": np.cos(2 * np.pi * minutes_of_day / 1440),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7)
    }


def predict_with_custom_thresholds(probabilities, thresholds, priority, le):
    class_map = {name: i for i, name in enumerate(le.classes_)}
    for class_name in priority:
        if class_name in thresholds and class_name in class_map:
            class_idx = class_map[class_name]
            if probabilities[class_idx] >= thresholds[class_name]:
                return class_idx
    return np.argmax(probabilities)


def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0)**2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def get_prediction_pipeline(route_name, target_datetime, mode,
                            progress=gr.Progress()):
    progress(0, desc="Начало расчета...")

    if mode == 'Прибыть к этому времени':
        calculation_time = target_datetime - timedelta(minutes=45)
    else:
        calculation_time = target_datetime

    progress(0.1, desc="Подготовка временных признаков...")
    time_features = prepare_time_features(calculation_time)

    route_canons = canonical_routes[route_name]
    final_results = {}

    for i, variant_rank in enumerate(['0', '1']):
        variant_name = "А" if variant_rank == '0' else "Б"
        progress(0.2 + i * 0.4, desc=f"Расчет для канона {variant_name}...")

        canon_data = route_canons[variant_rank]
        canon_info = canon_data['canon_medoid_info']
        segmented_path = canon_data['segmented_path']

        segment_features_list = []
        for segment in segmented_path:
            features = {
                'route_name': route_name,
                'variant_rank': variant_rank,
                **segment,
                **time_features
            }
            segment_features_list.append(features)

        x_xgb = pd.DataFrame(segment_features_list)
        x_xgb_encoded = pd.get_dummies(
            x_xgb, columns=['route_name', 'variant_rank']
        )
        x_xgb_final = x_xgb_encoded.reindex(
            columns=XGB_TRAINING_COLUMNS, fill_value=0
        )

        y_xgb_proba = xgboost_model.predict_proba(x_xgb_final)

        custom_thresholds = {
            'slow-jams': 0.7, 'slow': 0.5, 'fast': 0.4, 'normal': 0.35
        }
        class_priority = ['slow-jams', 'slow', 'fast', 'normal']

        predicted_indices = [
            predict_with_custom_thresholds(
                p, custom_thresholds, class_priority, le_geom_color
            ) for p in y_xgb_proba
        ]
        predicted_colors = le_geom_color.inverse_transform(predicted_indices)

        progress(0.4 + i*0.4, desc=f"Агрегация данных для канона {variant_name}...")
        color_counts = pd.Series(predicted_colors).value_counts()
        total_segments = len(predicted_colors)
        color_shares = {
            'color_share_fast': color_counts.get('fast', 0) / total_segments,
            'color_share_normal': color_counts.get('normal', 0) / total_segments,
            'color_share_slow': color_counts.get('slow', 0) / total_segments,
            'color_share_very_slow': color_counts.get('slow-jams', 0) / total_segments,
        }
        h_dist = haversine_distance(
            canon_info['start_lat'], canon_info['start_lon'],
            canon_info['end_lat'], canon_info['end_lon']
        )
        turn_coefficient = canon_info['total_distance_m'] / (h_dist + 1e-6)

        features_catboost = {
            **canon_info,
            'route_name': route_name,
            'variant_rank': int(variant_rank),
            'turn_coefficient': turn_coefficient,
            **color_shares,
            **time_features
        }
        x_catboost = pd.DataFrame([features_catboost])
        x_catboost_final = x_catboost.reindex(
            columns=CATBOOST_TRAINING_COLUMNS, fill_value=0
        )

        predicted_duration_sec = catboost_model.predict(x_catboost_final)[0]

        final_results[variant_name] = {
            'duration_sec': predicted_duration_sec,
            'distance_m': canon_info['total_distance_m'],
            'segmented_path_with_colors': list(zip(segmented_path,
                                                   predicted_colors))
        }

    progress(1.0, desc="Формирование карты...")
    return final_results


def visualize_on_map(results):
    if not results or len(results) < 2:
        return None

    optimal_key = 'А' if results['А']['duration_sec'] <= results['Б']['duration_sec'] else 'Б'
    alternative_key = 'Б' if optimal_key == 'А' else 'А'
    color_map = {
        'fast': '#33CC33',
        'normal': '#FFC300',
        'slow': '#FF3333',
        'slow-jams': 'darkred'
    }
    m = folium.Map()

    name_optimal = (
        f"Оптимальный: Канон {optimal_key} "
        f"(~{results[optimal_key]['duration_sec']/60:.1f} мин)"
    )
    group_optimal = folium.FeatureGroup(name=name_optimal, show=True)

    name_alternative = (
        f"Альтернативный: Канон {alternative_key} "
        f"(~{results[alternative_key]['duration_sec']/60:.1f} мин)"
    )
    group_alternative = folium.FeatureGroup(name=name_alternative, show=True)

    main_line_weight = 6
    border_line_weight = 9
    opt_path_coords = []
    path_data_opt = results[optimal_key]['segmented_path_with_colors']
    if path_data_opt:
        start_coords = (path_data_opt[0][0]['start_lat'],
                        path_data_opt[0][0]['start_lon'])
        opt_path_coords.append(start_coords)
        for segment, _ in path_data_opt:
            end_coords = (segment['end_lat'], segment['end_lon'])
            opt_path_coords.append(end_coords)

    alt_path_coords = []
    path_data_alt = results[alternative_key]['segmented_path_with_colors']
    if path_data_alt:
        start_coords = (path_data_alt[0][0]['start_lat'],
                        path_data_alt[0][0]['start_lon'])
        alt_path_coords.append(start_coords)
        for segment, _ in path_data_alt:
            end_coords = (segment['end_lat'], segment['end_lon'])
            alt_path_coords.append(end_coords)

    folium.PolyLine(
        locations=alt_path_coords, color='gray', weight=5,
        opacity=0.7, dash_array='5, 5'
    ).add_to(group_optimal)
    folium.PolyLine(
        locations=opt_path_coords, color='white',
        weight=border_line_weight, opacity=1.0
    ).add_to(group_optimal)

    all_points_for_bounds = []
    for segment, color in results[optimal_key]['segmented_path_with_colors']:
        points = [(segment['start_lat'], segment['start_lon']),
                  (segment['end_lat'], segment['end_lon'])]
        all_points_for_bounds.extend(points)
        folium.PolyLine(
            locations=points, color=color_map.get(color, 'gray'),
            weight=main_line_weight, opacity=1.0
        ).add_to(group_optimal)

    folium.PolyLine(
        locations=opt_path_coords, color='gray', weight=5,
        opacity=0.7, dash_array='5, 5'
    ).add_to(group_alternative)
    folium.PolyLine(
        locations=alt_path_coords, color='white',
        weight=border_line_weight, opacity=1.0
    ).add_to(group_alternative)

    for segment, color in results[alternative_key]['segmented_path_with_colors']:
        points = [(segment['start_lat'], segment['start_lon']),
                  (segment['end_lat'], segment['end_lon'])]
        folium.PolyLine(
            locations=points, color=color_map.get(color, 'gray'),
            weight=main_line_weight, opacity=1.0
        ).add_to(group_alternative)

    group_optimal.add_to(m)
    group_alternative.add_to(m)

    if all_points_for_bounds:
        lats = [p[0] for p in all_points_for_bounds]
        lons = [p[1] for p in all_points_for_bounds]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        m.fit_bounds(bounds, padding=(20, 20))

    start_point = results[optimal_key]['segmented_path_with_colors'][0][0]
    end_point = results[optimal_key]['segmented_path_with_colors'][-1][0]
    folium.Marker(
        location=[start_point['start_lat'], start_point['start_lon']],
        icon=folium.Icon(color='blue', icon='play', prefix='fa')
    ).add_to(m)
    folium.Marker(
        location=[end_point['end_lat'], end_point['end_lon']],
        icon=folium.Icon(color='red', icon='circle', prefix='fa')
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m._repr_html_()


def main_interface(route_name, weekday_name, hour_input, minute_input, mode):
    weekday_map_ru = {
        "Понедельник": 0, "Вторник": 1, "Среда": 2, "Четверг": 3,
        "Пятница": 4, "Суббота": 5, "Воскресенье": 6
    }
    try:
        if not route_name:
            raise gr.Error("Пожалуйста, выберите маршрут.")

        today = datetime.now()
        today_weekday = today.weekday()
        target_weekday = weekday_map_ru[weekday_name]
        days_ahead = target_weekday - today_weekday
        if days_ahead < 0:
            days_ahead += 7
        target_date = today + timedelta(days=days_ahead)
        target_datetime = datetime(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=int(hour_input),
            minute=int(minute_input)
        )
    except (KeyError, ValueError, TypeError) as e:
        raise gr.Error(f"Некорректный день недели или время. Ошибка: {e}")

    results = get_prediction_pipeline(route_name, target_datetime, mode)
    if not results:
        return "Не удалось получить результаты. Проверьте логи.", None

    optimal_key = 'А' if results['А']['duration_sec'] <= results['Б']['duration_sec'] else 'Б'
    alternative_key = 'Б' if optimal_key == 'А' else 'А'
    text_output = "### Итоговые предсказания:\n"

    def format_route_info(key, data, is_optimal):
        title = "Оптимальный" if is_optimal else "Альтернативный"
        duration_sec = data['duration_sec']
        distance_km = data['distance_m'] / 1000

        line = (
            f"- **{title} маршрут (Канон {key}):** "
            f"{duration_sec:.0f} сек. (~{duration_sec/60:.1f} мин.) / "
            f"**{distance_km:.1f} км**"
        )

        if mode == 'Прибыть к этому времени':
            departure_time = target_datetime - timedelta(seconds=int(duration_sec))
            line += (
                f"\n  - *Рекомендуем выехать не позже: "
                f"**{departure_time.strftime('%H:%M')}***"
            )
        return line

    text_output += format_route_info(
        optimal_key, results[optimal_key], True
    ) + "\n"
    text_output += format_route_info(
        alternative_key, results[alternative_key], False
    )
    map_html = visualize_on_map(results)
    return text_output, map_html


theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# 🚦 Предсказание времени поездки и трафика")
    gr.Markdown("Выберите маршрут, день недели, время и режим, чтобы получить прогноз.")

    available_routes = sorted(list(canonical_routes.keys()))

    with gr.Row():
        with gr.Column(scale=1):
            route_input = gr.Dropdown(
                choices=available_routes,
                label="Маршрут",
                value=available_routes[0] if available_routes else None,
                interactive=True
            )
            now = datetime.now()

            weekdays_ru = [
                "Понедельник", "Вторник", "Среда", "Четверг",
                "Пятница", "Суббота", "Воскресенье"
            ]
            weekday_input = gr.Dropdown(
                label="День недели",
                choices=weekdays_ru,
                value=weekdays_ru[now.weekday()]
            )

            with gr.Row():
                hour_input = gr.Dropdown(
                    label="Час",
                    choices=[f"{h:02d}" for h in range(24)],
                    value=f"{now.hour:02d}"
                )
                minute_input = gr.Dropdown(
                    label="Минута",
                    choices=[f"{m:02d}" for m in range(60)],
                    value=f"{now.minute:02d}"
                )

            mode_input = gr.Radio(
                choices=['Выехать в это время', 'Прибыть к этому времени'],
                label="Режим",
                value='Выехать в это время'
            )
            submit_btn = gr.Button("Рассчитать", variant="primary")
        with gr.Column(scale=2):
            result_text = gr.Markdown(value="### Результаты будут здесь...")
            map_output = gr.HTML(label="Карта маршрута")

    submit_btn.click(
        fn=main_interface,
        inputs=[route_input, weekday_input, hour_input, minute_input, mode_input],
        outputs=[result_text, map_output]
    )

if __name__ == "__main__":
    demo.launch()
