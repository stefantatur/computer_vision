import numpy as np
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column
from bokeh.models import Div

# Загрузка истинных поз из poses.txt
poses_file = 'C:/Users/steph/PycharmProjects/SIFT_trajectory/dataset2/poses.txt'
gt_path = []

with open(poses_file, 'r') as f:
    for line in f:
        values = list(map(float, line.split()))
        t = values[9:12]  # Берём последние 3 координаты (X, Y, Z)
        gt_path.append(t)

gt_path = np.array(gt_path)[:, [0, 2]]  # Оставляем X и Z

# Загрузка предсказанных поз из camera_positions.npy
camera_positions = np.load('camera_positions.pkl', allow_pickle=True)
pred_path = np.array([pos[1].flatten() for pos in camera_positions])[:, [0, 2]]  # Оставляем X и Z

# Выравниваем длины массивов
min_len = min(len(gt_path), len(pred_path))
gt_path, pred_path = gt_path[:min_len], pred_path[:min_len]

# Масштабирование X координат
scaling_factor_x = np.max(np.abs(gt_path[:, 0])) / np.max(np.abs(pred_path[:, 0]))
pred_path[:, 0] *= scaling_factor_x

# Инвертирование Z координат
pred_path[:, 1] = -pred_path[:, 1]

# Устанавливаем начальную позицию предсказаний, чтобы она совпадала с начальной позицией Ground Truth
# Сначала сдвигаем предсказания так, чтобы их начальная точка была в (0, 0)
pred_path -= pred_path[0]

# Теперь добавляем начальную точку из Ground Truth
pred_path += gt_path[0]

# Проверяем новый диапазон для предсказанных координат
print("Scaled Predicted X range:", np.min(pred_path[:, 0]), np.max(pred_path[:, 0]))
print("Inverted Predicted Z range:", np.min(pred_path[:, 1]), np.max(pred_path[:, 1]))

# Проверяем начальные точки
print("Ground Truth initial position:", gt_path[0])
print("Predicted initial position:", pred_path[0])

def visualize_paths(gt_path, pred_path, file_out="camera_trajectory_plot.html"):
    output_file(file_out, title="Camera Trajectory Visualization")

    tools = "pan,wheel_zoom,box_zoom,reset"
    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T

    source = ColumnDataSource(data=dict(gtx=gt_x, gty=gt_y, px=pred_x, py=pred_y))

    # График с двумя траекториями
    fig = figure(title="Camera Trajectory", tools=tools, match_aspect=True, width_policy="max",
                 toolbar_location="above", x_axis_label="X", y_axis_label="Z")

    fig.line("gtx", "gty", source=source, color="blue", legend_label="Ground Truth", line_width=2)
    fig.line("px", "py", source=source, color="green", legend_label="Predicted", line_width=2)

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    # Отображаем график
    show(column(Div(text="<h1>Camera Trajectory Visualization</h1>"), fig, sizing_mode='scale_width'))

# Визуализируем траекторию
visualize_paths(gt_path, pred_path)