import numpy as np
import matplotlib.pyplot as plt
import pickle

# Загружаем позиции камеры
with open('camera_positions.pkl', 'rb') as f:
    camera_positions = pickle.load(f)

# Извлекаем только трансляции (позиции) камеры для визуализации
positions = np.array([t.flatten() for R, t in camera_positions])

# Визуализация траектории
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Отображаем траекторию
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', color='b', label='Траектория камеры')

# Настройки графика
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Траектория движения камеры')

plt.legend()
plt.show()