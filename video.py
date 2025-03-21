import cv2
import os

# Папка с изображениями
folder_path = 'C:/Users/steph/PycharmProjects/SIFT_trajectory/dataset2/image_l'
output_video_path = 'C:/Users/steph/PycharmProjects/SIFT_trajectory/output.avi'

# Загружаем файлы и сортируем
frames = sorted(os.listdir(folder_path))

# Проверяем размер первого кадра
first_frame = cv2.imread(os.path.join(folder_path, frames[0]))
frame_height, frame_width, _ = first_frame.shape

# Параметры видео (замедлили FPS с 10 до 5)
FPS = 5
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (frame_width, frame_height))
video_writer_matches = cv2.VideoWriter(output_video_path.replace('.avi', '_matches.avi'), fourcc, FPS,
                                       (frame_width * 2, frame_height))

# Инициализируем SIFT
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Создаём видео
for i in range(len(frames) - 1):
    file_path1 = os.path.join(folder_path, frames[i])
    file_path2 = os.path.join(folder_path, frames[i + 1])

    img1 = cv2.imread(file_path1)
    img2 = cv2.imread(file_path2)

    if img1 is None or img2 is None:
        continue

    # Добавляем кадр в первое видео (обычное)
    video_writer.write(img1)

    # Обнаруживаем ключевые точки
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        continue

    # Используем knnMatch и фильтруем Good Matches
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Ограничиваем количество отображаемых дескрипторов (например, 50)
    max_descriptors = 50
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_descriptors]

    # Рисуем только Good Matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Добавляем кадр в видео с ключевыми точками
    video_writer_matches.write(match_img)

# Добавляем последний кадр
video_writer.write(cv2.imread(os.path.join(folder_path, frames[-1])))

# Закрываем видеофайлы
video_writer.release()
video_writer_matches.release()

# Объединяем два видео в одно
final_output = output_video_path.replace('.avi', '_final.avi')

cmd = f'ffmpeg -y -i "{output_video_path}" -i "{output_video_path.replace(".avi", "_matches.avi")}" -filter_complex "[0:v:0] [1:v:0] concat=n=2:v=1[outv]" -map "[outv]" "{final_output}"'
os.system(cmd)

print(f"Видео сохранено в: {final_output}")


# Функция для воспроизведения видео
def play_video(video_path, window_name="Видео"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка открытия видео: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(window_name, frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):  # Увеличили задержку кадров (200 мс вместо 30 мс)
            break

    cap.release()
    cv2.destroyAllWindows()


# Запускаем воспроизведение видео
play_video(output_video_path, "Оригинальное видео")
play_video(output_video_path.replace('.avi', '_matches.avi'), "Видео с ключевыми точками (Good Matches)")
play_video(final_output, "Объединённое видео")