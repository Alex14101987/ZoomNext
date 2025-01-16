import os
import cv2
import numpy as np

# Путь к папке с выходными данными
output_dir = 'test'
video_output_path = 'output_video.avi'  # Путь для сохранения видео

# Получение всех подпапок в выходной папке
subfolders = [f.path for f in os.scandir(output_dir) if f.is_dir()]

# Список для хранения всех кадров для видео
frames_for_video = []

# Проход по каждой подпапке
for subfolder in subfolders:
    images_folder_path = os.path.join(subfolder, 'images')
    masks_folder_path = os.path.join(subfolder, 'masks')

    # Получение списка файлов изображений и масок
    image_files = sorted([f for f in os.listdir(images_folder_path) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(masks_folder_path) if f.endswith('.png')])

    # Проход по всем кадрам и маскам
    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(images_folder_path, image_file)
        mask_path = os.path.join(masks_folder_path, mask_file)

        # Чтение изображения и маски
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # Убедимся, что размеры совпадают
        if image.shape[:2] != mask.shape[:2]:
            # Изменяем размер маски до размера изображения
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Склеиваем изображение и маску
        combined = np.hstack((image, mask))

        # Добавление текста с именем файла на объединенный кадр
        text = f"Image: {image_file} | Mask: {mask_file}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        frames_for_video.append(combined)

# Определяем параметры видео
height, width, _ = frames_for_video[0].shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_output_path, fourcc, 3, (width, height))

# Запись кадров в видео
for frame in frames_for_video:
    video_writer.write(frame)

# Освобождаем ресурсы
video_writer.release()

print(f"Видеоролик сохранен как {video_output_path}")
