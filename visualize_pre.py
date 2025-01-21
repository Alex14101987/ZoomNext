import os
import cv2
import numpy as np

# Укажите пути к папкам
predict_folder = 'outputs\\videoPvtV2B5_ZoomNeXt_BS2_LR0.0001_E10_H384_W384_OPMadam_OPGMfinetune_SCconstant_AMP_INFOfinetune\\exp_9\\pre\\moca_mask_te'
source_folder = 'MoCA\\TestDataset_per_sq'
output_folder = 'test'

# Создание выходной папки, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Определяем кодек и создаем объект VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Получаем список всех папок в source_folder
video_folders = os.listdir(source_folder)

for video_folder in video_folders:
            # # Сохраняем результат в выходной папке
    result_path = os.path.join(output_folder, video_folder)
    video_writer = cv2.VideoWriter(f'{result_path}.mp4', fourcc, 6, (384*3, 384))

    imgs_folder = os.path.join(source_folder, video_folder, 'Imgs')
    masks_folder = os.path.join(predict_folder, video_folder, 'Imgs')

    # Получаем список изображений
    img_files = sorted([f for f in os.listdir(imgs_folder) if f.endswith(('.jpg'))])
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.png')])

    for img_file, mask_file in zip(img_files, mask_files):
        img_path = os.path.join(imgs_folder, img_file)
        mask_path = os.path.join(masks_folder, mask_file)

        # Загружаем изображение и маску
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Загружаем маску в градациях серого

        # Создаем цветную маску (3 канала)
        color_mask = np.zeros_like(img)
        color_mask[:, :, 2] = mask  # Красный цвет для размеченных областей на комбинированном изображении

        # Накладываем маску на изображение
        result = cv2.addWeighted(img, 1, color_mask, 0.5, 0)  # Комбинируем изображение и маску

        # Создаем маску для отображения на правой части (сохраняя полутона из исходной маски)
        right_mask = np.stack((mask, mask, mask), axis=-1)

        # Создаем новый кадр, объединяя исходное изображение, наложенную маску и маску
        combined_frame = np.hstack((img, result, right_mask))  # Объединяем по горизонтали

        # Записываем объединенный кадр в видео
        video_writer.write(combined_frame)
    video_writer.release()
    # break

print("Обработка завершена!")
