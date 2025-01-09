import torch, os, glob
import matplotlib.pyplot as plt
import numpy as np

# print(torch.__version__)  # Должно вывести: 2.1.2
# print(torch.cuda.is_available())  # Должно вывести: True
# print(torch.version.cuda)
# dataset_root = "ZoomNext_dataset/Train"
# for animal in os.listdir(dataset_root):
#     animal_path = os.path.join(dataset_root, animal)
#     if os.path.isdir(animal_path):
#         images = glob.glob(os.path.join(animal_path, "*.jpg"))
#         boxes = glob.glob(os.path.join(animal_path, "*.txt"))
#         print(f"Animal: {animal}, Images: {len(images)}, Boxes: {len(boxes)}")

# import os
# import numpy as np
# import cv2
# from PIL import Image

# def overlay_images(folder1, folder2, output_video_path):
#     # Получаем список изображений в папках
#     images1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#     images2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

#     # Проверяем, есть ли изображения в обеих папках
#     if not images1 or not images2:
#         print("Одна из папок не содержит изображений.")
#         return

#     # Сравниваем количество изображений
#     min_images = min(len(images1), len(images2))

#     # Настраиваем VideoWriter
#     first_image = Image.open(os.path.join(folder1, images1[0])).convert("RGBA")
#     width, height = first_image.size
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для записи видео
#     out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))  # 30 FPS

#     for i in range(min_images):
#         # Загружаем изображения через PIL
#         img1 = Image.open(os.path.join(folder1, images1[i])).convert("RGBA")
#         img2 = Image.open(os.path.join(folder2, images2[i])).convert("RGBA")
        
#         # Изменяем размер второго изображения, если они не совпадают
#         img2 = img2.resize(img1.size, Image.LANCZOS)
        
#         # Накладываем изображения
#         blended = Image.blend(img1, img2, alpha=0.5)  # alpha контролирует степень наложения
        
#         # Преобразуем в формат, удобный для OpenCV
#         blended_cv = cv2.cvtColor(np.array(blended), cv2.COLOR_RGBA2BGR)
        
#         # Отображаем результат через OpenCV
#         cv2.imshow("Overlay", cv2.resize(blended_cv, (640, 390)))
        
#         # Записываем в видеофайл
#         out.write(blended_cv)
        
#         # Ждем 1мс нажатия клавиши
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     out.release()  # Освобождаем VideoWriter
#     cv2.destroyAllWindows()

# # Пример использования
# folder1 = 'MoCA/TestDataset_per_sq/people_autumn/GT'  # Укажите путь к первой папке с изображениями
# folder2 = 'outputs/PvtV2B5_ZoomNeXt_BS1_LR0.0001_E10_H384_W384_OPMadam_OPGMfinetune_SCconstant_AMP/exp_3/pre/moca_mask_te/people_autumn/Imgs'  # Укажите путь ко второй папке с изображениями
# output_video_path = 'output_video.avi'  # Путь к выходному видеофайлу

# overlay_images(folder1, folder2, output_video_path)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def visualize_boxes(pred_boxes, true_boxes, save_path='boxes_visualization.png'):

    fig, axes = plt.subplots(1, 5, figsize=(8 * 5, 8))  # Создаем 1 строку и num_boxes столбцов
    
    for i in range(5):
        ax = axes[i]
        ax.set_xlim(0, 1)  # Устанавливаем фиксированный масштаб по X
        ax.set_ylim(0, 1)  # Устанавливаем фиксированный масштаб по Y
        # Переворачиваем ось Y
        ax.invert_yaxis()
        # Отображение сетки
        ax.grid(True)

        # Преобразование истинных боксов из xywh в xmin, ymin, xmax, ymax
        true_box = true_boxes[i].detach().cpu().numpy()
        true_x_min = true_box[0] - true_box[2] / 2
        true_y_min = true_box[1] - true_box[3] / 2
        true_x_max = true_box[0] + true_box[2] / 2
        true_y_max = true_box[1] + true_box[3] / 2
        
        # Отрисовка истинных боксов (красный)
        ax.add_patch(patches.Rectangle((true_x_min, true_y_min),
                                        true_x_max - true_x_min,
                                        true_y_max - true_y_min,
                                        fill=False, color='red', linewidth=2, label='True Box'))

        # Преобразование предсказанных боксов из xywh в xmin, ymin, xmax, ymax
        pred_box = pred_boxes[i].detach().cpu().numpy()
        pred_x_min = pred_box[0] - pred_box[2] / 2
        pred_y_min = pred_box[1] - pred_box[3] / 2
        pred_x_max = pred_box[0] + pred_box[2] / 2
        pred_y_max = pred_box[1] + pred_box[3] / 2

        # Отрисовка предсказанных боксов (синий)
        ax.add_patch(patches.Rectangle((pred_x_min, pred_y_min),
                                        pred_x_max - pred_x_min,
                                        pred_y_max - pred_y_min,
                                        fill=False, color='blue', linewidth=2, label='Predicted Box'))

        # Добавление легенды
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Сохраняем график в файл
    plt.close(fig)  # Закрываем фигуру


# pred_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2],
#                            [0.0, 0.0, 0.1, 0.1],
#                            [0.0, 0.0, 0.2, 0.2],
#                            [0.0, 0.0, 0.1, 0.1],
#                            [0.1681, -0.1703, 0.0844, 0.0253]], device='cuda:0')

# true_boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1],
#                            [0.0, 0.0, 0.2, 0.2],
#                            [0.0, 0.0, 0.2, 0.2],
#                            [0.2, 0.2, 0.3, 0.3],
#                            [0.4734, 0.6028, 0.6031, 0.8431]], device='cuda:0')

# result = intersection_area(pred_boxes, true_boxes)
# print(f'Среднее нормализованное пересечение: {result.item()}')



def update_loss_plot(loss_values, filename='loss.png', window_size=10):
    """Обновляет график потерь, усредняя значения, и сохраняет его в файл."""
    # Усреднение по скользящему окну
    averaged_loss = [
        sum(loss_values[i:i + window_size]) / len(loss_values[i:i + window_size])
        for i in range(0, len(loss_values), window_size)
    ]
    
    # Создание графика
    plt.figure(figsize=(10, 5))
    
    max_loss = max(averaged_loss)
    plt.ylim(0, max_loss * 1.1)  # Устанавливаем верхний предел на 10% выше максимального значения
    
    plt.grid()
    plt.savefig(filename)  # Сохраняем график в файл
    plt.close()  # Закрываем фигуру


import cv2
def draw_bounding_boxes(image, boxes, color=(0,0,255)):
    """
    Отрисовывает bounding boxes на изображении.
    
    :param image: Исходное изображение в формате numpy array.
    :param boxes: Bounding boxes в формате (x_center, y_center, width, height).
    :return: Изображение с отрисованными bounding boxes.
    """
    for box in boxes:
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.shape[1])
        y1 = int((y_center - height / 2) * image.shape[0])
        x2 = int((x_center + width / 2) * image.shape[1])
        y2 = int((y_center + height / 2) * image.shape[0])
        
        # Рисуем прямоугольник на изображении
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Красный цвет

    return image

import os

def clean_annotation_files(directory_path):
    # Проходим по всем файлам в указанной папке
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # Проверяем, является ли файл текстовым
            file_path = os.path.join(directory_path, filename)
            
            # Читаем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Проверяем количество строк
            if len(lines) > 1:
                # Сохраняем только первую строку
                lines = lines[:1]
                
                # Записываем обратно в файл
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(lines)
                
                print(f"Файл '{filename}' был очищен, оставлена только первая строка.")

# Пример использования
directory = "ZoomNext_dataset_xywh\\Train\\arctic_fox"  # Замените на ваш путь
clean_annotation_files(directory)
