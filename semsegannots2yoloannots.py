import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def create_yolo_annotations(mask_image, image_shape, class_id=0):
    """Создает аннотацию в формате YOLO на основе бинарной маски."""
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []  # Возвращаем пустой список, если контуры не найдены

    # Объединяем все контуры в один bounding box
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    center_x = (x + x + w) / 2 / image_shape[1]
    center_y = (y + y + h) / 2 / image_shape[0]
    width = w / image_shape[1]
    height = h / image_shape[0]

    return [(class_id, center_x, center_y, width, height)]

def process_directory(root_dir, output_dir):
    """Рекурсивно обходит директорию и обрабатывает изображения и маски."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Создаем папки Train и Test
    train_output_dir = os.path.join(output_dir, 'Train')
    test_output_dir = os.path.join(output_dir, 'Test')

    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    all_annotations = []  # Список для хранения всех аннотаций

    for dataset_type in ['TrainDataset_per_sq', 'TestDataset_per_sq']:
        dataset_path = os.path.join(root_dir, dataset_type)
        for animal_dir in os.listdir(dataset_path):
            animal_path = os.path.join(dataset_path, animal_dir)

            if os.path.isdir(animal_path):
                animal_output_dir = os.path.join(train_output_dir if dataset_type == 'TrainDataset_per_sq' else test_output_dir, animal_dir)

                if not os.path.exists(animal_output_dir):
                    os.makedirs(animal_output_dir)

                # Обработка изображений и масок для каждого животного
                imgs_dir = os.path.join(animal_path, 'Imgs')
                gts_dir = os.path.join(animal_path, 'GT')

                for file in os.listdir(imgs_dir):
                    if file.endswith('.jpg'):
                        img_path = os.path.join(imgs_dir, file)
                        mask_path = os.path.join(gts_dir, file.replace('.jpg', '.png'))

                        if os.path.exists(mask_path):
                            image = cv2.imread(img_path)
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                            annotations = create_yolo_annotations(mask, image.shape)
                            all_annotations.append((img_path, annotations))  # Сохраняем путь к изображению и аннотации

                            # Сохраняем изображение в выходной директории
                            output_img_path = os.path.join(animal_output_dir, file)
                            cv2.imwrite(output_img_path, image)

                            # Сохраняем аннотации в файл
                            annotation_file_path = os.path.join(animal_output_dir, file.replace('.jpg', '.txt'))
                            with open(annotation_file_path, 'w') as f:
                                for annotation in annotations:
                                    f.write(f"{annotation[0]} {annotation[1]:.6f} {annotation[2]:.6f} {annotation[3]:.6f} {annotation[4]:.6f}\n")

    return all_annotations

def draw_random_bounding_box(all_annotations):
    """Выводит случайное изображение с отрисованными bounding boxes."""
    if not all_annotations:
        print("Нет аннотаций для отображения.")
        return

    img_path, annotations = random.choice(all_annotations)
    image = cv2.imread(img_path)
    image_shape = image.shape

    for annotation in annotations:
        class_id, center_x, center_y, width, height = annotation

        # Преобразование координат из YOLO-формата в абсолютные для отрисовки
        abs_x = int((center_x - width / 2) * image_shape[1])
        abs_y = int((center_y - height / 2) * image_shape[0])
        abs_w = int(width * image_shape[1])
        abs_h = int(height * image_shape[0])

        # Отрисовываем bounding box
        cv2.rectangle(image, (abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h), (255, 0, 0), 2)

    # Отображаем изображение
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    root_directory = "MoCA_Video"
    output_directory = "ZoomNext_dataset_xywh"
    all_annotations = process_directory(root_directory, output_directory)
    draw_random_bounding_box(all_annotations)

