import os
import cv2
import numpy as np
import re
import shutil

def count_files_in_directory(directory, extensions):
    """Подсчитывает файлы в директории с указанными расширениями."""
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(extensions)])

def main(fold_name):
    images_dir = f'ZoomNextDataset/{fold_name}/images'
    masks_dir = f'ZoomNextDataset/{fold_name}/masks'
    output_dir = 'test1'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

    def extract_number_and_letter(filename):
        match = re.search(r'(\d+)([a-zA-Z]*)', filename)
        if match:
            number = int(match.group(1))
            letter = match.group(2)
            return number, letter
        return float('inf'), ''

    image_files.sort(key=extract_number_and_letter)
    mask_files.sort(key=extract_number_and_letter)

    groups = {}
    for image_file in image_files:
        number, letter = extract_number_and_letter(image_file)
        key = letter

        if key not in groups:
            groups[key] = []
        groups[key].append((number, image_file))

    for letter, files in groups.items():
        print(f'Обрабатываем группу: {letter}, количество файлов: {len(files)}')
        group_number = 1
        current_folder_name = f'{fold_name}_{letter}_{group_number}'
        current_folder_path = os.path.join(output_dir, current_folder_name)

        images_folder_path = os.path.join(current_folder_path, 'Imgs')
        masks_folder_path = os.path.join(current_folder_path, 'GT')
        os.makedirs(images_folder_path, exist_ok=True)
        os.makedirs(masks_folder_path, exist_ok=True)

        last_number = None
        image_count = 0
        mask_count = 0

        for number, image_file in files:
            if last_number is None or abs(number - last_number) > 5:
                if last_number is not None:
                    group_number += 1
                    current_folder_name = f'{fold_name}_{letter}_{group_number}'
                    current_folder_path = os.path.join(output_dir, current_folder_name)

                    images_folder_path = os.path.join(current_folder_path, 'Imgs')
                    masks_folder_path = os.path.join(current_folder_path, 'GT')
                    os.makedirs(images_folder_path, exist_ok=True)
                    os.makedirs(masks_folder_path, exist_ok=True)

            new_image_name = f"{image_count * 5:05d}.jpg"
            new_mask_name = f"{image_count * 5:05d}.png"

            image_path = os.path.join(images_dir, image_file)
            new_image_path = os.path.join(images_folder_path, new_image_name)
            if not os.path.exists(new_image_path):
                cv2.imwrite(new_image_path, cv2.imread(image_path))
                image_count += 1

            mask_file = f"{image_file[:-4]}.png"
            mask_path = os.path.join(masks_dir, mask_file)
            new_mask_path = os.path.join(masks_folder_path, new_mask_name)

            if os.path.exists(mask_path):
                if not os.path.exists(new_mask_path):
                    cv2.imwrite(new_mask_path, cv2.imread(mask_path))
                    mask_count += 1
            else:
                black_mask = np.zeros((384, 384, 3), dtype=np.uint8)
                cv2.imwrite(new_mask_path, black_mask)
                mask_count += 1

            last_number = number

        # Проверка количества файлов в папках и удаление, если их меньше 5
        image_files_count = count_files_in_directory(images_folder_path, '.jpg')
        # mask_files_count = count_files_in_directory(masks_folder_path, '.png')
        print(f'Папка: {current_folder_path}, Изображения: {image_files_count}')

        if image_files_count < 5:
            shutil.rmtree(current_folder_path)
            print(f'Удалена папка: {current_folder_path}')

    print("Готово!")


if __name__ == '__main__':
    for fold_name in ['VID_20241103_143353', 'VID_20241123_135831', 'VID_20241123_140459']:
        main(fold_name)
