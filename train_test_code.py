import argparse
import glob, datetime
import shutil
import time
import albumentations as A
import torch
import yaml
from mmengine import Config
from torch.utils import data
import methods as model_zoo
from utils import io, pipeline, pt_utils, py_utils, recorder
import os
from torch.utils import data
import cv2
import numpy as np
import torch.nn.functional as F
from temp import draw_bounding_boxes, visualize_boxes


def ensure_tensor_size(tensor, target_shape):
    # Проверяем, совпадает ли количество элементов
    target_num_elements = target_shape[0] * target_shape[1] * target_shape[2] * target_shape[3]
    current_num_elements = tensor.numel()

    if current_num_elements < target_num_elements:
        # Если текущий тензор меньше целевого, добавляем пустые тензоры
        padding = torch.zeros(target_num_elements - current_num_elements).view(-1)
        tensor = torch.cat((tensor.view(-1), padding), dim=0).view(target_shape)
    elif current_num_elements > target_num_elements:
        # Если текущий тензор больше целевого, обрезаем его
        tensor = tensor.view(-1)[:target_num_elements].view(target_shape)
    else:
        # Если количество элементов совпадает, просто изменяем размерность
        tensor = tensor.view(target_shape)

    return tensor

def process_images(data_batch):
    # Целевые размерности
    target_shapes = {
        "image_s": (5, 3, 192, 192),
        "image_m": (5, 3, 384, 384),
        "image_l": (5, 3, 768, 768)
    }

    for key in target_shapes.keys():
        if key in data_batch:
            # print(f'===input data["{key}"]=== <class {type(data_batch[key])}> {data_batch[key].shape}')
            data_batch[key] = ensure_tensor_size(data_batch[key], target_shapes[key])
            # print(f'===output data["{key}"]=== <class {type(data_batch[key])}> {data_batch[key].shape}')
    return data_batch

def custom_collate_fn(batch):
    images_s = []
    images_m = []
    images_l = []
    boxes = []

    for item in batch:
        # Средняя версия изображения
        image_m = item["data"]["image_m"]
        images_m.append(image_m)

        # Создаем уменьшенную версию изображения из image_m
        image_s = torch.nn.functional.interpolate(image_m, size=(192, 192), mode='bilinear', align_corners=False)
        images_s.append(image_s)

        # Увеличенная версия изображения
        image_l = torch.nn.functional.interpolate(image_m, scale_factor=2, mode='bilinear', align_corners=False)
        images_l.append(image_l)

        # Проверяем размерность bounding boxes
        box_tensor = item["data"]["boxes"]
        if box_tensor.dim() == 3:  # Если размерность 3, добавляем дополнительное измерение
            box_tensor = box_tensor.unsqueeze(1)  # Преобразуем в (N, 1, 6)

        boxes.append(box_tensor)

    # Объединяем изображения
    images_s_tensor = torch.cat(images_s, dim=0)
    images_m_tensor = torch.cat(images_m, dim=0)
    images_l_tensor = torch.cat(images_l, dim=0)

    # Объединяем bounding boxes
    boxes_tensor = torch.cat(boxes, dim=0) if boxes else torch.zeros(0, 1, 6)

    return {
        "data": {
            "image_s": images_s_tensor,
            "image_m": images_m_tensor,
            "image_l": images_l_tensor,
            "boxes": boxes_tensor,
        }
    }

# def box_loss_fn(pred_boxes, true_boxes):
#     # Убедитесь, что pred_boxes имеет размерность [N, 4, H, W]
#     pred_boxes = pred_boxes.squeeze(-1).squeeze(-1)  # Теперь [N, 4]

#     # Убедитесь, что true_boxes имеет размерность [N, 4]
#     true_boxes = true_boxes.squeeze(1)  # Удаляем размерность 1, теперь [N, 4]

#     # Приведение типов
#     pred_boxes = pred_boxes.to(torch.float32)
#     true_boxes = true_boxes.to(torch.float32)

#     # Потери для координат центра
#     center_loss = F.mse_loss(pred_boxes[:, 0], true_boxes[:, 0]) + F.mse_loss(pred_boxes[:, 1], true_boxes[:, 1])

#     # Потери для ширины и высоты
#     size_loss = F.mse_loss(pred_boxes[:, 2], true_boxes[:, 2]) + F.mse_loss(pred_boxes[:, 3], true_boxes[:, 3])

#     # Взвешенная сумма потерь
#     alpha = 9  # Коэффициент для центров
#     beta = 1   # Коэффициент для размеров
#     loss = (alpha * center_loss + beta * size_loss)*100

#     print(f'===boxes=== Predict: {[f"{x:.4f}" for x in pred_boxes[0].tolist()]} '
#       f'True: {[f"{x:.4f}" for x in true_boxes[0].tolist()]} '
#       f'Loss: {loss.item():.4f}')

#     visualize_boxes(pred_boxes, true_boxes)
#     return loss

import torch
import torch.nn.functional as F

def box_loss_fn(pred_boxes, true_boxes, masks):
    # Отладочные принты для проверки входных данных
    # print("pred_boxes:", pred_boxes.shape)
    print("true_boxes:", true_boxes.shape)
    # print("masks:", masks.shape)

    num_images = pred_boxes.size(0)  # 5
    num_pred_boxes = pred_boxes.size(1)  # 247
    coord_loss = 0.0
    class_loss = 0.0
    confidence_loss = 0.0

    for i in range(num_images):
        true_box = true_boxes[i, 0]  # Это будет тензор с 6 элементами

        if true_box[0].item() >= 0:  # Используем .item() для извлечения скалярного значения
            for j in range(num_pred_boxes):
                if masks[i][j] == 1:  # Если маска действительна
                    pred_box = pred_boxes[i, j, 2:6]  # [center_x, center_y, width, height]

                    # Преобразуем координаты из формата [center_x, center_y, width, height] в [x1, y1, x2, y2]
                    pred_x1 = pred_box[0] - pred_box[2] / 2
                    pred_y1 = pred_box[1] - pred_box[3] / 2
                    pred_x2 = pred_box[0] + pred_box[2] / 2
                    pred_y2 = pred_box[1] + pred_box[3] / 2
                    pred_box = torch.tensor([pred_x1, pred_y1, pred_x2, pred_y2], device=pred_boxes.device, requires_grad=True)

                    # Преобразуем истинный бокс
                    true_x1 = true_box[2] - true_box[4] / 2
                    true_y1 = true_box[3] - true_box[5] / 2
                    true_x2 = true_box[2] + true_box[4] / 2
                    true_y2 = true_box[3] + true_box[5] / 2
                    true_box_converted = torch.tensor([true_x1, true_y1, true_x2, true_y2], device=pred_boxes.device, requires_grad=False)

                    # Вычисляем потери для координат
                    coord_loss += F.smooth_l1_loss(pred_box.unsqueeze(0), true_box_converted.unsqueeze(0))

                    # Получаем class_id на том же устройстве, что и pred_boxes
                    class_id = true_box[0].long().to(pred_boxes.device)

                    # Вычисляем класс потерь
                    class_loss += F.cross_entropy(pred_boxes[i, j, 0:1].unsqueeze(0), class_id.unsqueeze(0))

                    # Применяем сигмоиду к предсказанной уверенности
                    pred_confidence = torch.sigmoid(pred_boxes[i, j, 1:2])

                    # Вычисляем потери уверенности
                    target_confidence = torch.tensor([1.0], device=pred_boxes.device).expand_as(pred_boxes[i, j, 1:2])
                    confidence_loss += F.binary_cross_entropy(pred_confidence, target_confidence)

        else:
            for j in range(num_pred_boxes):
                if masks[i][j] == 1:  # Если маска действительна
                    target_confidence = torch.tensor([0.0], device=pred_boxes.device).expand_as(pred_boxes[i, j, 1:2])
                    pred_confidence = torch.sigmoid(pred_boxes[i, j, 1:2])  # Применяем сигмоиду
                    confidence_loss += F.binary_cross_entropy(pred_confidence, target_confidence)

    # Общая потеря
    total_loss = coord_loss + class_loss + confidence_loss
    print("===Total loss===", total_loss.item())

    return total_loss  # Убедитесь, что это тензор, который требует градиентов






def construct_frame_transform():
    return A.Compose(
        [
            A.Rotate(limit=0, p=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101),  # Не поворачиваем
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0, p=1.0),  # Не изменяем яркость и контраст
        ]
    )

def construct_video_transform():
    return A.ReplayCompose(
        [
            A.HorizontalFlip(p=0.0),  # Никогда не переворачиваем
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0, p=1.0),  # Не изменяем яркость и контраст
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0, p=1.0),  # Не изменяем цвет
        ]
    )

# def construct_frame_transform():
#     return A.Compose(
#         [
#             A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101),
#             A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.5),
#         ]
#     )


# def construct_video_transform():
#     return A.ReplayCompose(
#         [
#             A.HorizontalFlip(p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
#             A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
#         ]
#     )


# def get_number_from_tail(string):
#     return int(re.findall(pattern="\\d+$", string=string)[0])

# class VideoDataset(data.Dataset):
#     def __init__(self, dataset_root: str, shape: dict, num_frames: int = 1):
#         super().__init__()
#         self.shape = shape
#         self.num_frames = num_frames
#         self.stride = num_frames - 1 if num_frames > 1 else 1

#         self.total_data_paths = []
#         for animal in os.listdir(dataset_root):
#             animal_path = os.path.join(dataset_root, animal)
#             if os.path.isdir(animal_path):
#                 image_paths = sorted(glob.glob(os.path.join(animal_path, "*.jpg")))
#                 box_paths = sorted(glob.glob(os.path.join(animal_path, "*.txt")))

#                 valid_names = []
#                 for image_path in image_paths:
#                     image_name = os.path.basename(image_path).replace('.jpg', '')
#                     corresponding_box_path = os.path.join(animal_path, f"{image_name}.txt")
#                     if corresponding_box_path in box_paths:
#                         valid_names.append((image_path, corresponding_box_path))

#                 for clip_idx in range(0, len(valid_names), self.stride):
#                     clip_info = []
#                     for i in range(self.num_frames):
#                         if clip_idx + i < len(valid_names):
#                             image_path, box_path = valid_names[clip_idx + i]
#                             clip_info.append((image_path, box_path, animal, i, clip_idx))
#                     if clip_info:
#                         self.total_data_paths.append(clip_info)
class VideoDataset(data.Dataset):
    def __init__(self, dataset_root: str, shape: dict, num_frames: int = 5):
        super().__init__()
        self.shape = shape
        self.num_frames = num_frames
        self.stride = num_frames  # Изменено на num_frames

        self.total_data_paths = []
        for animal in os.listdir(dataset_root):
            animal_path = os.path.join(dataset_root, animal)
            if os.path.isdir(animal_path):
                image_paths = sorted(glob.glob(os.path.join(animal_path, "*.jpg")))
                box_paths = sorted(glob.glob(os.path.join(animal_path, "*.txt")))

                valid_names = []
                for image_path in image_paths:
                    image_name = os.path.basename(image_path).replace('.jpg', '')
                    corresponding_box_path = os.path.join(animal_path, f"{image_name}.txt")
                    if corresponding_box_path in box_paths:
                        valid_names.append((image_path, corresponding_box_path))

                for clip_idx in range(0, len(valid_names), self.stride):
                    clip_info = []
                    current_index = 0  # Начинаем с 0 для индекса

                    # Добавляем доступные кадры в clip_info
                    for i in range(self.num_frames):
                        if clip_idx + i < len(valid_names):
                            image_path, box_path = valid_names[clip_idx + i]
                            clip_info.append((image_path, box_path, animal, current_index, clip_idx))
                            current_index += 1  # Увеличиваем индекс

                    # Убедимся, что clip_info имеет нужную длину
                    if len(clip_info) == self.num_frames:
                        # Сортируем clip_info по индексу изображения
                        clip_info.sort(key=lambda x: int(os.path.basename(x[0]).replace('.jpg', '')))
                        clip_info = [(clip_info[index][0], clip_info[index][1], clip_info[index][2], index, clip_idx) for index in range(len(clip_info))]
                        self.total_data_paths.append(clip_info)

        self.frame_specific_transformation = construct_frame_transform()
        self.frame_share_transformation = construct_video_transform()

    def __getitem__(self, index):
        images = []
        boxes = []
        paths = []  # Новый список для хранения путей к изображениям

        clip_info = self.total_data_paths[index]

        for image_path, box_path, _, idx_in_group, _ in clip_info:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            box = self.load_yolo_boxes(box_path)
            paths.append(image_path)

            if idx_in_group == 0:
                shared_transformed = self.frame_share_transformation(image=image, box=box)
            else:
                shared_transformed = A.ReplayCompose.replay(
                    saved_augmentations=shared_transformed["replay"], image=image, box=box
                )
            specific_transformed = self.frame_specific_transformation(
                image=shared_transformed["image"], box=shared_transformed["box"]
            )
            image = specific_transformed["image"]
            box = specific_transformed["box"]

            image_s = cv2.resize(image, (384, 384))
            image_s = torch.from_numpy(image_s).float().div(255).permute(2, 0, 1)
            
            images.append(image_s)
            boxes.append(torch.from_numpy(box).unsqueeze(0))

            if len(images) == self.num_frames:
                break

        images_tensor = torch.stack(images, dim=0)
        boxes_tensor = torch.stack(boxes, dim=0) if boxes else torch.zeros(self.num_frames, 1, 4)

        return dict(
            data={
                "paths": paths,  # Добавляем пути к изображениям
                "image_m": images_tensor,  # Размерность [N, 3, 384, 384]
                "boxes": boxes_tensor,      # Размерность [N, 1, 4]
            }
        )

    def load_yolo_boxes(self, box_path):
        boxes = []
        with open(box_path, 'r') as f:
            for line in f.readlines():
                _, center_x, center_y, width, height = map(float, line.strip().split())
                class_id = 0
                confidence = 1
                boxes.append([class_id, confidence, center_x, center_y, width, height])
        return np.array(boxes)

    def __len__(self):
        return len(self.total_data_paths)
    
def parse_cfg():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data-cfg", type=str, default="./dataset.yaml")
    parser.add_argument("--model-name", type=str, choices=model_zoo.__dict__.keys())
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--metric-names",
        nargs="+",
        type=str,
        default=["sm", "wfm", "mae", "em", "fmeasure", "iou", "dice"],
        choices=recorder.GroupedMetricRecorder.supported_metrics,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    with open(cfg.data_cfg, mode="r") as f:
        cfg.dataset_infos = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    # cfg.exp_name = py_utils.construct_exp_name(model_name=cfg.model_name, cfg=cfg)
    cfg.output_dir = os.path.join(cfg.proj_root, cfg.output_dir)
    # cfg.path = py_utils.construct_path(output_dir=cfg.output_dir, exp_name=cfg.exp_name)
    cfg.device = "cuda:0"

    # py_utils.pre_mkdir(cfg.path)
    # with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
        # f.write(cfg.pretty_text)
    # shutil.copy(__file__, cfg.path.trainer_copy)

    # cfg.tb_logger = recorder.TBLogger(tb_root=cfg.path.tb)
    return cfg


cfg = parse_cfg()

import torch


def postprocess(output):
    # Обработка выходных данных
    pred_boxes = output['pred_boxes']  # Извлекаем предсказания из словаря
    # print('===pred_boxes.shape===', pred_boxes.shape)  # Output shape should be [5, 6000, 6]
    batch_size = pred_boxes.shape[0]
    
    results = []  # Список для хранения предсказаний для каждого кадра
    masks = []  # Список для хранения масок

    for b in range(batch_size):
        # Извлекаем предсказания для текущего кадра
        predictions = pred_boxes[b]  # Shape: [6000, 6]

        # Фильтруем предсказания по порогу уверенности
        valid_predictions = predictions[predictions[:, 1] > 0.5]  # Убедитесь, что score > 0.5
        
        # Если есть валидные предсказания
        if valid_predictions.shape[0] > 0:
            boxes = valid_predictions[:, 2:6]  # [center_x, center_y, width, height]
            scores = valid_predictions[:, 1]  # confidence scores
            class_ids = valid_predictions[:, 0].long()  # class_ids

            # Преобразуем box из формата [center_x, center_y, width, height] в [x1, y1, x2, y2]
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            boxes_tensor = torch.stack((x1, y1, x2, y2), dim=1)  # Shape: [num_boxes, 4]

            # Форматируем выход
            frame_result = []
            for i in range(boxes_tensor.shape[0]):
                box = boxes_tensor[i]
                score = scores[i]
                class_id = class_ids[i]

                # Преобразуем box обратно в формат [center_x, center_y, width, height]
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]

                # Добавляем предсказание в результат
                frame_result.append([class_id.item(), score.item(), center_x.item(), center_y.item(), width.item(), height.item()])
            
            # Преобразуем список предсказаний в тензор
            frame_result_tensor = torch.tensor(frame_result, dtype=torch.float32)  # Shape: [num_boxes, 6]
            results.append(frame_result_tensor)  # Добавляем предсказания для текущего кадра
            masks.append(torch.ones(frame_result_tensor.shape[0], dtype=torch.int))  # Маска для валидных предсказаний
        else:
            # Если предсказаний нет, добавляем пустой тензор с 6 колонками
            results.append(torch.empty((0, 6), dtype=torch.float32))  # Добавляем пустой тензор
            masks.append(torch.zeros(1, dtype=torch.int))  # Указываем, что предсказаний нет

    # Приводим результаты и маски к нужной размерности
    max_predictions = max([result.shape[0] for result in results])  # Максимальное количество предсказаний в кадре
    padded_results = torch.zeros((batch_size, max_predictions, 6), dtype=torch.float32)  # Паддинг для результатов
    padded_masks = torch.zeros((batch_size, max_predictions), dtype=torch.int)  # Паддинг для масок

    for i in range(batch_size):
        padded_results[i, :results[i].shape[0], :] = results[i]  # Заполняем предсказания
        padded_masks[i, :masks[i].shape[0]] = masks[i]  # Заполняем маски

    print('==========len(results)============', len(results), [len(r) for r in results])
    
    # Теперь возвращаем результаты и маски
    return padded_results, padded_masks  # Возвращаем тензор предсказаний и соответствующие маски

import cv2
def draw_bounding_boxes(image, boxes, color=(0,0,255)):
    """
    Отрисовывает bounding boxes на изображении.
    
    :param image: Исходное изображение в формате numpy array.
    :param boxes: Bounding boxes в формате (x_center, y_center, width, height).
    :return: Изображение с отрисованными bounding boxes.
    """
    for box in boxes:
        # if np.all(box == 0):
        #     continue
        class_id, conf, x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.shape[1])
        y1 = int((y_center - height / 2) * image.shape[0])
        x2 = int((x_center + width / 2) * image.shape[1])
        y2 = int((y_center + height / 2) * image.shape[0])
        
        # Рисуем прямоугольник на изображении
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Форматируем строку с конфиденсом
        conf_text = f"Conf: {conf:.2f}"
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Чтобы текст не выходил за пределы изображения
        cv2.putText(image, conf_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image