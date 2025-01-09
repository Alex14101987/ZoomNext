import abc, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b5
from .layers import MHSIU, RGPU, SimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from torch.nn import SmoothL1Loss



class _ZoomNeXt_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    @abc.abstractmethod
    def body(self):
        pass

    # def forward(self, data, iter_percentage=1, **kwargs):
    #     logits = self.body(data=data)

    #     if self.training:
    #         mask = data["mask"]
    #         prob = logits.sigmoid()

    #         losses = []
    #         loss_str = []

    #         sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
    #         losses.append(sod_loss)
    #         loss_str.append(f"bce: {sod_loss.item():.5f}")

    #         ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
    #         ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
    #         losses.append(ual_loss)
    #         loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")
    #         return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))
    #     else:
    #         return logits

    # def forward(self, data, iter_percentage=1, **kwargs):
    #     # Получаем логиты от основной модели
    #     logits = self.body(data=data)
    #     print(type(logits), logits)
    #     # Создаем словарь для хранения выходных данных
    #     outputs = {}

    #     if self.training:
    #         true_boxes = data["boxes"]  # Получаем истинные координаты ограничивающих прямоугольников
    #         # print('===true_boxes.shape===', true_boxes.shape)

    #         # Извлечение H и W из логитов
    #         N, _, H, W = logits.shape  # Предполагаем, что logits имеет размерность [N, 4, H, W]

    #         # Проверка размерности true_boxes
    #         if true_boxes.dim() == 3:  # [N, num_boxes, 4]
    #             num_boxes = true_boxes.size(1)

    #             # Если у нас только один bounding box на изображение
    #             if num_boxes == 1:
    #                 true_boxes = true_boxes.squeeze(1)  # Удаляем размерность 1

    #                 # Изменяем размерности true_boxes для соответствия [N, 4, H * W]
    #                 true_boxes = true_boxes.unsqueeze(2)  # Теперь [N, 4, 1]
    #                 true_boxes = true_boxes.expand(-1, -1, H)  # Теперь [N, 4, H]
    #                 true_boxes = true_boxes.unsqueeze(3)  # Теперь [N, 4, H, 1]
    #                 true_boxes = true_boxes.expand(-1, -1, -1, W)  # Теперь [N, 4, H, W]

    #             # Если у нас несколько bounding boxes на изображение
    #             elif num_boxes > 1:
    #                 # Здесь вы можете выбрать подходящий способ обработки нескольких bounding boxes
    #                 # Например, вы можете создать маску для истинных bounding boxes
    #                 # и использовать только те, которые соответствуют предсказанным logits
    #                 # В этом случае вам нужно будет изменить логику вычисления потерь

    #                 # Пример: вы можете использовать только первый bounding box для вычисления потерь
    #                 true_boxes = true_boxes[:, 0, :]  # Используем только первый bbox для упрощения
    #                 true_boxes = true_boxes.unsqueeze(2)  # Теперь [N, 4, 1]
    #                 true_boxes = true_boxes.expand(-1, -1, H)  # Теперь [N, 4, H]
    #                 true_boxes = true_boxes.unsqueeze(3)  # Теперь [N, 4, H, 1]
    #                 true_boxes = true_boxes.expand(-1, -1, -1, W)  # Теперь [N, 4, H, W]

    #             else:
    #                 raise ValueError("Expected at least one bounding box per image.")

    #             # Проверка размерностей
    #             assert logits.shape == true_boxes.shape, f"Shape mismatch: logits {logits.shape}, true_boxes {true_boxes.shape}"

    #             # Вычисляем потери
    #             box_loss = SmoothL1Loss()(logits, true_boxes)

    #             # Добавляем в выходной словарь
    #             outputs["loss"] = box_loss
    #             outputs["pred_boxes"] = logits  # Добавляем предсказанные bounding boxes в выходные данные

    #             return outputs  # Возвращаем выходные данные
    #     else:
    #         outputs["pred_boxes"] = logits  # Оставляем в исходной размерности [N, 4, H, W]
    #         return outputs  # Возвращаем выходные данные



    def forward(self, data, **kwargs):
        # Получаем логиты от основной модели
        logits = self.body(data=data)
        # print(type(logits))

        # Создаем словарь для хранения выходных данных
        outputs = {}

        if self.training:
            true_boxes = data["boxes"]  # Получаем истинные координаты ограничивающих прямоугольников

            # Извлекаем boxes, scores и classes из логитов
            boxes_list = logits  # Теперь logits - это список в формате [class_id, confidence, center_x, center_y, width, height]
            # print('===types_outputs===', type(boxes_list[0]))

            # Проверка размерностей
            N = len(boxes_list)  # Получаем количество изображений
            num_boxes = [len(boxes) for boxes in boxes_list]  # Получаем количество предсказанных боксов для каждого изображения

            # Создаем пустой тензор для предсказанных боксов
            max_boxes = max(num_boxes)  # Максимальное количество боксов в батче
            boxes_tensor = torch.zeros((N, max_boxes, 6), dtype=torch.float32)  # [N, max_boxes, 6]

            # Заполняем тензор предсказанными боксами
            for i, boxes in enumerate(boxes_list):
                for j, box in enumerate(boxes):
                    boxes_tensor[i, j, :] = torch.tensor(box, dtype=torch.float32)

            # Проверка размерностей
            print('===outputs.shapes===', boxes_tensor.shape)

            # Проверка размерности true_boxes
            if true_boxes.dim() == 3:  # [N, num_boxes, 6]
                true_num_boxes = true_boxes.size(1)

                # Проверка на наличие истинных боксов
                if true_num_boxes == 0:
                    # Если нет ни одного bounding box, создаем пустой тензор
                    true_boxes = torch.empty((N, max_boxes, 6), dtype=torch.float32)  # Пустой тензор [N, max_boxes, 6]
                else:
                    # Здесь предполагается, что true_boxes уже имеет нужную размерность
                    true_boxes = true_boxes  # Оставляем как есть

                # Проверка размерностей
                assert boxes_tensor.shape[0] == true_boxes.shape[0], f"Shape mismatch: boxes_tensor {boxes_tensor.shape}, true_boxes {true_boxes.shape}"

                # # Вычисляем потери для всех боксов
                # box_loss = self.compute_loss(boxes_tensor, true_boxes)

                # outputs["loss"] = box_loss
                outputs["pred_boxes"] = boxes_tensor  # Добавляем предсказанные bounding boxes в выходные данные
                return outputs  # Возвращаем выходные данные
        else:
            # Для режима инференса, извлекаем предсказанные боксы и уверенности
            boxes_tensor = logits  # Теперь logits - это список в формате [class_id, confidence, center_x, center_y, width, height]

            # Применяем NMS (Non-Maximum Suppression), если необходимо
            filtered_boxes_list = []
            for boxes in boxes_tensor:
                if len(boxes) > 0:
                    # Применяем NMS для каждого кадра
                    indices = self.nms(torch.tensor([box[2:4] for box in boxes]), torch.tensor([box[1] for box in boxes]), threshold=0.5)
                    filtered_boxes = [boxes[i] for i in indices]
                    filtered_boxes_list.append(filtered_boxes)
                else:
                    filtered_boxes_list.append([])

            outputs["pred_boxes"] = filtered_boxes_list  # Сохраняем результаты в выходной словарь

            return outputs

    # def compute_loss(self, pred_boxes, true_boxes):
    #     # Отладочные принты для проверки входных данных
    #     print("pred_boxes:", pred_boxes.shape)
    #     print("true_boxes:", true_boxes.shape)

    #     num_images = pred_boxes.size(0)  # 5
    #     num_pred_boxes = pred_boxes.size(1)  # 247
    #     coord_loss = 0.0
    #     class_loss = 0.0
    #     confidence_loss = 0.0

    #     for i in range(num_images):
    #         # Отладочный вывод для текущего изображения
    #         print(f"Processing image {i}: true_boxes[i] = {true_boxes[i]}")
    #         print(f"Processing image {i}: pred_boxes[i] = {pred_boxes[i]}")
    #         # Извлекаем истинный бокс для текущего изображения
    #         true_box = true_boxes[i, 0]  # Это будет тензор с 6 элементами

    #         # Проверяем, что класс действителен
    #         if true_box[0].item() >= 0:  # Используем .item() для извлечения скалярного значения
    #             # Ваши вычисления потерь для действительных боксов
    #             for j in range(num_pred_boxes):
    #                 pred_box = pred_boxes[i, j, 2:6]  # [center_x, center_y, width, height]

    #                 # Преобразуем координаты из формата [center_x, center_y, width, height] в [x1, y1, x2, y2]
    #                 pred_x1 = pred_box[0] - pred_box[2] / 2
    #                 pred_y1 = pred_box[1] - pred_box[3] / 2
    #                 pred_x2 = pred_box[0] + pred_box[2] / 2
    #                 pred_y2 = pred_box[1] + pred_box[3] / 2
    #                 pred_box = torch.tensor([pred_x1, pred_y1, pred_x2, pred_y2], device=pred_boxes.device)

    #                 # Преобразуем истинный бокс
    #                 true_x1 = true_box[2] - true_box[4] / 2
    #                 true_y1 = true_box[3] - true_box[5] / 2
    #                 true_x2 = true_box[2] + true_box[4] / 2
    #                 true_y2 = true_box[3] + true_box[5] / 2
    #                 true_box_converted = torch.tensor([true_x1, true_y1, true_x2, true_y2], device=pred_boxes.device)

    #                 # Вычисляем потери для координат
    #                 coord_loss += F.smooth_l1_loss(pred_box.unsqueeze(0), true_box_converted.unsqueeze(0))

    #                 # Получаем class_id на том же устройстве, что и pred_boxes
    #                 class_id = true_box[0].long().to(pred_boxes.device)  # Перемещаем class_id на GPU

    #                 # Вычисляем класс потерь
    #                 class_loss += F.cross_entropy(pred_boxes[i, j, 0:1].unsqueeze(0), class_id.unsqueeze(0))

    #                 # Вычисляем потери уверенности
    #                 target_confidence = torch.tensor([1.0], device=pred_boxes.device).expand_as(pred_boxes[i, j, 1:2])  # Изменяем размер целевого тензора
    #                 confidence_loss += F.binary_cross_entropy(pred_boxes[i, j, 1:2], target_confidence)

    #         else:
    #             # Если предсказанный бокс не имеет соответствующего истинного бокса,
    #             # добавляем потерю за ложное срабатывание (confidence loss)
    #             for j in range(num_pred_boxes):
    #                 target_confidence = torch.tensor([0.0], device=pred_boxes.device).expand_as(pred_boxes[i, j, 1:2])
    #                 confidence_loss += F.binary_cross_entropy(pred_boxes[i, j, 1:2], target_confidence)

    #     # Общая потеря
    #     total_loss = coord_loss + class_loss + confidence_loss
    #     print("Total coordinate loss:", coord_loss.item())
    #     print("Total class loss:", class_loss.item())
    #     print("Total confidence loss:", confidence_loss.item())
    #     print("Total loss:", total_loss.item())

    #     return total_loss


    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)

        return param_groups


# class PvtV2B2_ZoomNeXt(_ZoomNeXt_Base):
#     def __init__(
#         self,
#         pretrained=True,
#         num_frames=1,
#         input_norm=True,
#         mid_dim=64,
#         siu_groups=4,
#         hmu_groups=6,
#         use_checkpoint=False,
#     ):
#         super().__init__()
#         self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

#         self.embed_dims = self.encoder.embed_dims
#         self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
#         self.siu_5 = MHSIU(mid_dim, siu_groups)
#         self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

#         self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
#         self.siu_4 = MHSIU(mid_dim, siu_groups)
#         self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

#         self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
#         self.siu_3 = MHSIU(mid_dim, siu_groups)
#         self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

#         self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
#         self.siu_2 = MHSIU(mid_dim, siu_groups)
#         self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

#         self.tra_1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBNReLU(64, mid_dim, 3, 1, 1)
#         )

#         self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
#         # self.predictor = nn.Sequential(
#         #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#         #     ConvBNReLU(64, 32, 3, 1, 1),
#         #     nn.Conv2d(32, 1, 1),
#         # )
#         self.predictor = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             ConvBNReLU(64, 32, 3, 1, 1),
#             nn.AdaptiveAvgPool2d(1),  # Уменьшаем размерность до [N, 32, 1, 1]
#             nn.Conv2d(32, 6, 1),  # 6, чтобы предсказывать координаты (class_id, confidence, x, y, w, h)
#             # nn.ReLU()
#         )

#     def set_backbone(self, pretrained: bool, use_checkpoint: bool):
#         self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

#     def normalize_encoder(self, x):
#         x = self.normalizer(x)
#         features = self.encoder(x)
#         c2 = features["reduction_2"]
#         c3 = features["reduction_3"]
#         c4 = features["reduction_4"]
#         c5 = features["reduction_5"]
#         return c2, c3, c4, c5

#     def body(self, data):
#         l_trans_feats = self.normalize_encoder(data["image_l"])
#         m_trans_feats = self.normalize_encoder(data["image_m"])
#         s_trans_feats = self.normalize_encoder(data["image_s"])

#         # Проверка после tra_5
#         l = self.tra_5(l_trans_feats[3])
#         m = self.tra_5(m_trans_feats[3])
#         s = self.tra_5(s_trans_feats[3])
        
#         lms = self.siu_5(l=l, m=m, s=s)
#         # print('===lms after siu_5=== NaN count:', lms.isnan().sum().item(), 'из', lms.numel(), 'возможных')
        
#         x = self.hmu_5(lms)
#         # if x.isnan().any():
#         #     # print('===NaN detected in x after hmu_5, using skip connection===')
#         #     x = lms  # Используем lms как "skip connection"
#         # print('===x after hmu_5=== NaN count:', x.isnan().sum().item(), 'из', x.numel(), 'возможных')

#         l, m, s = self.tra_4(l_trans_feats[2]), self.tra_4(m_trans_feats[2]), self.tra_4(s_trans_feats[2])
#         # print('===l after tra_4=== NaN count:', l.isnan().sum().item(), 'из', l.numel(), 'возможных')
#         # print('===m after tra_4=== NaN count:', m.isnan().sum().item(), 'из', m.numel(), 'возможных')
#         # print('===s after tra_4=== NaN count:', s.isnan().sum().item(), 'из', s.numel(), 'возможных')
        
#         lms = self.siu_4(l=l, m=m, s=s)
#         x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))
#         # if x.isnan().any():
#         #     # print('===NaN detected in x after hmu_4, using skip connection===')
#         #     x = lms  # Используем lms как "skip connection"
#         # print('===x after hmu_4=== NaN count:', x.isnan().sum().item(), 'из', x.numel(), 'возможных')

#         l, m, s = self.tra_3(l_trans_feats[1]), self.tra_3(m_trans_feats[1]), self.tra_3(s_trans_feats[1])
#         # print('===l after tra_3=== NaN count:', l.isnan().sum().item(), 'из', l.numel(), 'возможных')
#         # print('===m after tra_3=== NaN count:', m.isnan().sum().item(), 'из', m.numel(), 'возможных')
#         # print('===s after tra_3=== NaN count:', s.isnan().sum().item(), 'из', s.numel(), 'возможных')
        
#         lms = self.siu_3(l=l, m=m, s=s)
#         x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))
#         # if x.isnan().any():
#         #     # print('===NaN detected in x after hmu_3, using skip connection===')
#         #     x = lms  # Используем lms как "skip connection"
#         # print('===x after hmu_3=== NaN count:', x.isnan().sum().item(), 'из', x.numel(), 'возможных')

#         l, m, s = self.tra_2(l_trans_feats[0]), self.tra_2(m_trans_feats[0]), self.tra_2(s_trans_feats[0])
#         # print('===l after tra_2=== NaN count:', l.isnan().sum().item(), 'из', l.numel(), 'возможных')
#         # print('===m after tra_2=== NaN count:', m.isnan().sum().item(), 'из', m.numel(), 'возможных')
#         # print('===s after tra_2=== NaN count:', s.isnan().sum().item(), 'из', s.numel(), 'возможных')
        
#         lms = self.siu_2(l=l, m=m, s=s)
#         x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))
#         # if x.isnan().any():
#         #     # print('===NaN detected in x after hmu_2, using skip connection===')
#         #     x = lms  # Используем lms как "skip connection"
#         # print('===x after hmu_2=== NaN count:', x.isnan().sum().item(), 'из', x.numel(), 'возможных')

#         # print('===x before tra_1=== NaN count:', x.isnan().sum().item(), 'из', x.numel(), 'возможных')
#         x = self.tra_1(x)
#         # if x.isnan().any():
#         #     # print('===NaN detected in x after tra_1, using skip connection===')
#         #     x = lms  # Используем lms как "skip connection"
#         # print('===x after tra_1=== NaN count:', x.isnan().sum().item(), 'из', x.numel(), 'возможных')
        
#         x = torch.nan_to_num(x)  # Заменяем NaN на 0 или другое значение по вашему выбору
#         # print('===x before predictor=== NaN count:', x.isnan().sum().item(), 'из', x.numel(), 'возможных')
#         output = self.predictor(x)
#         # if output.isnan().any():
#         #     # print('===NaN detected in output after predictor, using skip connection===')
#         #     output = x  # Используем x как "skip connection"
#         # print('===output after predictor=== NaN count:', output.isnan().sum().item(), 'из', output.numel(), 'возможных')
#         # output = torch.clamp(output, min=0)
#         # print('===output data.shape model videoPvtV2B5_ZoomNeXt===', output.shape)
#         return output

import torch
import torch.nn as nn
import torchvision

class PvtV2B2_ZoomNeXt(_ZoomNeXt_Base):
    def __init__(
        self,
        pretrained=False,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        num_classes=1,  # Количество классов
        max_boxes=1000,   # Максимальное количество боксов для предсказания
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=False)

        self.embed_dims = self.encoder.embed_dims
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), 
            ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),  # Уменьшаем размерность до [N, 32, 1, 1]
            nn.Conv2d(32, max_boxes * (num_classes + 1 + 4), 1),  # добавили 4 для bbox
        )

        self.num_classes = num_classes  # Инициализируем num_classes для дальнейшего использования

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5

    def nms(self, boxes, scores, threshold):
        # Перемещение тензоров на GPU, если доступно
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        boxes = boxes.to(device)
        scores = scores.to(device)

        # Теперь вызываем torchvision.ops.nms
        indices = torchvision.ops.nms(boxes, scores, threshold)
        
        return indices

    def body(self, data):
        print('===========data["image_m"].shape=============', data["image_m"].shape)
        
        # Получаем трансформированные признаки из энкодера
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        # Обработка на самом высоком уровне
        l = self.tra_5(l_trans_feats[3])
        m = self.tra_5(m_trans_feats[3])
        s = self.tra_5(s_trans_feats[3])
        
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = self.tra_4(l_trans_feats[2]), self.tra_4(m_trans_feats[2]), self.tra_4(s_trans_feats[2])
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_3(l_trans_feats[1]), self.tra_3(m_trans_feats[1]), self.tra_3(s_trans_feats[1])
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_2(l_trans_feats[0]), self.tra_2(m_trans_feats[0]), self.tra_2(s_trans_feats[0])
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        x = self.tra_1(x)
        print('===last_x.shape===', x.shape)
        output = self.predictor(x)

        # Debugging: Check for NaN or Inf in model output
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Debug: Found NaN or Inf in predictor output")
            print("Output:", output)
        else:
            print('===Not found Nan or Inf in predictor output===')
            print('===predictor_outpets.shape===', output.shape)
        # Обработка выходных данных
        batch_size, _, height, width = output.shape
        output = output.view(batch_size, -1, self.num_classes + 1 + 4, height, width)  # [N, max_boxes, num_classes + 1 + 4, H, W]

        results = []  # Список для хранения предсказаний для каждого кадра

        for b in range(batch_size):
            boxes = []
            scores = []
            classes = []

            for c in range(self.num_classes):
                class_output = output[b, :, c + 1, :, :]  # Уверенность для текущего класса
                score = class_output.max(dim=-1)[0].max(dim=-1)[0]  # Получаем максимальную уверенность
                box_indices = class_output.argmax(dim=(-1))  # Получаем индексы боксов

                # Получаем координаты боксов
                for i in range(box_indices.shape[0]):
                    if score[i] > 0.5:  # Примените порог уверенности
                        # Извлекаем координаты bounding box
                        box = output[b, box_indices[i], -4:, 0, 0]  # Извлекаем последние 4 значения для координат
                        boxes.append(box)  # Добавляем координаты
                        scores.append(score[i])
                        classes.append(c)

            # Примените NMS
            if len(boxes) > 0:
                boxes_tensor = torch.stack(boxes)  # Преобразуем в тензор
                boxes_tensor = boxes_tensor.squeeze(1)  # Переводим размерность [boxes, 1, 4] в [boxes, 4]
                scores_tensor = torch.tensor(scores)  # Преобразуем в тензор
                indices = self.nms(boxes_tensor, scores_tensor, threshold=0.5)

                # Форматируем вывод в виде class_id, confidence, center_x, center_y, width, height
                frame_result = []
                for i in indices:
                    box = boxes_tensor[i]
                    score = scores_tensor[i]
                    class_id = classes[i]

                    # Преобразуем box из формата [x1, y1, x2, y2] в [center_x, center_y, width, height]
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    frame_result.append([class_id, score.item(), center_x.item(), center_y.item(), width.item(), height.item()])
                
                results.append(frame_result)  # Добавляем предсказания для текущего кадра
            else:
                results.append([])  # Добавляем пустой список, если нет предсказанных боксов
        print('==========len(results)============', len(results), len(results[0]), len(results[1]), len(results[2]), len(results[3]), len(results[4]))
        return results  # Возвращаем список предсказаний для каждого кадра в батче





class PvtV2B5_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)


class videoPvtV2B5_ZoomNeXt(PvtV2B5_ZoomNeXt):
    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "temperal_proj" in name:
                    param_groups["retrained"].append(param)
                else:
                    param_groups["pretrained"].append(param)

        return param_groups

