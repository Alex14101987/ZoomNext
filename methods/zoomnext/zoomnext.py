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

    # def forward(self, data, **kwargs):
    #     # Получаем логиты от основной модели
    #     logits = self.body(data=data)
    #     # print(type(logits))

    #     # Создаем словарь для хранения выходных данных
    #     outputs = {}

    #     if self.training:
    #         true_boxes = data["boxes"]  # Получаем истинные координаты ограничивающих прямоугольников

    #         # Извлекаем boxes, scores и classes из логитов
    #         boxes_list = logits  # Теперь logits - это список в формате [class_id, confidence, center_x, center_y, width, height]
    #         # print('===types_outputs===', type(boxes_list[0]))

    #         # Проверка размерностей
    #         N = len(boxes_list)  # Получаем количество изображений
    #         num_boxes = [len(boxes) for boxes in boxes_list]  # Получаем количество предсказанных боксов для каждого изображения

    #         # Создаем пустой тензор для предсказанных боксов
    #         max_boxes = max(num_boxes)  # Максимальное количество боксов в батче
    #         boxes_tensor = torch.zeros((N, max_boxes, 6), dtype=torch.float32)  # [N, max_boxes, 6]

    #         # Заполняем тензор предсказанными боксами
    #         for i, boxes in enumerate(boxes_list):
    #             for j, box in enumerate(boxes):
    #                 boxes_tensor[i, j, :] = torch.tensor(box, dtype=torch.float32)

    #         # Проверка размерностей
    #         print('===outputs.shapes===', boxes_tensor.shape)

    #         # Проверка размерности true_boxes
    #         if true_boxes.dim() == 3:  # [N, num_boxes, 6]
    #             true_num_boxes = true_boxes.size(1)

    #             # Проверка на наличие истинных боксов
    #             if true_num_boxes == 0:
    #                 # Если нет ни одного bounding box, создаем пустой тензор
    #                 true_boxes = torch.empty((N, max_boxes, 6), dtype=torch.float32)  # Пустой тензор [N, max_boxes, 6]
    #             else:
    #                 # Здесь предполагается, что true_boxes уже имеет нужную размерность
    #                 true_boxes = true_boxes  # Оставляем как есть

    #             # Проверка размерностей
    #             assert boxes_tensor.shape[0] == true_boxes.shape[0], f"Shape mismatch: boxes_tensor {boxes_tensor.shape}, true_boxes {true_boxes.shape}"

    #             # # Вычисляем потери для всех боксов
    #             # box_loss = self.compute_loss(boxes_tensor, true_boxes)

    #             # outputs["loss"] = box_loss
    #             outputs["pred_boxes"] = boxes_tensor  # Добавляем предсказанные bounding boxes в выходные данные
    #             return outputs  # Возвращаем выходные данные
    #     else:
    #         # Для режима инференса, извлекаем предсказанные боксы и уверенности
    #         boxes_tensor = logits  # Теперь logits - это список в формате [class_id, confidence, center_x, center_y, width, height]

    #         # Применяем NMS (Non-Maximum Suppression), если необходимо
    #         filtered_boxes_list = []
    #         for boxes in boxes_tensor:
    #             if len(boxes) > 0:
    #                 # Применяем NMS для каждого кадра
    #                 indices = self.nms(torch.tensor([box[2:4] for box in boxes]), torch.tensor([box[1] for box in boxes]), threshold=0.5)
    #                 filtered_boxes = [boxes[i] for i in indices]
    #                 filtered_boxes_list.append(filtered_boxes)
    #             else:
    #                 filtered_boxes_list.append([])

    #         outputs["pred_boxes"] = filtered_boxes_list  # Сохраняем результаты в выходной словарь

    #         return outputs

    def forward(self, data, **kwargs):

        # Создаем словарь для хранения выходных данных
        outputs = {
            "pred_boxes": self.body(data=data)
        }
        # print('===outputs.shape===', outputs['pred_boxes'].shape)
        return outputs


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
        # num_anchors=9
        # max_boxes=1000,   # Максимальное количество боксов для предсказания
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
            nn.Conv2d(64, num_classes + 5, kernel_size=1)  # 9 анкерных боксов, 80 классов и 5 для bbox
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

    def body(self, data):
        # print('===========data["image_m"].shape=============', data["image_m"].shape)
        
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
        # print('===last_x.shape===', x.shape)
        output = self.predictor(x)

        # # Debugging: Check for NaN or Inf in model output
        # if torch.isnan(output).any() or torch.isinf(output).any():
        #     print("Debug: Found NaN or Inf in predictor output")
        #     print('===predictor_outputs.shape===', output.shape)
        # else:
        #     print('===Not found Nan or Inf in predictor output===')
        #     print('===predictor_outputs.shape===', output.shape)
        return output


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

