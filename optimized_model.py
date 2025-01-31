import torch
import torch.quantization
from torch.utils import data
from main_for_video import pt_utils, model_zoo, io, VideoTestDataset
import configs.vcod_finetune as cfg
import os
import torch
import onnx
import tensorrt as trt
import os
import torch.nn.utils.prune as prune  # Импортируйте модуль прунинга


class ModelConverter:
    def __init__(self, model_path, model_class, cfg_attributes):
        self.model_path = model_path
        self.model_class = model_class
        self.cfg_attributes = cfg_attributes
        self.model = None

    def _initialize_model(self):
        pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)
        self.model = self.model_class(num_frames=cfg.num_frames, pretrained=False, use_checkpoint=False)
        self.model.to("cuda:0")
        io.load_weight(model=self.model, load_path=self.model_path, strict=True)
        self.model.eval()

    def prune(self, pruning_ratio=0.5, norm_type=1, dim=0):
        """
        Применение структурированного прунинга к модели.
        
        :param pruning_ratio: Доля весов, которые нужно обрезать (по умолчанию 0.5).
        :param norm_type: Тип нормы для оценки важности (1 для L1, 2 для L2).
        :param dim: Размерность, по которой выполняется прунинг (0 для строк, 1 для столбцов).
        """
        self._initialize_model()

        parameters_to_prune = []

        # # Добавляем только линейные слои для прунинга
        # for name, module in self.model.named_modules():
        #     if isinstance(module, torch.nn.Linear):  # Применяем только к линейным слоям
        #         parameters_to_prune.append((module, 'weight'))
        # Добавляем все слои с атрибутом 'weight' для прунинга
        for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None and not isinstance(module, torch.nn.BatchNorm2d):
                    # Проверяем, что вес является многомерным тензором
                    if module.weight.dim() > 1:
                        parameters_to_prune.append((module, 'weight'))
                    # else:
                    #     print(f"Skipping {name} because its weight is 1-dimensional (shape: {module.weight.shape})")
        # Применение структурированного прунинга
        for module, param_name in parameters_to_prune:
            prune.ln_structured(
                module,
                name=param_name,
                amount=pruning_ratio,
                n=norm_type,  # L1 норма (1) или L2 норма (2)
                dim=dim,      # 0 для прунинга строк, 1 для прунинга столбцов
            )

        # Удаление масок прунинга и сохранение обрезанных весов
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        # Сохранение модели с обрезанными весами
        state_dict = self.model.state_dict()
        torch.save(state_dict, f'{self.model_path.split(".")[0]}_pruning_{pruning_ratio}.pth')
        print(f"Pruned model saved to {self.model_path.split('.')[0]}_pruning_{pruning_ratio}.pth")

    def half_presicion(self):
        self._initialize_model()
        for name, param in self.model.named_parameters():
            param.data = param.data.to(torch.float16)
            print(f"{name}: dtype={param.dtype}, shape={param.shape}")  
        half_presicion_model_path = self.model_path.replace('.pth', '__half_presicion.pth')
        torch.save(self.model.state_dict(), half_presicion_model_path)
        print(f"Оптимизированная модель сохранена по пути: {half_presicion_model_path}\n")

# Пример использования
if __name__ == '__main__':
    model_path = 'pvtv2-b5-5frame-zoomnext.pth'
    model_class = model_zoo.__dict__.get('PvtV2B5_ZoomNeXt')
    cfg_attributes = {name: value for name, value in vars(cfg).items()}

    converter = ModelConverter(model_path, model_class, cfg_attributes)
    converter.to_engine()
    converter.half_presicion()
    # converter.prune()
