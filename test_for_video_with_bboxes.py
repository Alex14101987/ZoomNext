# python test_for_video_with_bboxes.py --config configs/vcod_finetune.py --model-name PvtV2B5_ZoomNeXt
from train_test_code import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@torch.no_grad()
def test(model, cfg, loader):

    # model.eval()  # модель в режиме оценки

    # Создаем видеопоток для сохранения
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 5.0, (925, 693))

    overlap_frames = 0  # Количество перекрывающихся кадров
    current_image_index = 0  # Индекс текущего изображения

    # Получение путей к изображениям
    image_folder = os.path.join("ZoomNext_dataset_xywh/Train", "arctic_fox")
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    # print('===loader===', loader.dataset.total_data_paths)
    for batch_idx, batch in enumerate(loader):
        # print('===batches===', batch_idx, batch['data']['image_s'].shape, batch['data']['image_m'].shape, batch['data']['image_l'].shape, batch['data']['boxes'].shape)
        paths = batch["data"].pop("paths", None)
        data_batch = pt_utils.to_device(data=batch["data"], device=cfg.device)
                
        # Обработка данных аналогично тому, как это делается в функции train
        data_batch = {k: v.flatten(0, 1) for k, v in data_batch.items()}
        data_batch = process_images(data_batch)
        # avg_image = torch.mean(data_batch['image_m'])
        # print(f"Batch {batch_idx}: Average Image Value = {avg_image.item()}")
        with torch.no_grad():  # Отключаем градиенты для тестирования
            outputs = model(data=data_batch)
        pred_boxes, masks = postprocess(outputs)
        true_boxes = data_batch['boxes']
        # print('===test_shapes===', pred_boxes.shape, true_boxes.shape, masks.shape)
        box_loss_fn(pred_boxes, true_boxes, masks)

        pred_boxes = pred_boxes.squeeze(-1).squeeze(-1)
        pred_boxes = pred_boxes.cpu().numpy()  # Переводим в numpy для удобства
        # print(f"Batch {batch_idx}: True Boxes Shape: {true_boxes.shape}, Pred Boxes Shape: {pred_boxes.shape}")

        # Обработка кадров с перекрытием
        for frame_idx in range(pred_boxes.shape[0]):
            # loader.dataset.total_data_paths
            image_index = sum(loader.dataset.total_data_paths[batch_idx][frame_idx][-2:])
            # print('===image_index===', batch_idx, frame_idx, image_index)

            image_path = image_paths[image_index]  # Получаем путь к изображению
            image = cv2.imread(image_path)  # Читаем изображение
            current_pred_box = pred_boxes[frame_idx]
            # print('===image_path===', image_path)
            # print(f"Batch {batch_idx}: Pred Box: {current_pred_box}, True Box: {true_boxes[frame_idx].tolist()}, Path: {image_path}")
            # print(f"Batch {batch_idx}, Frame {frame_idx}: Pred Box: {current_pred_box}, True Box: {true_boxes[frame_idx].tolist()}")

            # Отрисовываем bounding boxes на изображении
            # print('===current_pred_box===', current_pred_box. shape, current_pred_box)
            # print('===true_boxes===', true_boxes. shape, true_boxes)
            for pred_box in current_pred_box:
                image = draw_bounding_boxes(image, [pred_box], color=(255, 0, 0))
            # image_with_boxes = draw_bounding_boxes(image, [true_boxes[frame_idx][0]])
            
            for i in range(true_boxes.shape[1]):
                image_with_boxes = draw_bounding_boxes(image, [true_boxes[frame_idx][i]])
                # print('======true_boxes[frame_idx][i]======', [true_boxes[frame_idx][i]])

            # Записываем обработанное изображение в видео
            out.write(image_with_boxes)
            # cv2.imshow('', image_with_boxes)
            # Увеличиваем индекс изображения для следующего кадра
            current_image_index += 1

        # После обработки последнего кадра в батче, увеличиваем индекс на количество перекрывающихся кадров
        current_image_index += (overlap_frames - 1)  # Уменьшаем на 1, чтобы не пропустить текущий кадр

    out.release()  # Закрываем видеопоток


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)
    
    # Create the model
    model_class = model_zoo.__dict__.get('PvtV2B5_ZoomNeXt')
    model = model_class(num_frames=cfg.num_frames, pretrained=cfg.pretrained, use_checkpoint=cfg.use_checkpoint)
    model.to(cfg.device)

    # Load model weights and optimizer state from a single file
    checkpoint_path = "weights/best.pth"  # Измените путь на нужный, если необходимо
    if os.path.exists(checkpoint_path):
        print(f"Loading weights and optimizer state from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Создание оптимизатора
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        
        # Проверка на соответствие
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"Warning: {e}")
            print("Loading optimizer state failed. Initializing new optimizer.")
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)  # Создаем новый оптимизатор
    else:
        print("No checkpoint found. Initializing new model and optimizer.")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    dataset = VideoDataset(
        dataset_root="ZoomNext_dataset_xywh/Train",
        shape=cfg.test.data.shape,
        num_frames=cfg.num_frames,
    )

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
    )
    test(model=model, cfg=cfg, loader=loader)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print('time: ', time.time() - start_time)
