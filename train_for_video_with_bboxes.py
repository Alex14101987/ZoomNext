# python train_for_video_with_bboxes.py --config configs/vcod_finetune.py --model-name PvtV2B5_ZoomNeXt
from train_test_code import *
from test_for_video_with_bboxes import test
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def train(model, cfg):
    # Инициализация датасета и загрузчика
    dataset = VideoDataset(
        dataset_root=cfg.train.data.dataset_infos.ZoomNext_dataset.root,
        shape=cfg.train.data.shape,
    )
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
    )

    # Инициализация счетчиков и оптимизатора
    counter = recorder.TrainingCounter(
        epoch_length=len(loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )
    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve()
    scaler = pipeline.Scaler(optimizer, cfg.train.use_amp, set_to_none=cfg.train.optimizer.set_to_none)

    if cfg.train.bn.freeze_encoder:
        model.encoder.requires_grad_(False)

    torch.autograd.set_detect_anomaly(True)  # Включаем обнаружение аномалий

    for _ in range(counter.num_epochs):
        model.train()
        if cfg.train.bn.freeze_status:
            pt_utils.frozen_bn_stats(model.encoder, freeze_affine=cfg.train.bn.freeze_affine)

        for _, batch in enumerate(loader):
            # Обновление learning rate
            scheduler.step(curr_idx=counter.curr_iter)
            # print(batch['data'].keys())
            # Подготовка данных
            data_to_move = {
                "image_s": batch['data']["image_s"],
                "image_m": batch['data']["image_m"],
                "image_l": batch['data']["image_l"],
                "boxes": batch['data']["boxes"]
            }
            
            data_batch = pt_utils.to_device(data=data_to_move, device=cfg.device)
            data_batch = {k: v.flatten(0, 1) for k, v in data_batch.items()}
            data_batch = process_images(data_batch)

            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp):
                outputs = model(data=data_batch, iter_percentage=counter.curr_percent)
                pred_boxes, masks = postprocess(outputs)
                true_boxes = data_batch["boxes"]
                print('===train_shapes===', pred_boxes.shape, true_boxes.shape, masks.shape)
                loss = box_loss_fn(pred_boxes, true_boxes, masks)

                # Проверка на NaN и Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Loss is NaN or Inf. Skipping this batch.")
                    continue  # Пропустить итерацию, если потери некорректны

                scaler.calculate_grad(loss=loss)  # Масштабируем потери и выполняем обратный проход
                # print('===use_fp16===', scaler.use_fp16)
                scaler.update_grad()

            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        counter.update_epoch_counter()

    # cfg.tb_logger.close_tb()

    print('Starting test.........................................')
    test(model=model, cfg=cfg, loader=loader)



def main():
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)
    model_class = model_zoo.__dict__.get('PvtV2B5_ZoomNeXt')
    model = model_class(num_frames=cfg.num_frames)
    model.to(cfg.device)
    train(model=model, cfg=cfg)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print('time: ', time.time() - start_time)