# python train_for_video_with_bboxes.py --config configs/vcod_finetune.py --model-name PvtV2B5_ZoomNeXt
from train_test_code import *
from test_for_video_with_bboxes import test
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# def train(model, cfg):

#     dataset = VideoDataset(
#         dataset_root="ZoomNext_dataset_xywh/Train",  # Используем значение корневой директории
#         shape=cfg.train.data.shape,
#         num_frames=cfg.num_frames,
#     )

#     loader = data.DataLoader(
#         dataset=dataset,
#         batch_size=1,
#         num_workers=0,
#         shuffle=False,
#         drop_last=True,
#         pin_memory=True,
#         collate_fn=custom_collate_fn,
#         worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
#     )

#     counter = recorder.TrainingCounter(
#         epoch_length=len(loader),
#         epoch_based=cfg.train.epoch_based,
#         num_epochs=cfg.train.num_epochs,
#         num_total_iters=cfg.train.num_iters,
#     )
#     optimizer = pipeline.construct_optimizer(
#         model=model,
#         initial_lr=cfg.train.lr,
#         mode=cfg.train.optimizer.mode,
#         group_mode=cfg.train.optimizer.group_mode,
#         cfg=cfg.train.optimizer.cfg,
#     )
#     scheduler = pipeline.Scheduler(
#         optimizer=optimizer,
#         num_iters=counter.num_total_iters,
#         epoch_length=counter.num_inner_iters,
#         scheduler_cfg=cfg.train.scheduler,
#         step_by_batch=cfg.train.sche_usebatch,
#     )
#     scheduler.record_lrs(param_groups=optimizer.param_groups)
#     scheduler.plot_lr_coef_curve()
#     scaler = pipeline.Scaler(optimizer, cfg.train.use_amp, set_to_none=cfg.train.optimizer.set_to_none)
#     loss_recorder = recorder.HistoryBuffer()

#     if cfg.train.bn.freeze_encoder:
#         model.encoder.requires_grad_(False)

#     # Загрузка лучших весов и состояния оптимизатора из одного файла
#     best_weights_path = os.path.join("weights", "best.pth")
#     if os.path.exists(best_weights_path):
#         # print(f"Loading weights from {best_weights_path}...")
#         checkpoint = torch.load(best_weights_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     else:
#         print("No best weights found. Initializing new weights.")

#     best_loss = float('inf')  # Переменная для отслеживания минимальной ошибки
#     loss_values = []
#     for _ in range(counter.num_epochs):

#         model.train()
#         # if cfg.train.bn.freeze_status:
#         #     pt_utils.frozen_bn_stats(model.encoder, freeze_affine=cfg.train.bn.freeze_affine)

#         # an epoch starts
#         for batch_idx, batch in enumerate(loader):
#             scheduler.step(curr_idx=counter.curr_iter)  # update learning rate

#             data_batch = pt_utils.to_device(data=batch["data"], device=cfg.device)
#             data_batch = {k: v.flatten(0, 1) for k, v in data_batch.items()}
#             data_batch = process_images(data_batch)

#             # avg_image = torch.mean(data_batch['image_m'])
#             # print(f"Epoch {_}, Batch {batch_idx}: Average Image Value = {avg_image.item()}")

#             with torch.amp.autocast('cuda', enabled=cfg.train.use_amp):
#                 outputs = model(data=data_batch, iter_percentage=counter.curr_percent)

#             true_boxes = data_batch["boxes"]
#             pred_boxes = outputs["pred_boxes"]
#             loss = box_loss_fn(pred_boxes, true_boxes)
#             loss_values.append(loss.item())
#                 # Обновление графика и сохранение в файл
#             # update_loss_plot(loss_values)
#             # loss = loss / cfg.train.grad_acc_step
#             scaler.calculate_grad(loss=loss)
#             print('===loss===', loss)
#             if counter.every_n_iters(cfg.train.grad_acc_step):  # Accumulates scaled gradients.
#                 scaler.update_grad()

#             item_loss = loss.item()
#             data_shape = tuple(data_batch["boxes"].shape)
#             loss_recorder.update(value=item_loss, num=data_shape[0])

#             if counter.is_last_total_iter():
#                 break
#             counter.update_iter_counter()

#         # Сохранение весов после эпохи
#         epoch_avg_loss = loss_recorder.global_avg  # Средний loss за эпоху
#         print(f"Epoch {counter.curr_epoch}: Average Loss = {epoch_avg_loss:.6f} =================================================")

#         # Сохраняем "последние" веса и параметры оптимизатора в один файл
#         last_weights_path = os.path.join("weights", "last.pth")
#         os.makedirs("weights", exist_ok=True)  # Создаём директорию weights при необходимости

#         # Создаем словарь для сохранения
#         last_weights = {
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#         }

#         # Сохраняем словарь в файл
#         torch.save(last_weights, last_weights_path)
#         # print(f"Saved last weights and optimizer state to {last_weights_path}")

#         # Проверяем и сохраняем лучшие веса в один файл
#         if epoch_avg_loss < best_loss:
#             best_loss = epoch_avg_loss
#             best_weights_path = os.path.join("weights", "best.pth")
            
#             # Создаем словарь для лучших весов
#             best_weights = {
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }
            
#             # Сохраняем словарь в файл
#             torch.save(best_weights, best_weights_path)
#             # print(f"New best loss: {best_loss:.6f}. Saved best weights and optimizer state to {best_weights_path}")

#         counter.update_epoch_counter()
def train(model, cfg):
    # Инициализация датасета и загрузчика
    tr_dataset = VideoDataset(
        dataset_root=cfg.train.data.dataset_infos.ZoomNext_dataset.root,
        shape=cfg.train.data.shape,
    )
    loader = data.DataLoader(
        dataset=tr_dataset,
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
    # scaler = GradScaler()
    # iter_time_recorder = recorder.HistoryBuffer()

    if cfg.train.bn.freeze_encoder:
        model.encoder.requires_grad_(False)

    torch.autograd.set_detect_anomaly(True)  # Включаем обнаружение аномалий

    for _ in range(counter.num_epochs):
        model.train()
        if cfg.train.bn.freeze_status:
            pt_utils.frozen_bn_stats(model.encoder, freeze_affine=cfg.train.bn.freeze_affine)

        for batch_idx, batch in enumerate(loader):
            # iter_start_time = time.perf_counter()
            scheduler.step(curr_idx=counter.curr_iter)  # Обновление learning rate
            print('Keys of batch data', batch['data'].keys())

            data_to_move = {
                "image_s": batch['data']["image_s"],
                "image_m": batch['data']["image_m"],
                "image_l": batch['data']["image_l"],
                "boxes": batch['data']["boxes"]
            }
                      
            data_batch = pt_utils.to_device(data=data_to_move, device=cfg.device)
            data_batch = {k: v.flatten(0, 1) for k, v in data_batch.items()}
            data_batch = process_images(data_batch)
            print('Keys of batch data', data_batch.keys())

            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp):
                outputs = model(data=data_batch, iter_percentage=counter.curr_percent)

            # print('===outputs===', outputs.keys())
            # for key, value in outputs.items():
            #     if torch.isnan(value).any():
            #         print(f"Outputs[{key}] contains NaN values.")
            #     if torch.isinf(value).any():
            #         print(f"Outputs[{key}] contains Inf values.")
            #     if (value == 0).any():
            #         print(f"Outputs[{key}] contains 0 values.")

            true_boxes = data_batch["boxes"]
            pred_boxes = outputs["pred_boxes"]
            loss = box_loss_fn(pred_boxes, true_boxes)
            scaler.calculate_grad(loss=loss)
            # Проверка градиентов
            for param in model.parameters():
                if param.grad is not None:
                    print(f"Gradient for {param}: {param.grad}")
                # else:
                    # print("No gradient for this parameter.")
            scaler.scaler._per_optimizer_states[id(optimizer)]['found_inf_per_device'] = {"device_0": False}
            print('===optimizer_state_in_train===', scaler.scaler._per_optimizer_states[id(optimizer)])
            scaler.update_grad()

            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        # Конец эпохи
        recorder.plot_results(
            dict(img=data_batch["image_m"], msk=data_batch["mask"], **outputs["vis"]),
            save_path=os.path.join(cfg.path.pth_log, "img", f"epoch_{counter.curr_epoch}.png"),
        )
        io.save_weight(model=model, save_path=cfg.path.final_state_net)
        counter.update_epoch_counter()

    cfg.tb_logger.close_tb()
    io.save_weight(model=model, save_path=cfg.path.final_state_net)

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