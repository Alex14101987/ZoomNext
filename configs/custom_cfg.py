num_frames = 5
__BATCHSIZE = 1
has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358

__NUM_EPOCHS = 3
_SHAPE = dict(h=384, w=384)
# __NUM_TR_SAMPLES = 3040 + 1000
# __ITER_PER_EPOCH = __NUM_TR_SAMPLES // __BATCHSIZE
# __NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH
train = dict(
    batch_size=__BATCHSIZE,
    num_workers=1,
    use_amp=False,
    num_epochs=__NUM_EPOCHS,
    lr=0.0001,
    epoch_based=True,
    num_iters=None,
    grad_acc_step=1,
    sche_usebatch=True,
    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(
            weight_decay=0,
            diff_factor=0.1,
        ),
    ),
    scheduler=dict(
        warmup=dict(num_iters=0),
        mode="constant",
        cfg=dict(coef=1),
    ),
    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=True,
    ),
    data=dict(
        shape=_SHAPE,
        names="ZoomNext_dataset_xywh",
        dataset_infos=dict(
            ZoomNext_dataset=dict(
                root="ZoomNext_dataset_xywh/Train",  # Путь к папке с изображениями и аннотациями
                box=dict(
                    path="ZoomNext_dataset_xywh/Train"  # Путь к папке с изображениями и аннотациями
                ),
            ),
        ),
    ),
)
test = dict(
    batch_size=__BATCHSIZE,
    num_frames=num_frames,
    num_workers=2,
    clip_range=None,

    data=dict(
        shape=_SHAPE,
        names=["ZoomNext_dataset_xywh/Test"],
        save_results=False,
    ),
)

