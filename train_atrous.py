import os
import multiprocessing
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer, gradproc


try:
    from .cfgs.config import cfg
    from .reader import Data
except Exception:
    from cfgs.config import cfg
    from reader import Data

class Model(ModelDesc):

    def __init__(self, mode='train'):
        self.is_train = mode == 'train'

    def _get_inputs(self):
        return [
            InputDesc(tf.float32, (None, cfg.img_size, cfg.img_size, 1), 'img_input'),
            InputDesc(tf.float32, (None, cfg.img_size, cfg.img_size, 1), 'img_output')
        ]

    def _build_graph(self, inputs):
        img_input, img_output = inputs

        img_input = img_input * (1.0 / 255)
        img_output = img_output * (1.0 / 255)

        img_input = tf.identity(img_input, 'network_input')

        with argscope([Conv2D], kernel_shape=3, nl=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer()):
            img_pred = (LinearWrap(img_input)
                       .Conv2D('en_conv1', 64)
                       # .MaxPooling('en_pool1', 2, padding="SAME")
                       .Conv2D('en_conv2', 128, dilation_rate=2)
                       # .MaxPooling('en_pool2', 2, padding="SAME")
                       .Conv2D('en_conv3', 256, dilation_rate=4)
                       # .MaxPooling('en_pool3', 2, padding="SAME")
#                        .Conv2D('en_conv4', 512)
#                        .Conv2D('en_conv5', 512)
#                        .Conv2D('en_conv6', 512)
#                        .Conv2D('de_conv6', 512)
#                        .Conv2D('de_conv5', 512)
#                        .Conv2D('de_conv4', 256)
                       # .tf.image.resize_images((cfg.img_size // 4, cfg.img_size // 4))
                       .Conv2D('de_conv3', 128, dilation_rate=8)
                       # .tf.image.resize_images((cfg.img_size // 2, cfg.img_size // 2))
                       .Conv2D('de_conv2', 64, dilation_rate=16)
                       # .tf.image.resize_images((cfg.img_size, cfg.img_size))
                       .Conv2D('de_conv1', 1)())

        img_pred = tf.identity(img_pred, 'img_pred')

        diff = img_pred - img_output

        loss = tf.nn.l2_loss(diff) / (cfg.img_size ** 2)
        loss = tf.identity(loss, name='loss')

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        else:
            wd_cost = tf.constant(0.0)

        img_show = tf.concat([img_input, img_output, img_pred], axis=2)
        tf.summary.image('img-show', img_show, max_outputs=3)

        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')
    

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-3, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(train_or_test, batch_size):
    is_train = train_or_test == 'train'

    filename_list = cfg.train_list if is_train else cfg.test_list
    ds = Data(filename_list, rotate=False, flip_ver=is_train, flip_horiz=is_train, shuffle=is_train)

    sample_num = ds.size()

    augmentors = [
        # random rotate and flip should be applied to both input and label, thus cannot be added here
        imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
        imgaug.ToUint8()
    ]
    ds = AugmentImageComponent(ds, augmentors)

    if is_train:
        ds = PrefetchDataZMQ(ds, min(8, multiprocessing.cpu_count()))
    ds = BatchData(ds, batch_size, remainder = not is_train)
    return ds, sample_num

def get_config(args, model):
    ds_train, sample_num = get_data('train', args.batch_size_per_gpu)
    ds_val, _ = get_data('test', args.batch_size_per_gpu)

    return TrainConfig(
        dataflow = ds_train,
        callbacks = [
            ModelSaver(),
            PeriodicTrigger(InferenceRunner(ds_val, [ScalarStats('cost')]),
                            every_k_epochs=5),
            HumanHyperParamSetter('learning_rate'),
        ],
        model = model,
        steps_per_epoch = sample_num // (args.batch_size_per_gpu * get_nr_gpu()),
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size_per_gpu', type=int, default=16)
    parser.add_argument('--logdir', help="directory of logging", default=None)
    parser.add_argument('--flops', action="store_true", help="print flops and exit")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()
    if args.flops:

        input_desc = [
            InputDesc(tf.float32, (1, cfg.img_size, cfg.img_size, 3), 'imgs'),
            InputDesc(tf.float32, (1, cfg.img_size, cfg.img_size, 3), 'mask')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        if args.logdir != None:
            logger.set_logger_dir(os.path.join("train_log", args.logdir))
        else:
            logger.auto_set_dir()

        config = get_config(args, model)
        if args.load:
            config.session_init = get_model_loader(args.load)
        
        trainer = SyncMultiGPUTrainerParameterServer(get_nr_gpu())
        launch_train_with_config(config, trainer)
