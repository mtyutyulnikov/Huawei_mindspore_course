import os
import argparse
from mindspore import context

from mindspore.train.callback import SummaryCollector
from mindspore.train.callback import TimeMonitor

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

from mindspore.profiler import Profiler

from mindspore.context import ParallelMode

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # Define the dataset.
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # Define the mapping to be operated.
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # Use the map function to apply data operations to the dataset.
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # Perform shuffle and batch operations.
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    """
    Lenet network structure
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # Define the required operation.
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # Use the defined operation to construct a forward network.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Import the library required for model training.
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
 

def train_net(args, model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define a training method."""
    # Load the training dataset.
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[TimeMonitor(), LossMonitor(125), summary_collector], dataset_sink_mode=False)
#    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125), summary_collector], dataset_sink_mode=False)




def test_net(network, model, data_path):
    """Define a validation method."""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
#    acc = model.eval(ds_eval, callbacks=[summary_collector], dataset_sink_mode=False)
    acc = model.eval(ds_eval, dataset_sink_mode=False)


    print("{}".format(acc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    args = parser.parse_known_args()[0]
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    profiler = Profiler(output_path='./summary_dir/profiler_data')

    #context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)


    # Instantiate the network.
    net = LeNet5()


    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)


    # Set model saving parameters.
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # Use model saving parameters.
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    # Initialize a SummaryCollector callback instance, and use it in model.train or model.eval
    summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)




    train_epoch = 5
    mnist_path = "./datasets/MNIST_Data"
    dataset_size = 1
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
#    model = Model(net, net_loss, net_opt, amp_level="O2", loss_scale_manager=None)


    train_net(args, model, train_epoch, mnist_path, dataset_size, ckpoint, False)
#    test_net(net, model, mnist_path)

    profiler.analyse()
