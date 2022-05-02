from mindspore import context
import numpy as np
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import nn
from mindspore.ops import TensorSummary
from mindspore import Tensor
from mindspore import Model
from mindspore.train.callback import SummaryCollector, TimeMonitor
from mindspore.profiler import Profiler
import os
import argparse


def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data


class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))
        self.tensor_summary = TensorSummary()

    def construct(self, x):
        x = self.fc(x)
        self.tensor_summary("tensor", x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LinReg Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    args = parser.parse_known_args()[0]
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    profiler = Profiler(output_path='./summary_dir/profiler_data')

    data_number = 1600
    batch_number = 16
    repeat_number = 1

    ds_train = create_dataset(data_number, batch_size=batch_number, repeat_size=repeat_number)
    dict_datasets = next(ds_train.create_dict_iterator())

    net = LinearNet()
    model_params = net.trainable_params()

    x_model_label = np.array([-10, 10, 0.1])
    y_model_label = (x_model_label * Tensor(model_params[0]).asnumpy()[0][0] +
                     Tensor(model_params[1]).asnumpy()[0])

    net = LinearNet()
    net_loss = nn.loss.MSELoss()
    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

    model = Model(net, net_loss, opt)

    epoch = 1
    summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)
    model.train(epoch, ds_train, callbacks=[TimeMonitor(), summary_collector], dataset_sink_mode=False)

    profiler.analyse()

