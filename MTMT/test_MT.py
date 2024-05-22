import os
import argparse
import torch
from test_MT_util import test_all_case
# from networks.EGNet import build_model
from networks.MTMT import build_model
# from networks.EGNet_onlyDSS import build_model
# from networks.EGNet_task3 import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='path of input data', help='path of detaset')
parser.add_argument('--model', type=str,  default='EGNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--base_lr', type=float,  default=0.005, help='base learning rate')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
# parser.add_argument('--epoch_name', type=str,  default='iter_10000.pth', help='choose one epoch/iter as pretained')
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--scale', type=int,  default=416, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--subitizing', type=float,  default=5.0, help='subitizing loss weight')
parser.add_argument('--repeat', type=int,  default=6, help='repeat')
FLAGS = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = '../weights/finetuned_mtmt_model.pth'
test_save_path = '../results/MTMT'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(snapshot_path)
num_classes = 1

img_list = [os.path.splitext(f)[0] for f in os.listdir(FLAGS.root_path) if f.endswith('.png')]
# gt_list = [os.path.join(FLAGS.gt, f) for f in os.listdir(FLAGS.gt) if f.endswith('.png')]
# data_path = [(os.path.join(FLAGS.root_path, 'ShadowImages', img_name + '.jpg'),
#              os.path.join(FLAGS.root_path, 'ShadowMasks', img_name + '.png'))
#             for img_name in img_list]
data_path = [(os.path.join(FLAGS.root_path, img_name + '.png'),
             os.path.join(FLAGS.root_path, img_name + '.png'),)# damy path（target_pathとして扱う）
            for img_name in img_list]


def test_calculate_metric():
    net = build_model('resnext101').cuda()
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=True, test_save_path=test_save_path, trans_scale=FLAGS.scale)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    with open('record/test_record_EGNet_meanteacher.txt', 'a') as f:
        f.write(snapshot_path+' ')
        f.write(str(metric)+' --UCF\r\n')
    print('Test ber results: {}'.format(metric))
