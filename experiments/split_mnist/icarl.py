import avalanche as avl

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import SGD
from torchvision import transforms

from experiments.utils import set_seed, create_default_args

from avalanche.benchmarks import SplitCIFAR100
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import *
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training import ICaRL

from openTSNE import TSNE
import plot_utills as utils
import matplotlib.pyplot as plt

def icarl_smnist_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode='constant')
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+28), crop[1]:(crop[1]+28)]
    else:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+28), crop[1]:(crop[1]+28)][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t


def icarl_smnist(override_args=None):
    """
    "iCaRL: Incremental Classifier and Representation Learning",
    Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert;
    Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2001-2010
    https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
    """
    args = create_default_args({'cuda': 0, 'batch_size': 128, 'nb_exp': 10,
                                'memory_size': 200, 'epochs': 30, 'lr_base': 2.,
                                'lr_milestones': [49, 63], 'lr_factor': 5.,
                                'wght_decay': 0.00001, 'train_mb_size': 256,
                                'seed': 2222}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loggers=[interactive_logger])

    # _____________________________Strategy
    model: IcarlNet = make_icarl_net(num_classes=10,n=5,c=1)
    model.apply(initialize_icarl_net)

    optim = SGD(model.parameters(), lr=args.lr_base,
                weight_decay=args.wght_decay, momentum=0.9)
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, args.lr_milestones, gamma=1.0 / args.lr_factor))

    strategy = ICaRL(
        model.feature_extractor, model.classifier, optim,
        args.memory_size,
        buffer_transform=transforms.Compose([icarl_smnist_augment_data]),
        fixed_memory=True, train_mb_size=args.batch_size,
        train_epochs=args.epochs, eval_mb_size=args.batch_size,
        plugins=[sched], device=device, evaluator=eval_plugin
    )
    # Dict to iCaRL Evaluation Protocol: Average Incremental Accuracy
    dict_iCaRL_aia = {}
    # ___________________________________________train and eval
    for i, experience in enumerate(benchmark.train_stream):
        strategy.train(experience, num_workers=4)
        res = strategy.eval(benchmark.test_stream[:i + 1], num_workers=4)
        dict_iCaRL_aia['Top1_Acc_Stream/Exp'+str(i)] = res['Top1_Acc_Stream/eval_phase/test_stream/Task000']
        # data for plotting
        output_dims = 64
        plot_data={'feature': np.empty((0,output_dims)), 'label': []}
        for i in range(len(benchmark.original_test_dataset)//30):
            feature=strategy.model.feature_extractor(benchmark.original_test_dataset[i][0].reshape(1,1,28,28).to('cuda'))# 特徴量を取得
            plot_data['feature']=np.append(plot_data['feature'],feature.detach().cpu().numpy(),axis=0)
            plot_data['label'].append(benchmark.original_test_dataset[i][1])
        tsne = TSNE()
        embedding= tsne.fit(plot_data['feature'])
        utils.plot(embedding, plot_data['label'],s=20)
        plt.savefig(f'weights/icarl/mem{args.memory_size}_exp{experience.current_experience}.png')


    return dict_iCaRL_aia


if __name__ == '__main__':
    res = icarl_smnist({'memory_size': 10})
    res = icarl_smnist({'memory_size': 100})
    res = icarl_smnist({'memory_size': 1000})
    print(res)
