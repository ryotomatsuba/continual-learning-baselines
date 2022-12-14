import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MLP
from experiments.utils import set_seed, create_default_args


def lwf_smnist(override_args=None):
    """
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this experiment is based on
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf

    To reproduce the results of the paper it is needed to add a LwF penalization which grows over time and
    to diminish over time the cross-entropy contribution. Since this is not a LwF dependant choice,
    we chose to not modify the cross-entropy loss. Adding this part should close the gap.
    """
    args = create_default_args({'cuda': 0, 'lwf_alpha': [0, 0.5, 0.66, 0.75, 0.8],
                                'lwf_temperature': 1, 'epochs': 20,
                                'layers': 2, 'hidden_size': 400,
                                'learning_rate': 0.001, 'train_mb_size': 128, 'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)
    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.layers)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.LwF(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        alpha=args.lwf_alpha, temperature=args.lwf_temperature,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = lwf_smnist()
    print(res)
