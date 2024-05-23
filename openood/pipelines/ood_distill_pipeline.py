import numpy as np
import torch

import openood.utils.comm as comm
from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer2
from openood.utils import setup_logger
from openood.datasets.utils import get_dataloader2
from openood.networks.knn import generate_outliers


class OODDistillPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']
        
        # get vit dataloader
        loader_dict2 = get_dataloader2(self.config)
        train_loader2, _ = loader_dict2['train'], loader_dict2['val']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer2(net, train_loader, train_loader2, val_loader, self.config)
        evaluator = get_evaluator(self.config)
        
        # # teacher model trainer
        # trainer = get_trainer2()

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            if self.config.trainer.name == 'mos':
                net, train_metrics, num_groups, group_slices = \
                    trainer.train_epoch(epoch_idx)
                val_metrics = evaluator.eval_acc(net,
                                                 val_loader,
                                                 train_loader,
                                                 epoch_idx,
                                                 num_groups=num_groups,
                                                 group_slices=group_slices)
            elif self.config.trainer.name in ['cider', 'npos']:
                net, train_metrics = trainer.train_epoch(epoch_idx)
                # cider and npos only trains the backbone
                # cannot evaluate ID acc without training the fc layer
                val_metrics = train_metrics
            else:
                net, train_metrics = trainer.train_epoch(epoch_idx)
                val_metrics = evaluator.eval_acc(net, val_loader, None,
                                                 epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc(net, test_loader)

        if comm.is_main_process():
            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)
