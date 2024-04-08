
import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()


def initialization(cfgs, training):
    # cfgs --  {'dataset_name': 'CASIA-B', 'dataset_root': 'CASIA-B-pkl', 'num_workers': 1, 'dataset_partition': './datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}, 'evaluator_cfg': {'enable_float16': True, 'restore_ckpt_strict': True, 'restore_hint': 60000, 'save_name': 'DeepGaitV2', 'eval_func': 'evaluate_indoor_dataset', 'sampler': {'batch_size': 4, 'sample_type': 'all_ordered', 'type': 'InferenceSampler', 'batch_shuffle': False, 'frames_all_limit': 720}, 'transform': [{'type': 'BaseSilCuttingTransform'}], 'metric': 'euc', 'cross_view_gallery': False}, 'loss_cfg': [{'loss_term_weight': 1.0, 'margin': 0.2, 'type': 'TripletLoss', 'log_prefix': 'triplet'}, {'loss_term_weight': 1.0, 'scale': 16, 'type': 'CrossEntropyLoss', 'log_prefix': 'softmax', 'log_accuracy': True}], 'model_cfg': {'model': 'DeepGaitV2', 'Backbone': {'mode': 'p3d', 'in_channels': 1, 'layers': [1, 1, 1, 1], 'channels': [64, 128, 256, 512]}, 'SeparateBNNecks': {'class_num': 74}}, 'optimizer_cfg': {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}, 'scheduler_cfg': {'gamma': 0.1, 'milestones': [20000, 40000, 50000], 'scheduler': 'MultiStepLR'}, 'trainer_cfg': {'find_unused_parameters': False, 'enable_float16': True, 'with_test': False, 'fix_BN': False, 'log_iter': 100, 'restore_ckpt_strict': True, 'optimizer_reset': False, 'scheduler_reset': False, 'restore_hint': 0, 'save_iter': 30000, 'save_name': 'DeepGaitV2', 'sync_BN': True, 'total_iter': 60000, 'sampler': {'batch_shuffle': True, 'batch_size': [4, 8], 'frames_num_fixed': 30, 'frames_num_max': 50, 'frames_num_min': 25, 'sample_type': 'fixed_ordered', 'type': 'TripletSampler', 'frames_skip_num': 4}, 'transform': [{'type': 'Compose', 'trf_cfg': [{'type': 'RandomPerspective', 'prob': 0.2}, {'type': 'BaseSilCuttingTransform'}, {'type': 'RandomHorizontalFlip', 'prob': 0.2}, {'type': 'RandomRotate', 'prob': 0.2}]}]}}
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    # use the config loader over here...
    cfgs = config_loader(opt.cfgs)
    # cfs value -- {'dataset_name': 'CASIA-B', 'dataset_root': 'CASIA-B-pkl', 'num_workers': 1, 'dataset_partition': './datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}, 'evaluator_cfg': {'enable_float16': True, 'restore_ckpt_strict': True, 'restore_hint': 60000, 'save_name': 'DeepGaitV2', 'eval_func': 'evaluate_indoor_dataset', 'sampler': {'batch_size': 4, 'sample_type': 'all_ordered', 'type': 'InferenceSampler', 'batch_shuffle': False, 'frames_all_limit': 720}, 'transform': [{'type': 'BaseSilCuttingTransform'}], 'metric': 'euc', 'cross_view_gallery': False}, 'loss_cfg': [{'loss_term_weight': 1.0, 'margin': 0.2, 'type': 'TripletLoss', 'log_prefix': 'triplet'}, {'loss_term_weight': 1.0, 'scale': 16, 'type': 'CrossEntropyLoss', 'log_prefix': 'softmax', 'log_accuracy': True}], 'model_cfg': {'model': 'DeepGaitV2', 'Backbone': {'mode': 'p3d', 'in_channels': 1, 'layers': [1, 1, 1, 1], 'channels': [64, 128, 256, 512]}, 'SeparateBNNecks': {'class_num': 74}}, 'optimizer_cfg': {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}, 'scheduler_cfg': {'gamma': 0.1, 'milestones': [20000, 40000, 50000], 'scheduler': 'MultiStepLR'}, 'trainer_cfg': {'find_unused_parameters': False, 'enable_float16': True, 'with_test': False, 'fix_BN': False, 'log_iter': 100, 'restore_ckpt_strict': True, 'optimizer_reset': False, 'scheduler_reset': False, 'restore_hint': 0, 'save_iter': 30000, 'save_name': 'DeepGaitV2', 'sync_BN': True, 'total_iter': 60000, 'sampler': {'batch_shuffle': True, 'batch_size': [4, 8], 'frames_num_fixed': 30, 'frames_num_max': 50, 'frames_num_min': 25, 'sample_type': 'fixed_ordered', 'type': 'TripletSampler', 'frames_skip_num': 4}, 'transform': [{'type': 'Compose', 'trf_cfg': [{'type': 'RandomPerspective', 'prob': 0.2}, {'type': 'BaseSilCuttingTransform'}, {'type': 'RandomHorizontalFlip', 'prob': 0.2}, {'type': 'RandomRotate', 'prob': 0.2}]}]}}
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
