import yaml
from addict import Dict
import click
import os
import os.path as osp

@click.command()
@click.option('--model', '-m', type=str, default='inception_resnet_v2')
@click.option('--gpu', '-g', type=str, default=0)
def main(model, gpu):
    config = Dict(yaml.load(open(osp.join('extra_files', 'config.yaml'))))
    model_config = config.model[model]

    if not osp.exists(osp.join(config.train.root, model_config.train_dir)):
        os.makedirs(osp.join(config.train.root, model_config.train_dir))

    print('train {} on gpu {} ...'.format(model, gpu))
    print('check log file on {}'.format(osp.join(config.train.root, model_config.train_dir, 'train.log')))
    command = 'CUDA_VISIBLE_DEVICES={} ~/opt/anaconda3/envs/kaggle/bin/python train_image_classifier.py \
                --train_dir={} \
                --dataset_name=furniture \
                --dataset_split_name=train \
                --dataset_dir={} \
                --model_name={} \
                --checkpoint_path={} \
                --checkpoint_exclude_scopes={} \
                > {}.log 2>&1'.format(
                                    gpu,
                                    osp.join(config.train.root, model_config.train_dir),
                                    config.train.dataset_dir,
                                    model_config.model_name,
                                    model_config.checkpoint_path,
                                    model_config.checkpoint_exclude_scopes,
                                    osp.join(config.train.root, model_config.train_dir, 'train'),
                                    )
    os.system(command)

if __name__ == '__main__':
    main()
