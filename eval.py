import os
import glob
import numpy as np
import shutil
import re
import click
from addict import Dict
import yaml
import os.path as osp

@click.command()
@click.option('--gpu', '-g', type=str, default=0)
def main(gpu):
    flag = True
    config = Dict(yaml.load(open(osp.join('extra_files', 'config.yaml'))))
    model_list = config.model
    ckpt_list = {}
    for model in model_list:
        model_name = model_list[model]['model_name']
        ckpt_list[model_name] = list()

    while flag:
        for model_t in model_list:
            model = model_list[model_t]['model_name']
            train_path = osp.join(config.train.root, model)
            ckpt_list_new = glob.glob(osp.join(train_path, '*.meta'))
            ckpt_list_new = [f[:-5] for f in ckpt_list_new]
            ckpt_list_diff = [a for a in ckpt_list_new if a not in ckpt_list[model]]

            if ckpt_list_diff is None:
                pass
            else:
                for ckpt in ckpt_list_diff:
                    iteration = re.findall(r"\d+\.?\d*", ckpt)
                    iteration = iteration[-1]
                    
                    eval_path = osp.join(config.eval.root, model)
                    if not os.path.exists(eval_path):
                        os.makedirs(eval_path)
                    for t in ['data-00000-of-00001', 'index', 'meta']:
                        shutil.copy(ckpt + '.' + t, osp.join(eval_path, 'model.ckpt-' + iteration + '.' + t))

                    ckpt_list[model].append(ckpt)
                    print('evaluating...')
                    print(ckpt)
                    command = 'CUDA_VISIBLE_DEVICES={} python -u eval_image_classifier.py \
                                --dataset_name={} \
                                --dataset_dir={} \
                                --dataset_split_name={} \
                                --model_name={} \
                                --checkpoint_path={} \
                                --eval_dir={} \
                                --batch_size=32 \
                                > _eval_tmp.log 2>&1'.format(
                                    gpu,
                                    config.eval.dataset_name,
                                    config.eval.dataset_dir,
                                    config.eval.dataset_split_name,
                                    model,
                                    ckpt,
                                    osp.join(config.eval.root, model),
                                )
                    os.system(command)

                    f = open('_eval_tmp.log', 'r')
                    lines = f.readlines()
                    for t in lines:
                        if 'eval/Accuracy' in t:
                            line = t[t.index("y")+2:]
                            line = line[:line.index("]")]
                    f.close()

                    f_l = open(osp.join(config.eval.root, model + '/eval.log'), 'a')
                    f_l.write('{} {} \n'.format(iteration, line))
                    f_l.close()
                    print('iteration {} current_acc {}'.format(iteration, line)) 

if __name__ == '__main__':
    main()