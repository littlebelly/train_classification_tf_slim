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
    config = Dict(yaml.load(open(osp.join('extra_files', 'config.yaml'))))
    model_list = config.model
    flag = True
    ckpt_list = {}
    for model in model_list:
        model_name = model_list[model]['model_name']
        ckpt_list[model_name] = list()

    while flag:
        for model_t in model_list:
            model = model_list[model_t]['model_name']
            eval_path = osp.join(config.eval.root, model)
            ckpt_list_new = glob.glob(osp.join(eval_path, '*.meta'))
            ckpt_list_new = [f[:-5] for f in ckpt_list_new]
            ckpt_list_diff = [a for a in ckpt_list_new if a not in ckpt_list[model]]

            if ckpt_list_diff is None:
                pass
            else:
                for ckpt in ckpt_list_diff:
                    iteration = re.findall(r"\d+\.?\d*", ckpt)
                    iteration = iteration[-1]

                    test_path = osp.join(config.test.root, model)
                    if not os.path.exists(test_path):
                        os.makedirs(test_path)

                    ckpt_list[model].append(ckpt)
                    print('Testing ...')
                    print(ckpt)
                    command = 'CUDA_VISIBLE_DEVICES={} python -u ex_test_image_classifier.py \
                                --test_list={} \
                                --test_dir={} \
                                --model_name={} \
                                --checkpoint_path={} \
                                --output_pred_file={}/{}-{}.txt \
                                --batch_size=1 \
                                > _test_tmp.log 2>&1'.format(
                                    gpu,
                                    config.test.list,
                                    config.test.dataset_dir,
                                    model,
                                    ckpt,
                                    test_path, model, iteration
                                )
                    os.system(command)

if __name__ == '__main__':
    main()
