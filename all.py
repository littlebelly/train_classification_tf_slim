import os
import glob
import numpy as np
import re
import shutil
import glob
import pandas
import argparse
import time
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-n', '--num', type=int, required=True)
    parser.add_argument('-c', '--clean_up', type=bool, required=True)
    args = parser.parse_args()
    root = args.root
    num = args.num
    clean_up = args.clean_up

    if clean_up:
       old_files = glob.glob(os.path.join('extra_files', 'txt_list', '*.txt'))
       for old_file in old_files:
           os.remove(old_file)
    
    path_all = os.listdir(root)
    print('copying...')
    for path in path_all:
        path = os.path.join(root, path)
        ckpt_list = glob.glob(os.path.join(path, '*.txt'))
        ckpt_num_list = []
        for ckpt in ckpt_list:
            ckpt = re.findall(r"\d+", ckpt)
            ckpt = ckpt[-1]
            ckpt = int(ckpt)
            ckpt_num_list.append(ckpt)
        ckpt_num_list.sort(reverse=True)
        for i in range(1, num+1):
            ckpt_num = ckpt_num_list[i]
            _tmp_path = path + '/*' + str(ckpt_num) + '.txt'
            file_name = glob.glob(_tmp_path)
            file_name = file_name[0]
            new_file_name = os.path.join('extra_files', 'txt_list', file_name[len(path)+1:])
            shutil.copy(file_name, new_file_name)

    print('ensembling...')
    file_dict = {}
    num_cls = 128
    root_path = os.path.join('extra_files', 'txt_list')
    file_list = os.listdir(root_path)
    pbar = tqdm(total=len(file_list))
    for file_name in file_list:
        print(file_name)
        pbar.update(1)
        file = os.path.join(root_path,file_name)
        file_dict[file_name]={}
        with open(file, 'r') as fb:
            lines = fb.readlines()
            print(len(lines))
            for line in lines:
                records = line.strip().split()
                img_name = records[0]
                probs = []
                for i in range(num_cls):
                     probs.append(float(records[i+1]))
                file_dict[file_name][img_name] = probs
    pbar.close() 
    key_0 = list(file_dict.keys())
    final = {}
    ensemble = {}
    for k in file_dict[key_0[0]]:
        final[k] = 0
        for k1 in file_dict.keys():
            final[k] += np.array(file_dict[k1][k])
        final[k] /= len(file_dict.keys())
        indx = np.where(final[k] == np.max(final[k]))
        ensemble[k] = indx[0][0]
    
    cls_to_label = {}
    with open(os.path.join('extra_files','labels.txt'), 'r') as fb:
        lines = fb.readlines()
        for line in lines:
            record = line.strip().split(':')
            cls_idx = int(record[0])
            cls_label = record[1]
            cls_to_label[cls_idx] = cls_label

    localtime = time.localtime()
    time_sam = time.strftime("%Y-%m-%d_%H-%M-%S", localtime)
    
    fw = open(time_sam + '.csv', 'w')
    fw.write('id,predicted\n')
    for key in ensemble.keys():
        filename = key
        filelabel = cls_to_label[ensemble[key]]
        fw.write(filename[:-4]+','+filelabel+'\n')
    
    missing_list = open(os.path.join('extra_files','test_missing_list.txt'), 'r')
    for line in missing_list:
        line = line.strip()
        fw.write(line + '\n')
    
    fw.close()
    missing_list.close()
    print('Finished!')

if __name__ == '__main__':
    main()

