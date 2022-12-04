import yaml
import os 
import argparse


with open('./config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    
epoch = config['epoch']
lr_rate = config['lr_rate']
size = config['test_size']
dir_name = config['dir_name']
set_thres = config['set_thres']
print(epoch, lr_rate, size, dir_name, set_thres)

parser = argparse.ArgumentParser()
parser.add_argument("--seer", type=int, default=0)
args = parser.parse_args()

print('seer : ', args.seer)