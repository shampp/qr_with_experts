import numpy as np
import time
import datetime as dt
from experiments import *
from data import dataset

data_dir = '../Data/'

def main():
    for dt in dataset:
        #df = preprocess(dt)
        #tokenize(df,dt)
        run_bandit_round(dt)
        #run_bandit_arm(dt)
        #run_bandit_round(dt)
        

if __name__ == '__main__':
    main()
