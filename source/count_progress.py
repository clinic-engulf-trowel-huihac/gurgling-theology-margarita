#!/usr/bin/env python
import os
import argparse
import pandas as pd


def main():
    # parse one argument that is the experiment name
    parser = argparse.ArgumentParser(description='Count the progress of the experiment')
    parser.add_argument('experiment_name', type=str, help='The name of the experiment')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    experiment_dir = f'results/Experiment_{experiment_name}'

    files = os.listdir(experiment_dir)
    print(f'Found {len(files)} instances of Experiment {experiment_name}')
    for file in files:
        print(f'    {file.ljust(32)}', end='')
        for f in os.listdir(f'{experiment_dir}/{file}'):
            if f.endswith('.csv'):
                df = pd.read_csv(f'{experiment_dir}/{file}/{f}')
                df_done = df[df['DONE'] == True]
                has_data = df[df['Energy'] > 0]
                print(f'|  Progress: {len(df_done)}/{len(df)}    {len(df_done)/len(df)*100:.2f}%     (Empty: {len(df_done) - len(has_data)})')


if __name__ == '__main__':
    main()