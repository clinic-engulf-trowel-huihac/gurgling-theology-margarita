#!/usr/bin/env python3

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "data/models/huggingface"
os.environ['TORCH_HOME'] = 'data/models/torch'
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import argparse
from datetime import datetime
import time
import csv

import numpy as np
from scipy import stats
import pandas as pd

from powermeter import powermeter
from model_prep import test_set, bs_gen
from result_processing import process_results

def experiment(args):
    exp_start_time = time.time()

    meter = powermeter('Experiment_' + args.experiment, gpu_id=args.id, force_node_name=args.force)
    log(meter.log_file, meter.general_info())
    results_pth = prep_model_db(meter)

    model = test_set(args, meter.gpu_name, results_pth, gpu_id=args.id, test_length=10)
    log(meter.log_file, 'Models to be tested:')
    for index, row in model.test_db.iterrows():
        log(meter.log_file, f'  {row["Model Name"]}')

    for _ in model:
        begin_time = time.time()
        log(meter.log_file, f'\n\n--- {model.full_name} ---')
        if not args.plot_only:
            if model.done: 
                log(meter.log_file, '  Model already tested. Skipping ...')
                continue
            os.system('nvidia-smi')
            model.update_model_db('DONE', True)
            for rep in range(int(args.repetitions)):
                log(meter.log_file, f'    Rep {rep+1} of {args.repetitions}:')
                # Perform experiment
                meter.prep_new_model(model.full_name, rep+1, [str(model.device.index)])
                batchsize = get_bs(meter, model)
                run_experiment(meter, model, batchsize, rep)
        
        process_results(meter, model)
        model.clean_up()

        time_taken = time.time() - begin_time
        log(meter.log_file, f'\n  Model done in {int(time_taken//60)} m {time_taken%60:.2f} s')
    time_taken = time.time() - exp_start_time
    log(meter.log_file, f'\n\nExperiment done in {int(time_taken//3600)} h {int(time_taken%3600//60)} m {time_taken%60:.2f} s')

def prep_model_db(meter):
    results_pth = os.path.join(meter.exp_path, f'{meter.gpu_name_short}_results.csv')
    if not os.path.exists(results_pth):
        df = pd.read_csv('source/model_db.csv')
        new_columns = ['Max BS', 'Optimal BS', 'Throughput', 'Latency', 'Energy', 'DONE']
        df = df.assign(**{col: np.nan for col in new_columns})
        df.to_csv(results_pth, index=False)

    return results_pth

def run_experiment(meter, model, batchsize, rep):        
    while True:
        next(iter(batchsize))
        log(meter.log_file, f'            Batchsize {str(batchsize)+":":<8}', end='')
        start_time = datetime.now()
        # Check if batchsize fits in GPU
        try:
            model.reload()
            if model.use_tensorrt:  model.build_trt_engine(batchsize)
            model.infer(batchsize)
        except RuntimeError as e:
            print('--- CAUGHT EXCEPTION ---', flush=True)
            print(e, flush=True)
            print('--- END OF EXCEPTION ---', flush=True)
            if 'torch_tensorrt' in str(e) and 'AssertionError' in str(e):  log(meter.log_file, 'TRT build failed', end='')
            elif 'TensorRT execution context' in str(e):                   log(meter.log_file, 'TRT execution context failed', end='')
            elif 'ValueError' in str(e):                                   log(meter.log_file, 'TRT value error', end='')
            elif 'CUDA out of memory' in str(e):                           log(meter.log_file, 'GPU out of memory', end='')
            elif 'Expected canUse32BitIndexMath' in str(e):                log(meter.log_file, 'Expected canUse32BitIndexMath', end='')
            elif 'invalid configuration argument' in str(e):               log(meter.log_file, 'Invalid configuration argument', end='')
            elif 'torch.eye' in str(e):
                log(meter.log_file, str(e))
                return
            else:  log(meter.log_file, str(e))
            break
        else:
            meter.prep_new_measurement(batchsize, rep+1, model.forward_pass) 
            meter.start_measurement()
            timestamps = model.infer(batchsize)
            if not meter.end_measurement(timestamps):
                log(meter.log_file, 'nvdia-smi failed, rerun the current test... ', end='')
                meter.start_measurement()
                timestamps = model.infer(batchsize)
                meter.end_measurement(timestamps)
        time_diff = (datetime.now() - start_time).total_seconds()
        log(meter.log_file, f" --Done in {time_diff:.2f} s")

    last_OOM_bs = int(batchsize)
    batchsize = batchsize.step_back()
    step = int(batchsize/4)
    granularity = 2
    if model.use_tensorrt:  granularity = int(2 ** max(1, np.ceil(np.log2(batchsize)) - 4))
    log(meter.log_file, f'        Granularity: {granularity}')
    while step >= granularity:
        batchsize += step
        if int(batchsize) == last_OOM_bs:
            batchsize -= step
            step = int(step/2)
            continue

        log(meter.log_file, f'            Batchsize {str(batchsize)+":":<8}', end='')
        start_time = datetime.now()
        try:
            model.reload()
            if model.use_tensorrt:  model.build_trt_engine(batchsize)
            model.infer(batchsize)

        except RuntimeError as e:
            print('--- CAUGHT EXCEPTION ---', flush=True)
            print(e, flush=True)
            print('--- END OF EXCEPTION ---', flush=True)
            if 'torch_tensorrt' in str(e) and 'AssertionError' in str(e):  log(meter.log_file, 'TRT build failed', end='')
            elif 'TensorRT execution context' in str(e):                   log(meter.log_file, 'TRT execution context failed', end='')
            elif 'ValueError' in str(e):                                   log(meter.log_file, 'TRT value error', end='')
            elif 'CUDA out of memory' in str(e):                           log(meter.log_file, 'GPU out of memory', end='')
            elif 'Expected canUse32BitIndexMath' in str(e):                log(meter.log_file, 'Expected canUse32BitIndexMath', end='')
            elif 'invalid configuration argument' in str(e):               log(meter.log_file, 'Invalid configuration argument', end='')
            elif 'torch.eye' in str(e):
                log(meter.log_file, str(e))
                return
            else:  log(meter.log_file, str(e))

            last_OOM_bs = int(batchsize)
            batchsize -= step
            step = int(step/2)
        else:
            meter.prep_new_measurement(batchsize, rep+1, model.forward_pass)
            meter.start_measurement()
            timestamps = model.infer(batchsize)
            if not meter.end_measurement(timestamps):
                log(meter.log_file, 'nvdia-smi failed, rerun the current test... ', end='')
                meter.start_measurement()
                timestamps = model.infer(batchsize)
                meter.end_measurement(timestamps)

        time_diff = (datetime.now() - start_time).total_seconds()
        log(meter.log_file, f" --Done in {time_diff:.2f} s")

def get_bs(meter, model):
    if model.use_tensorrt:  return bs_gen('exp', 1)

    if np.isnan(model.max_bs):
        begin_time = time.time()
        log(meter.log_file, '        Max batch size not found. Finding max batch size ...\n       ', end='')
        batchsize = bs_gen('exp', 1)

        while True:
            next(iter(batchsize))
            log(meter.log_file, f' {str(batchsize)}', end='')
            try:
                model.reload()
                if model.use_tensorrt:  model.build_trt_engine(batchsize)
                model.infer(batchsize)
            except (RuntimeError, AssertionError) as e:
                if isinstance(e, AssertionError):                  log(meter.log_file, '-TRT', end='')
                elif 'CUDA out of memory' in str(e):               log(meter.log_file, '-OOM', end='')
                elif 'Expected canUse32BitIndexMath' in str(e):    log(meter.log_file, '-32b', end='')
                elif 'invalid configuration argument' in str(e):   log(meter.log_file, '-ICA', end='')
                else:                                              raise e
                break
            
        batchsize = batchsize.step_back()

        step = int(batchsize/4)
        while step >= 2:
            batchsize += step
            log(meter.log_file, f' {str(batchsize)}', end='')
            try:
                model.reload()
                if model.use_tensorrt:  model.build_trt_engine(batchsize)
                model.infer(batchsize)
            except (RuntimeError, AssertionError) as e:
                if isinstance(e, AssertionError):                  log(meter.log_file, '-TRT', end='')
                elif 'CUDA out of memory' in str(e):               log(meter.log_file, '-OOM', end='')
                elif 'Expected canUse32BitIndexMath' in str(e):    log(meter.log_file, '-32b', end='')
                elif 'invalid configuration argument' in str(e):   log(meter.log_file, '-ICA', end='')
                else:                                              raise e
                batchsize -= step
                step = int(step/2)
        
        log(meter.log_file, f'\n        Max batch size found: {batchsize}')
        model.reload()

        time_taken = time.time() - begin_time
        log(meter.log_file, f'        Max batch size found in {int(time_taken//60)} m {time_taken%60:.2f} s')

        model.max_bs = batchsize
        model.update_model_db('Max BS', int(batchsize))
        log(meter.log_file, f'      Experiment resumes:')
    
    log(meter.log_file, f'        Max batch size: {model.max_bs}')
    max_bs = model.max_bs
    step = 2 ** int(np.log2(max_bs/8))
    if step < 1:    step = 1
    log(meter.log_file, f'        Batch size step: {step}')
    log(meter.log_file, '        Experiment starts:')
    return bs_gen('lin', step)

def estimate_bs(meter, model):
    model_stats = pd.read_csv('source/model_stats.csv')
    model_stats = model_stats[(model_stats['model_name'] == model.model_name) & (model_stats['input_size_csv'] == model.input_size)].iloc[0]
    estimated_bs = (meter.gpu_mem_size * 1024 * 1024 - model_stats['params_tinfo'] * 2) / (model_stats['activations'] * 2)
    log2_n = np.log2(estimated_bs)
    lower_exp = int(np.floor(log2_n))
    upper_exp = lower_exp + 1

    lower_bs = 2 ** lower_exp
    upper_bs = 2 ** upper_exp

    if abs(lower_bs - estimated_bs) <= abs(upper_bs - estimated_bs):
        print(f'Estimated BS: {estimated_bs}, Closest BS: {lower_bs}, Closest Exp: {lower_exp}')
        return lower_exp
    else:
        print(f'Estimated BS: {estimated_bs}, Closest BS: {upper_bs}, Closest Exp: {upper_exp}')
        return upper_exp

def log(pth, msg, end='\n'):
    print(msg, end=end, flush=True)
    with open(pth, 'a') as f:
        f.write(f'{msg}{end}')

def main():
    parser = argparse.ArgumentParser(description="DL Energy Experiment CLI")

    parser.add_argument('-e', '--experiment', default='test', help='Select experiment type')
    parser.add_argument('--type', help='Test all models of the same kind')
    parser.add_argument('--family', help='Test all models of the same family')
    parser.add_argument('--model', help='Test a specific model')
    parser.add_argument('-r', '--repetitions', default=3, help='Experiment repetitions')
    parser.add_argument('--id', default='0', help='GPU ID')
    parser.add_argument('-p', '--plot_only', action='store_true', help='Plot results only')
    parser.add_argument('-f', '--force', default=None, help='Force the node name for plotting data not run on the same node')

    args = parser.parse_args()
    experiment(args)


if __name__ == "__main__":
    main()