import os
import json
import time
import csv

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import multiprocessing as mp

import numpy as np
from scipy import stats
import pandas as pd

import matplotlib.pyplot as plt


def process_results(meter, model):
    begin_time = time.time()
    log(meter.log_file, f'\n    Processing results ...')  
    pallate = {
        'total': '#002147',
        'cpu': '#0071c5',
        'dram': '#12279e',
        'gpu': '#76b900',
        'throughput': 'blue',
        'latency': 'red',
        'cpu_util': 'brown',
        'dram_used': 'gray',
        'gpu_util': 'green',
        'vram_util': 'orange',
        'vram_used': 'purple',
        'gpu_temp': 'pink',

    }

    with open(os.path.join(meter.exp_path, 'hardware_info', 'summary.json'), 'r') as f:
        hw_summary = json.load(f)

    results_pth = os.path.join(meter.exp_path, 'results', model.full_name)
    # check if the results dir exists
    if not os.path.isdir(results_pth):
        log(meter.log_file, f'    No results found for {model.full_name}')
        return

    raw_pth = os.path.join(results_pth, 'raw_data')
    # print the dirs in the results dir
    bs_list = [d for d in os.listdir(raw_pth) if os.path.isdir(os.path.join(raw_pth, d))]

    if not bs_list:
        log(meter.log_file, f'    No batch sizes found for {model.full_name}')
        return
    
    bs_list = list(map(int, bs_list))
    bs_list.sort()
    bs_list = list(map(str, bs_list))

    input_tuples = [(raw_pth, hw_summary, pallate, model.full_name, bs) for bs in bs_list]
    nproc = mp.cpu_count()
    if nproc > len(input_tuples):
        nproc = len(input_tuples)

    ## Temporatily disable for quick plotting
    log(meter.log_file, f'    Processing {len(input_tuples)} batchsizes with {nproc} processes')
    with mp.Pool(nproc) as p:
        p.starmap(process_batch, input_tuples)
        p.close()

    process_model(results_pth, bs_list, model, pallate, hw_summary)


    time_taken = time.time() - begin_time
    log(meter.log_file, f'\n    Results processed in {time_taken:.2f} s')



def process_model(results_pth, bs_list, model, pallate, hw):
    header = ['Latency', 'Throughput', 'CPU Energy', 'DRAM Energy', 'GPU Energy', 'Total Energy',
                'CPU Util', 'DRAM Used', 'GPU Util', 'VRAM Util', 'VRAM Used', 'GPU Temp']

    results = {'mean': [header], 'std': [header], 'sem': [header], 'ci_lower': [header], 'ci_upper': [header]}

    for bs in bs_list:
        bs_pth = os.path.join(results_pth, 'raw_data', bs)
        df = pd.read_csv(os.path.join(bs_pth, 'batch_result_stats.csv'), index_col=0)
        for key in results.keys():    results[key].append(df.loc[key].to_list())

    for key in results.keys():    results[key] = pd.DataFrame(results[key], columns=results[key].pop(0))

    # print(results)
    # save the results as csv
    results['mean'].to_csv(os.path.join(results_pth, 'model_result.csv'), index=False)
    batch_sizes = np.array(list(map(int, bs_list)))
    # find the min gpu energy
    min_energy = results['mean']['GPU Energy'].min()
    max_energy = results['mean']['GPU Energy'].max()
    min_idx = results['mean']['GPU Energy'].idxmin()
    min_bs = batch_sizes[min_idx]

    # plotting the results
    plt.figure()
    fig, axis = plt.subplots(nrows=4, ncols=1)        

    # The first row - energy consumption of cpu, dram, gpu
    axis[0].plot(batch_sizes, results['mean']['GPU Energy']/1000, color=pallate['gpu'], marker='o')
    axis[0].fill_between(batch_sizes, results['ci_lower']['GPU Energy']/1000, results['ci_upper']['GPU Energy']/1000, color=pallate['gpu'], alpha=0.2)
    axis[0].plot(batch_sizes, results['mean']['CPU Energy']/1000, color=pallate['cpu'], marker='o')
    axis[0].fill_between(batch_sizes, results['ci_lower']['CPU Energy']/1000, results['ci_upper']['CPU Energy']/1000, color=pallate['cpu'], alpha=0.2)
    axis[0].plot(batch_sizes, results['mean']['DRAM Energy']/1000, color=pallate['dram'], marker='o')
    axis[0].fill_between(batch_sizes, results['ci_lower']['DRAM Energy']/1000, results['ci_upper']['DRAM Energy']/1000, color=pallate['dram'], alpha=0.2)

    axis[0].set_ylim([0,None])
    label_plot(axis[0], 'Energy Consumption v.s. Batch Size', 'Batch Size', 'Energy Consumption (mJ)')
    axis[0].grid(True, linestyle='--', linewidth=0.5)
    axis[0].legend(['GPU Energy', '95% CI', 'CPU Energy', '95% CI', 'DRAM Energy', '95% CI'])

    # The second row - Throughput and Latency
    line1 = axis[1].plot(batch_sizes, results['mean']['Throughput'], color=pallate['throughput'], marker='o')
    axis[1].fill_between(batch_sizes, results['ci_lower']['Throughput'], results['ci_upper']['Throughput'], color=pallate['throughput'], alpha=0.2)
    axis[1].tick_params(axis='y', labelcolor=pallate['throughput'])
    axis[1].yaxis.label.set_color(pallate['throughput'])
    # twin axis
    ax1t = axis[1].twinx()
    line2 = ax1t.plot(batch_sizes, results['mean']['Latency'], color=pallate['latency'], marker='o')
    ax1t.fill_between(batch_sizes, results['ci_lower']['Latency'], results['ci_upper']['Latency'], color=pallate['latency'], alpha=0.2)
    ax1t.set_ylabel('Latency (ns)')
    ax1t.tick_params(axis='y', labelcolor=pallate['latency'])
    ax1t.yaxis.label.set_color(pallate['latency'])

    label_plot(axis[1], 'Throughput & Latency v.s. Batch Size', 'Batch Size', 'Throughput (images/sec)')
    axis[1].grid(True, linestyle='--', linewidth=0.5)
    axis[1].legend(line1+line2, ['Throughput', 'Latency'], loc='lower right')

    # The third row - GPU Utilization and VRAM Utilization
    axis[2].plot(batch_sizes, results['mean']['GPU Util'], color=pallate['gpu_util'], marker='o')
    axis[2].plot(batch_sizes, results['mean']['VRAM Util'], color=pallate['vram_util'], marker='o')
    axis[2].plot(batch_sizes, results['mean']['VRAM Used'], color=pallate['vram_used'], marker='o')
    axis[2].plot(batch_sizes, results['mean']['GPU Temp'], color=pallate['gpu_temp'], marker='o')

    label_plot(axis[2], 'GPU Utilization / Metrics v.s. Batch Size', 'Batch Size', 'Utilization (%) / Temperature (C)')
    axis[2].grid(True, linestyle='--', linewidth=0.5)
    axis[2].legend(['GPU Utilization', 'VRAM Utilization', 'VRAM Used', 'GPU Temperature'])

    # The fourth row - CPU Utilization and DRAM Utilization
    axis[3].plot(batch_sizes, results['mean']['CPU Util'], color=pallate['cpu_util'], marker='o')
    axis[3].plot(batch_sizes, results['mean']['DRAM Used'], color=pallate['dram_used'], marker='o')

    label_plot(axis[3], 'CPU/DRAM Utilization v.s. Batch Size', 'Batch Size', 'Utilization (%)')
    axis[3].grid(True, linestyle='--', linewidth=0.5)
    axis[3].legend(['CPU Utilization', 'DRAM Utilization'])

    for ax in axis:    ax.vlines(min_bs, ax.get_ylim()[0], ax.get_ylim()[1], color='black', linestyle=':')

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.3, hspace=0.25)
    fig.set_size_inches(12, 20)
    plt.savefig(os.path.join(results_pth, 'model_result.pdf'), format='pdf', bbox_inches='tight')
    plt.close('all')

    # Plotting the Throughput vs energy curve
    GPU_TDP = int(float(hw['GPU_TDP']))
    x = np.linspace(0.9*min_energy/1e6, 1.1*max_energy/1e6, 64)
    y = GPU_TDP / x

    plt.figure()
    fig, axis = plt.subplots(nrows=2, ncols=2)

    # TOP LEFT
    axis[0][0].plot(results['mean']['GPU Energy']/1e6, results['mean']['Throughput'], color=pallate['total'], marker='o', label=model.full_name)
    axis[0][0].text(results['mean']['GPU Energy'][0]/1e6, results['mean']['Throughput'][0], '  BS=1', ha='left', va='bottom')
    axis[0][0].text(results['mean']['GPU Energy'][min_idx]/1e6, results['mean']['Throughput'][min_idx], f'  ME BS={min_bs}', ha='left', va='bottom')
    axis[0][0].plot(x, y, linestyle='-', color='black', linewidth=4, alpha=0.2, label=f'TDP @ {GPU_TDP}W')

    axis[0][0].set_title('GPU Energy vs Throughput') 
    axis[0][0].set_xlabel('GPU Energy (Joules)')
    axis[0][0].set_ylabel('Throughput (images/sec)')
    axis[0][0].grid(True, linestyle='--', linewidth=0.5)
    axis[0][0].legend(loc='upper right')

    # TOP RIGHT
    axis[0][1].plot(results['mean']['GPU Energy']/1e6, results['mean']['Throughput'], color=pallate['total'], marker='o', label=model.full_name)
    axis[0][1].text(results['mean']['GPU Energy'][0]/1e6, results['mean']['Throughput'][0], '  BS=1', ha='left', va='bottom')
    axis[0][1].text(results['mean']['GPU Energy'][min_idx]/1e6, results['mean']['Throughput'][min_idx], f'  ME BS={min_bs}', ha='left', va='bottom')
    axis[0][1].plot(x, y, linestyle='-', color='black', linewidth=4, alpha=0.2, label=f'TDP @ {GPU_TDP}W')

    axis[0][1].set_xscale('log')
    axis[0][1].set_yscale('log')
    axis[0][1].minorticks_on()

    axis[0][1].set_title('GPU Energy vs Throughput (Log)') 
    axis[0][1].set_xlabel('GPU Energy (Joules)')
    axis[0][1].set_ylabel('Throughput (images/sec)')
    axis[0][1].grid(which='both', axis='both', linestyle='--', linewidth=0.5)
    axis[0][1].legend(loc='upper right')

    # BOTTOM LEFT
    model_TDP = [GPU_energy/1e6 * Throughput for GPU_energy, Throughput in zip(results['mean']['GPU Energy'], results['mean']['Throughput'])]
    axis[1][0].plot(batch_sizes, model_TDP, color=pallate['total'], marker='o', label=model.full_name)
    axis[1][0].text(1, model_TDP[0], '  BS=1', ha='left', va='bottom')
    axis[1][0].text(min_bs, model_TDP[min_idx], f'  ME BS={min_bs}', ha='center', va='top')
    
    axis[1][0].axhline(y=GPU_TDP, linestyle='-', color='black', linewidth=4, alpha=0.2, label=f'TDP @ {GPU_TDP}W')

    axis[1][0].set_title('Batch Size vs Model TDP') 
    axis[1][0].set_xlabel('Batch Size')
    axis[1][0].set_ylabel('Model TDP (W)')
    axis[1][0].grid(which='both', axis='both', linestyle='--', linewidth=0.5)
    axis[1][0].legend(loc='lower right')

    # BOTTOM RIGHT
    axis[1][1].bar([str(bs) for bs in batch_sizes], [GPU_TDP - tdp for tdp in model_TDP], color=pallate['total'])

    axis[1][1].set_title('Batch Size vs Model TDP Loss') 
    axis[1][1].set_xlabel('Batch Size')
    axis[1][1].set_ylabel('Model TDP Loss(W)')
    axis[1][1].grid(which='both', axis='both', linestyle='--', linewidth=0.5)

    fig.set_size_inches(16, 12)
    plt.savefig(os.path.join(results_pth, 'TDP_Util.pdf'), format='pdf', bbox_inches='tight')
    plt.close('all')



    # Update model db
    best_batch = pd.read_csv(os.path.join(results_pth, 'raw_data', str(min_bs), 'batch_result_stats.csv'), index_col=0)
    best_batch.to_csv(os.path.join(results_pth, f'best_batch_{min_bs}.csv'), index=True)

    # model.model_db = pd.read_csv(model.model_db_path)
    # posn = model.model_db.loc[model.model_db['Weight Name'] == model.weight_name].index[0]
    # model.model_db.loc[posn, 'Throughput'] = best_batch['Throughput']['mean']
    # model.model_db.loc[posn, 'Latency'] = best_batch['Latency']['mean']
    # model.model_db.loc[posn, 'Energy CPU'] = best_batch['CPU Energy']['mean']
    # model.model_db.loc[posn, 'Energy DRAM'] = best_batch['DRAM Energy']['mean']
    # model.model_db.loc[posn, 'Energy GPU'] = best_batch['GPU Energy']['mean']
    # model.model_db.loc[posn, 'Energy Total'] = best_batch['Total Energy']['mean']
    # model.model_db.loc[posn, 'DONE'] = True

    # model.model_db.to_csv(model.model_db_path, index=False)

    model.update_model_db('Optimal BS', int(min_bs))
    model.update_model_db('Throughput', best_batch['Throughput']['mean'])
    model.update_model_db('Latency', best_batch['Latency']['mean'])
    model.update_model_db('Energy', best_batch['GPU Energy']['mean'])
    # model.update_model_db('DONE', True)



def process_batch(bs_pth, hw, pallate, model, bs):
    def process_repetition(path, rep):
        # print(f'            Processing repetition {rep}...')

        # Read per batch timestamp file
        ts = pd.read_csv(os.path.join(path, 'timestamps.csv'), header=None)
        # rename the column
        ts.rename(columns={0:'timestamp'}, inplace=True)
        t_diff = ts.iloc[-1,0] - ts.iloc[0,0]
        t_threshold = t_diff * 0.05
        ts = ts[ts['timestamp'] > t_threshold + ts.iloc[0,0]]
        t0 = ts.iloc[0,0]
        ts['timestamp'] -= t0
        ts.set_index('timestamp', inplace=True)

        # Read CPU utilization file
        # cpu_util_data = pd.read_csv(os.path.join(path, 'cpuutil.csv'))
        # cpu_util_data['timestamp'] -= t0
        # cpu_util_data.set_index('timestamp', inplace=True)
        # cpu_util_data['DRAM_used'] = (hw['DRAM_size'] - cpu_util_data['available_DRAM_B']) / hw['DRAM_size'] * 100
        # # delete the available_DRAM_B column
        # del cpu_util_data['available_DRAM_B']
        # cpu_util_data = cpu_util_data.loc[cpu_util_data.index.repeat(2)]
        # cpu_util_data['CPU_util'] = cpu_util_data['CPU_util'].shift(-1)
        # cpu_util_data['DRAM_used'] = cpu_util_data['DRAM_used'].shift(-1)
        # cpu_util_data = cpu_util_data.iloc[1:-1]
        
        # Read CPU energy file
        # cpu_energy_data = pd.read_csv(os.path.join(path, 'cpudata.csv'))
        # cpu_energy_data['timestamp'] -= t0
        # cpu_energy_data.set_index('timestamp', inplace=True)
        # cpu_max_uj = {col : uj for col, uj in zip(cpu_energy_data.columns, hw['CPU_max_uj'])}

        # # remove a column if the column name does not contain 'package' or 'dram'
        # for col in cpu_energy_data.columns:
        #     if not any(word in col.lower() for word in ['package', 'dram']):
        #         del cpu_energy_data[col]

        # # Check if diff of each column is all positive
        # for col in cpu_energy_data.columns:
        #     if any(cpu_energy_data[col].diff() < 0):
        #         neg_idx = cpu_energy_data[col].diff()[cpu_energy_data[col].diff() < 0].index
        #         cpu_energy_data[col].loc[neg_idx.values[0]:] += cpu_max_uj[col]

        # cpu_energy_data['PKG'] = 0
        # cpu_energy_data['MEM'] = 0

        # for col in cpu_energy_data.columns[:-2]:
        #     if 'package' in col.lower():    cpu_energy_data['PKG'] += cpu_energy_data[col]
        #     elif  'dram' in col.lower():    cpu_energy_data['MEM'] += cpu_energy_data[col]

        # cpu_energy_data['PKG'] -= cpu_energy_data['PKG'].min()
        # cpu_energy_data['MEM'] -= cpu_energy_data['MEM'].min()
        
        # Read GPU data file
        gpu_data = pd.read_csv(os.path.join(path, 'gpudata.csv'))
        gpu_data['timestamp'] -= t0
        gpu_data.set_index('timestamp', inplace=True)

        gpu_power_data = gpu_data['power.draw [W]']
        gpu_power_data = gpu_power_data.loc[gpu_power_data.diff().shift(-1) != 0]
        gpu_power_data = gpu_power_data.loc[gpu_power_data.index.repeat(2)].shift(-1).iloc[1:-1]

        gpu_util_data = gpu_data['utilization.gpu [%]']
        vram_util_data = gpu_data['utilization.memory [%]']

        vram_used_data = gpu_data['memory.used [MiB]']
        vram_used_data = vram_used_data.loc[vram_used_data.diff().shift(-1) != 0]
        vram_used_data = vram_used_data / hw['GPU_mem_size'] * 100

        gpu_temp_data = gpu_data['temperature.gpu']

        # different data representation types
            # CPU energy:  Accumulation                  
            # DRAM energy: Accumulation                  
            # GPU power:   Average over past period      
            # CPU util:    Average over past period      
            # DRAM used:   Average over past period      
            # GPU util:    Average over past period
            # VRAM util:   Average over past period
            # VRAM used:   Instantaneous sample
            # GPU temp:    Instantaneous sample

        # Interpolate data at the batch timestamps
        # ts['PKG_uj']    = np.interp(ts.index, cpu_energy_data.index, cpu_energy_data['PKG'])
        # ts['MEM_uj']    = np.interp(ts.index, cpu_energy_data.index, cpu_energy_data['MEM'])
        ts['GPU_W']     = np.interp(ts.index, gpu_power_data.index, gpu_power_data)
        # ts['CPU_util']  = np.interp(ts.index, cpu_util_data.index, cpu_util_data['CPU_util'])
        # ts['DRAM_used'] = np.interp(ts.index, cpu_util_data.index, cpu_util_data['DRAM_used'])
        ts['GPU_util']  = np.interp(ts.index, gpu_util_data.index, gpu_util_data)
        ts['VRAM_util'] = np.interp(ts.index, vram_util_data.index, vram_util_data)
        ts['VRAM_used'] = np.interp(ts.index, vram_used_data.index, vram_used_data)
        ts['GPU_temp']  = np.interp(ts.index, gpu_temp_data.index, gpu_temp_data)
            
        data = [['Latency', 'Throughput', 'CPU Energy', 'DRAM Energy', 'GPU Energy', 'Total Energy', 
                'CPU Util', 'DRAM Used', 'GPU Util', 'VRAM Util', 'VRAM Used', 'GPU Temp']]
        for (t1, row1), (t2, row2) in zip(ts.iterrows(), ts[1:].iterrows()):
            # Latency (ns)
            latency = t2 - t1
            # Throughput (samples/s)
            throughput = int(bs) / latency * 1e9

            # CPU & DRAM energy
            # cpu_energy = (ts['PKG_uj'][t2] - ts['PKG_uj'][t1]) / int(bs)
            # dram_energy = (ts['MEM_uj'][t2] - ts['MEM_uj'][t1]) / int(bs)
            cpu_energy = 0
            dram_energy = 0

            # CPU and DRAM util
            # cpu_util_data_slice = cpu_util_data.loc[t1:t2]
            # time_arr = np.hstack((t1, cpu_util_data_slice.index, t2))
            # cpu_util_data_arr = np.hstack((row1['CPU_util'], cpu_util_data_slice['CPU_util'].values, row2['CPU_util']))
            # dram_used_arr = np.hstack((row1['DRAM_used'], cpu_util_data_slice['DRAM_used'].values, row2['DRAM_used']))
            # cpu_util = np.trapz(cpu_util_data_arr, time_arr) / (t2 - t1)
            # dram_used = np.trapz(dram_used_arr, time_arr) / (t2 - t1)
            cpu_util = 0
            dram_used = 0

            # GPU energy and util
            gpu_power_slice = gpu_power_data.loc[t1:t2]
            time_arr = np.hstack((t1, gpu_power_slice.index, t2))
            gpu_power_arr = np.hstack((row1['GPU_W'], gpu_power_slice, row2['GPU_W']))
            gpu_energy = np.trapz(gpu_power_arr, time_arr) / int(bs) / 1000
            total_energy = cpu_energy + dram_energy + gpu_energy

            gpu_util_slice = gpu_util_data.loc[t1:t2]
            time_arr = np.hstack((t1, gpu_util_slice.index, t2))
            gpu_util_arr = np.hstack((row1['GPU_util'], gpu_util_slice, row2['GPU_util']))
            gpu_util = np.trapz(gpu_util_arr, time_arr) / (t2 - t1)

            vram_util_slice = vram_util_data.loc[t1:t2]
            time_arr = np.hstack((t1, vram_util_slice.index, t2))
            vram_util_arr = np.hstack((row1['VRAM_util'], vram_util_slice, row2['VRAM_util']))
            vram_util = np.trapz(vram_util_arr, time_arr) / (t2 - t1)

            vram_used_slice = vram_used_data.loc[t1:t2]
            time_arr = np.hstack((t1, vram_used_slice.index, t2))
            vram_used_arr = np.hstack((row1['VRAM_used'], vram_used_slice, row2['VRAM_used']))
            vram_used = np.trapz(vram_used_arr, time_arr) / (t2 - t1)

            gpu_temp_slice = gpu_temp_data.loc[t1:t2]
            time_arr = np.hstack((t1, gpu_temp_slice.index, t2))
            gpu_temp_arr = np.hstack((row1['GPU_temp'], gpu_temp_slice, row2['GPU_temp']))
            gpu_temp = np.trapz(gpu_temp_arr, time_arr) / (t2 - t1)

            data.append([latency, throughput, cpu_energy, dram_energy, gpu_energy, total_energy,
                        cpu_util, dram_used, gpu_util, vram_util, vram_used, gpu_temp])

        # Create a dataframe from the data
        data = pd.DataFrame(data, columns=data.pop(0))
        data.to_csv(os.path.join(path, 'rep_result.csv'), index=False)

            

        # Plotting the raw data for information
        plt.figure()
        fig, axis = plt.subplots(nrows=4, ncols=4)

        # The first row - Sampling Periods
        period_list = gpu_power_data.index.diff()
        period_list = period_list[period_list > 0]
        axis[0,0].plot(period_list, color=pallate['gpu'])
        label_plot(axis[0,0], 'GPU Power Update Period', 'Samples', 'Time (ms)')

        axis[0,1].hist(period_list, color=pallate['gpu'], bins=25)
        label_plot(axis[0,1], 'GPU Power Update Period Histogram', 'Time (ms)', 'Sample Count')
        axis[0,1].legend(ks_test(period_list))

        # axis[0,2].plot(cpu_energy_data.index.diff(), color=pallate['cpu'])
        # label_plot(axis[0,2], 'CPU Sampling Period', 'Samples', 'Time (ms)')

        # axis[0,3].hist(cpu_energy_data.index.diff(), color=pallate['cpu'], bins=25)
        # label_plot(axis[0,3], 'CPU Sampling Period Histogram', 'Time (ms)', 'Sample Count')
        # axis[0,3].legend(ks_test(cpu_energy_data.index.diff()[1:].to_numpy()))

        # The second row - Samples: Power and Energy
        axis[1,0].plot(gpu_power_data.index, gpu_power_data, color=pallate['gpu'])
        axis[1,0].vlines(ts.index, gpu_power_data.min(), gpu_power_data.min() + (gpu_power_data.max()-gpu_power_data.min())/10, color='black', linestyle=':')
        label_plot(axis[1,0], 'GPU Power Draw over Time', 'Time (ms)', 'Power Draw (W)')

        axis[1,1].hist(gpu_power_data, color=pallate['gpu'], bins=25)
        label_plot(axis[1,1], 'GPU Power Draw Histogram', 'Power Draw (W)', 'Sample Count')
        axis[1,1].legend(ks_test(gpu_power_data))

        # axis[1,2].plot(cpu_energy_data.index, cpu_energy_data['PKG'], color=pallate['cpu'])
        # axis[1,2].plot(cpu_energy_data.index, cpu_energy_data['MEM'], color=pallate['dram'], linestyle='--')
        # label_plot(axis[1,2], 'CPU/DRAM Energy Consumption over Time', 'Time (ms)', 'Energy Consumption (mJ)')
        # axis[1,2].legend(['CPU', 'DRAM'])
        # axis[1,2].vlines(ts.index, cpu_energy_data['PKG'].min(), cpu_energy_data['PKG'].min() - (cpu_energy_data['PKG'].max()-cpu_energy_data['PKG'].min())/10, color='black', linestyle=':')

        axis[1,3].hist(diff_l(ts.index), color=pallate['total'], bins=16)
        label_plot(axis[1,3], 'Latency Histogram', 'Latency (ms)', 'Sample Count')
        axis[1,3].legend(ks_test(diff_l(ts.index)))

        # The third and fourth rows - CPU, DRAM, GPU and total energy consumption
        axis[2,0].plot(data['GPU Energy'], color=pallate['gpu'])
        label_plot(axis[2,0], 'GPU Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
        axis[3,0].hist(data['GPU Energy'], color=pallate['gpu'], bins=25)
        label_plot(axis[3,0], 'GPU Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
        axis[3,0].legend(ks_test(data['GPU Energy']))

        # axis[2,1].plot(data['CPU Energy'], color=pallate['cpu'])
        # label_plot(axis[2,1], 'CPU Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
        # axis[3,1].hist(data['CPU Energy'], color=pallate['cpu'], bins=25)
        # label_plot(axis[3,1], 'CPU Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
        # axis[3,1].legend(ks_test(data['CPU Energy']))

        # axis[2,2].plot(data['DRAM Energy'], color=pallate['dram'])
        # label_plot(axis[2,2], 'DRAM Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
        # axis[3,2].hist(data['DRAM Energy'], color=pallate['dram'], bins=25)
        # label_plot(axis[3,2], 'DRAM Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
        # axis[3,2].legend(ks_test(data['DRAM Energy']))

        axis[2,3].plot(data['Total Energy'], color=pallate['total'])
        label_plot(axis[2,3], 'Total Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
        axis[3,3].hist(data['Total Energy'], color=pallate['total'], bins=25)
        label_plot(axis[3,3], 'Total Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
        axis[3,3].legend(ks_test(data['Total Energy']))


        fig.set_size_inches(30, 22)
        plt.savefig(os.path.join(path, 'repetition_result.pdf'), format='pdf', bbox_inches='tight')
        plt.close('all')
        
    # ----------------------------------------------------------------------- #   
    # Process Batch
    rep_path = os.path.join(bs_pth, bs)
    rep_list = [d for d in os.listdir(rep_path) if os.path.isdir(os.path.join(rep_path, d))]
    rep_list = list(map(int, rep_list))
    rep_list.sort()
    rep_list = list(map(str, rep_list))

    for rep in rep_list: process_repetition(os.path.join(rep_path, rep), rep)
    
    batch_data = pd.concat([pd.read_csv(os.path.join(rep_path, rep, 'rep_result.csv')) for rep in rep_list], ignore_index=True)
    batch_data.to_csv(os.path.join(rep_path, f'batch_result.csv'), index=False)

    ks_p_values = {column: stats.kstest(batch_data[column], 'norm', args=(batch_data[column].mean(), batch_data[column].std()))[1]
                    for column in batch_data.columns}
    means = batch_data.mean()
    stds = batch_data.std()
    sems = batch_data.sem()
    ci_lower = means - 1.96 * sems
    ci_upper = means + 1.96 * sems

    result_stats = pd.DataFrame([ks_p_values, means, stds, sems, ci_lower, ci_upper], index=['ks_p', 'mean', 'std', 'sem', 'ci_lower', 'ci_upper'])
    result_stats.to_csv(os.path.join(rep_path, f'batch_result_stats.csv'), index=True)

    # Plotting the batch data
    plt.figure()
    fig, axis = plt.subplots(nrows=3, ncols=4)

    # The first and second rows - energy consumption of cpu, dram, gpu and total
    axis[0,0].plot(batch_data['GPU Energy'], color=pallate['gpu'])
    label_plot(axis[0,0], 'GPU Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
    axis[1,0].hist(batch_data['GPU Energy'], color=pallate['gpu'], bins=16)
    label_plot(axis[1,0], 'GPU Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
    axis[1,0].legend(ks_test(batch_data['GPU Energy']))

    axis[0,1].plot(batch_data['CPU Energy'], color=pallate['cpu'])
    label_plot(axis[0,1], 'CPU Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
    axis[1,1].hist(batch_data['CPU Energy'], color=pallate['cpu'], bins=16)
    label_plot(axis[1,1], 'CPU Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
    axis[1,1].legend(ks_test(batch_data['CPU Energy']))

    axis[0,2].plot(batch_data['DRAM Energy'], color=pallate['dram'])
    label_plot(axis[0,2], 'DRAM Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
    axis[1,2].hist(batch_data['DRAM Energy'], color=pallate['dram'], bins=16)
    label_plot(axis[1,2], 'DRAM Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
    axis[1,2].legend(ks_test(batch_data['DRAM Energy']))

    axis[0,3].plot(batch_data['Total Energy'], color=pallate['total'])
    label_plot(axis[0,3], 'Total Energy Consumption of Each Batch', 'Batch', 'Energy Consumption (uJ)')
    axis[1,3].hist(batch_data['Total Energy'], color=pallate['total'], bins=16)
    label_plot(axis[1,3], 'Total Energy Consumption Histogram', 'Energy Consumption (uJ)', 'Sample Count')
    axis[1,3].legend(ks_test(batch_data['Total Energy']))

    # The third row - Latency and Throughput
    axis[2,0].plot(batch_data['Latency'], color=pallate['total'])
    label_plot(axis[2,0], 'Latency of Each Batch', 'Batch', 'Latency (ns)')
    axis[2,1].hist(batch_data['Latency'], color=pallate['total'], bins=16)
    label_plot(axis[2,1], 'Latency Histogram', 'Latency (ns)', 'Sample Count')
    axis[2,1].legend(ks_test(batch_data['Latency']))

    axis[2,2].plot(batch_data['Throughput'], color=pallate['total'])
    label_plot(axis[2,2], 'Throughput of Each Batch', 'Batch', 'Throughput (samples/s)')
    axis[2,3].hist(batch_data['Throughput'], color=pallate['total'], bins=16)
    label_plot(axis[2,3], 'Throughput Histogram', 'Throughput (samples/s)', 'Sample Count')
    axis[2,3].legend(ks_test(batch_data['Throughput']))

    # save the plot
    fig.set_size_inches(30, 18)
    plt.savefig(os.path.join(rep_path, 'batch_result.pdf'), format='pdf', bbox_inches='tight')
    plt.close('all')

    print('.', end='', flush=True)

def label_plot(ax, title, x, y):
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

def ks_test(data):
    _, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    pass_fail = 'PASSED' if p_value > 0.05 else 'FAILED'
    return [f'KS Test: {pass_fail}, p-value: {p_value:.4f}']

def diff_l(list): return [list[i+1] - list[i] for i in range(len(list)-1)]

def log(pth, msg, end='\n'):
    print(msg, end=end, flush=True)
    with open(pth, 'a') as f:
        f.write(f'{msg}{end}')