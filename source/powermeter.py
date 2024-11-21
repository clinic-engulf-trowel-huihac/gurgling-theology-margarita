import time
from datetime import datetime
import json
import csv
import os
import os.path
from multiprocessing import Process, Value
import subprocess
import numpy as np
import pandas as pd
import torch
import psutil
from datetime import datetime, timezone


import contextlib
from io import StringIO

class powermeter:
    def __init__(self, exp_name, period_ms = 10, gpu_id = 0, force_node_name = None):
        self.exp_name = exp_name
        self.force_node_name = force_node_name
        self._get_cpu_info()
        self._get_gpu_info()
        self._setup_meter()

        self.period_ms = period_ms
        self.gpu_id = gpu_id

    def _get_cpu_info(self):
        lscpu = subprocess.run(['lscpu'], stdout=subprocess.PIPE)
        self.cpuinfo = dict((a.strip(), b.strip())
                            for a, b in (element.split(':', 1)
                                for element in lscpu.stdout.decode('utf-8').splitlines()))

    def _get_gpu_info(self):
        smi = subprocess.run(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE)
        self.gpuinfo = smi.stdout.decode('utf-8')
        smi = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,uuid,driver_version,power.limit', 
                '--format=csv,noheader,nounits'
            ], stdout=subprocess.PIPE)
        smi = smi.stdout.decode('utf-8') #.replace(' ', '_')        
        self.ngpus = smi.count('\n')
        self.gpu_name, self.gpu_mem_size, self.gpu_uuid, self.gpu_driver_v, self.gpu_tdp = smi.splitlines()[0].split(', ')
        self.gpu_mem_size = int(self.gpu_mem_size)
        self.gpu_name_short = self.gpu_name.split()
        self.gpu_name_short = '_'.join(word for word in self.gpu_name_short if word.lower() not in ['nvidia', 'tesla', 'geforce'])

        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            output = result.stdout
            nvcc_version = output.split('\n')[3].split(',')[1].strip()
            self.nvcc_version = nvcc_version
        except FileNotFoundError:
            print("CUDA is not installed, or 'nvcc' command not found.")
            self.nvcc_version = 'N/A'

    def _setup_meter(self):
        hostname = subprocess.run(['hostname'], stdout=subprocess.PIPE)
        if self.force_node_name is None:
            self.host_node = hostname.stdout.decode('utf-8')[:-1] + '_' + self.gpu_name_short
        else:
            self.host_node = self.force_node_name

        # for CPU
        if self.cpuinfo['Vendor ID'] == 'GenuineIntel':
            self.cpu_domains = []
            self.cpu_names = []
            self.cpu_max_uj = []
            self.cpu_uj_paths = []
            for root, _, filenames in os.walk("/sys/class/powercap/intel-rapl", topdown=True):
                if 'name' in filenames:
                    self.cpu_domains.append(root.split('/')[-1])
                    self.cpu_names.append(self._read_file(os.path.join(root, 'name')))
                    self.cpu_max_uj.append(int(self._read_file(os.path.join(root, 'max_energy_range_uj'))))
                    self.cpu_uj_paths.append(os.path.join(root, 'energy_uj'))

        elif self.cpuinfo['Vendor ID'] == 'AuthenticAMD':
            self.cpu_names = ['package-1', 'dram', 'package-0', 'dram']
            self.cpu_max_uj = [65712999613, 0, 65712999613, 0]
        else:
            raise Exception("Unsupported CPU")
        
        self.cpu_name = self.cpuinfo["Model name"]

        # for DRAM
        self.total_DRAM_B = psutil.virtual_memory().total

        # for GPU
        self.smi_query_attributes = 'index,\
                                    timestamp,\
                                    power.draw,\
                                    clocks.current.sm,\
                                    clocks.current.memory,\
                                    utilization.gpu,\
                                    utilization.memory,\
                                    temperature.gpu,\
                                    memory.used,\
                                    pstate'.replace(' ', '')

        # log hardware info
        if not os.path.exists(f'results/{self.exp_name}/'):
            os.system(f'mkdir results/{self.exp_name}/')

        if not os.path.exists(f'results/{self.exp_name}/{self.host_node}/'):
            os.system(f'mkdir results/{self.exp_name}/{self.host_node}/')

        self.exp_path = f'results/{self.exp_name}/{self.host_node}/'

        if not os.path.exists(f'results/{self.exp_name}/{self.host_node}/hardware_info/'):
            # print("Logging hardware info")
            os.system(f'mkdir results/{self.exp_name}/{self.host_node}/hardware_info/')

            with open(f'results/{self.exp_name}/{self.host_node}/hardware_info/summary.json', 'w') as hw_summary_file:
                hw_summary = {
                    'CPU Name': self.cpuinfo["Model name"],
                    'CPU_max_uj': self.cpu_max_uj,
                    'GPU Name': self.gpu_name,
                    'GPU_mem_size': self.gpu_mem_size,
                    'GPU_TDP': self.gpu_tdp,
                    'DRAM_size': self.total_DRAM_B,
                }
                json.dump(hw_summary, hw_summary_file)

            with open(f'results/{self.exp_name}/{self.host_node}/hardware_info/cpu_info.json', 'w') as cpu_info_file:
                json.dump(self.cpuinfo, cpu_info_file)

            with open(f'results/{self.exp_name}/{self.host_node}/hardware_info/gpu_info.xml', 'w') as gpu_info_file:
                gpu_info_file.write(self.gpuinfo)


        if not os.path.exists(f'results/{self.exp_name}/{self.host_node}/log/'):
            os.system(f'mkdir results/{self.exp_name}/{self.host_node}/log/')
        
        log_list = os.listdir(f'results/{self.exp_name}/{self.host_node}/log/')
        self.log_file = f'results/{self.exp_name}/{self.host_node}/log/logfile_{len(log_list)+1}.txt'

    def general_info(self):
        exp_name =     '-- ' + self.exp_name + ' --'
        time_ =        'Date and time:        ' + time.strftime('%d/%m/%Y %H:%M:%S')
        host =         'Host machine:         ' + os.uname()[1]
        cpu =          'CPU:                  ' + self.cpu_name
        gpu =          'GPU:                  ' + self.gpu_name
        gpu_mem_size = 'GPU memory size:      ' + str(int(self.gpu_mem_size/1024)) + ' GiB'
        gpu_id =       'GPU ID:               ' + str(self.gpu_id)
        uuid =         'GPU UUID:             ' + self.gpu_uuid
        cuda =         'CUDA version:         ' + self.nvcc_version
        driver =       'Driver version:       ' + self.gpu_driver_v

        max_len = max(len(exp_name), len(time_), len(host), len(cpu), len(gpu), len(gpu_mem_size), len(gpu_id), len(uuid), len(cuda), len(driver))
        output = ''
        output += '+ ' + '-'*(max_len) + ' +\n'
        output += '| ' + exp_name + ' '*(max_len - len(exp_name)) + ' |\n'
        output += '| ' + time_ + ' '*(max_len - len(time_)) + ' |\n'
        output += '| ' + host + ' '*(max_len - len(host)) + ' |\n'
        output += '| ' + cpu + ' '*(max_len - len(cpu)) + ' |\n'
        output += '| ' + gpu + ' '*(max_len - len(gpu)) + ' |\n'
        output += '| ' + gpu_mem_size + ' '*(max_len - len(gpu_mem_size)) + ' |\n'
        output += '| ' + gpu_id + ' '*(max_len - len(gpu_id)) + ' |\n'
        output += '| ' + uuid + ' '*(max_len - len(uuid)) + ' |\n'
        output += '| ' + cuda + ' '*(max_len - len(cuda)) + ' |\n'
        output += '| ' + driver + ' '*(max_len - len(driver)) + ' |\n'
        output += '+ ' + '-'*(max_len) + ' +\n'

        return output

    def _read_file(self, path):
        with open(path, "r") as f:
            contents = f.read().strip()
            return contents

    def _sample_cpu(self):
        period = self.period_ms/1000
        cpudata_file = open(f'{self.path}/cpudata.csv', 'a')
        cpuutil_file = open(f'{self.path}/cpuutil.csv', 'a')
        ctr = 0
        if self.cpuinfo['Vendor ID'] == 'GenuineIntel':
            while self.sampling_cpu.value:
                time.sleep(period)
                out = str(time.time_ns())
                for path in self.cpu_uj_paths:
                    out += ',' + self._read_file(path)
                cpudata_file.writelines(out + '\n')

                ctr += 1
                if ctr % 10 == 0:
                    out = f'{time.time_ns()},{psutil.cpu_percent()},{psutil.virtual_memory().available}'
                    cpuutil_file.writelines(out + '\n')

        elif self.cpuinfo['Vendor ID'] == 'AuthenticAMD':
            while self.sampling_cpu.value:
                out = str(time.time_ns())
                pkg_1 = subprocess.run(['sudo', 'rdmsr', '-p', '0', str(0xC001029B)], stdout=subprocess.PIPE)
                pkg_0 = subprocess.run(['sudo', 'rdmsr', '-p', '32', str(0xC001029B)], stdout=subprocess.PIPE)
                out += ',' + str(int(pkg_1.stdout.decode('utf-8'), 16)*15.3) + ',0'
                out += ',' + str(int(pkg_0.stdout.decode('utf-8'), 16)*15.3) + ',0'
                cpudata_file.writelines(out + '\n')

                ctr += 1
                if ctr % 20 == 0:
                    out = f'{time.time_ns()},{psutil.cpu_percent()},{psutil.virtual_memory().available}'
                    cpuutil_file.writelines(out + '\n')

        cpudata_file.close()
        cpuutil_file.close()

    def prep_new_model(self, model_name, repetition, device):
        self.model_name = model_name
        self.device = device
        self.model_path = f'results/{self.exp_name}/{self.host_node}/results/{model_name}/raw_data/'

        if repetition == 1:
            if not os.path.exists(f'results/{self.exp_name}/{self.host_node}/results/'):
                os.system(f'mkdir results/{self.exp_name}/{self.host_node}/results/')

            if not os.path.exists(f'results/{self.exp_name}/{self.host_node}/results/{model_name}'):
                os.system(f'mkdir results/{self.exp_name}/{self.host_node}/results/{model_name}')
                
            if os.path.exists(self.model_path):
                os.system(f'rm -r {self.model_path}')
            os.system(f'mkdir {self.model_path}')

        
    def prep_new_measurement(self, batchsize, repetition, forward_pass):
        self.path = os.path.join(self.model_path, str(batchsize))
        if not os.path.exists(self.path):
            os.system(f'mkdir {self.path}')
        self.path = os.path.join(self.path, str(repetition))
        if not os.path.exists(self.path):
            os.system(f'mkdir {self.path}')

        # if repetition == 1:
        #     with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA,
        #         ],
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True,
        #         with_flops=True,
        #         with_modules=True,
        #     ) as p:
        #         forward_pass(batchsize)
        #         forward_pass(batchsize)
        
        #     with open(f'{self.path}/profiler.txt', 'w') as f:
        #         f.write(p.key_averages().table())

        #     p.export_chrome_trace(f'{self.path}/testtrace.json')
        

        self.smi_query_cmd = ['nvidia-smi',
                                f'--query-gpu={self.smi_query_attributes}',
                                '--format=csv,nounits',
                                '-f', f'{self.path}/gpudata.csv',
                                '-lms', f'{self.period_ms}',
                                '-i', ','.join(self.device)]
        
        # with open(f'{self.path}/cpudata.csv', 'w') as cpudata_file:
        #     header = 'timestamp'
        #     for name in self.cpu_names:
        #         header += ',' + name
        #     cpudata_file.writelines(header + '\n')

        # with open(f'{self.path}/cpuutil.csv', 'w') as cpuutil_file:
        #     header = 'timestamp,CPU_util,available_DRAM_B'
        #     cpuutil_file.writelines(header + '\n')

        # self.sampling_cpu = Value('b', True)
        # self.cpu_measurement = Process(target=self._sample_cpu)

    def start_measurement(self):
        # self.cpu_measurement.start()
        self.gpu_measurement = subprocess.Popen(self.smi_query_cmd)
        time.sleep(1)

        if self.gpu_measurement.poll() is None:  print("GPU measurement started...", end='')
        else: print("GPU measurement failed to start...")
            

    def end_measurement(self, timestamps):
        time.sleep(1)
        # self.sampling_cpu.value = False
        self.gpu_measurement.terminate()
        # self.cpu_measurement.join()
        self.gpu_measurement.wait()
        time.sleep(0.1)

        print(f'terminated with code {self.gpu_measurement.poll()}    ', end='')

        with open(os.path.join(self.path, 'timestamps.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for timestamp in timestamps:
                writer.writerow([timestamp])

        # modify the nvidia-smi timestamps to epoch time (get rid of jetlag)
        result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp', '--format=csv,noheader'], stdout=subprocess.PIPE)
        epoch_time = time.time()
        nvsmi_time = result.stdout.decode().split('\n')[0]
        nvsmi_time = datetime.strptime(nvsmi_time, '%Y/%m/%d %H:%M:%S.%f')
        nvsmi_time = nvsmi_time.replace(tzinfo=timezone.utc).timestamp()
        jet_lag = round((epoch_time - nvsmi_time)/3600)

        attempts = 0.5
        while attempts < 5:
            try:
                time.sleep(attempts)  # Wait for longer each time
                print("-", end='')
                df = pd.read_csv(os.path.join(self.path, 'gpudata.csv'), sep=',\s+', engine='python')
                break
            except FileNotFoundError: attempts += 1
        
        if attempts == 5.5:    return False

        df['timestamp'] = (pd.to_datetime(df['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ns")
        df['timestamp'] += int(60*60*1e9) * jet_lag

        df.to_csv(os.path.join(self.path, 'gpudata.csv'), index=False)

        return True

