#!/usr/bin/env python
import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "data/models/huggingface"
os.environ['TORCH_HOME'] = 'data/models/torch'

import timm
import ptflops
import torch
import torchinfo
import pandas as pd


def main():

    header = 'model_name,input_size_csv,input_size_cfg,crop_pct_csv,crop_pct_cfg,params_csv,params_tinfo,macs_tinfo,params_ptflops,macs_ptflops,activations'
    model_df = pd.read_csv('source/results-imagenet_sorted.csv')

    with open('source/model_stats.csv', 'w') as file:  file.write(header + '\n')

    for index, row in model_df.iterrows():
        print(f"Processing model {index + 1}/{len(model_df)}")
        model_name = row['model']
        # print(f"Model: {model_name}")
        input_size_csv = row['img_size']
        crop_pct_csv = row['crop_pct']
        params_csv = row['param_count']

        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        model = model.cuda()

        cfg = timm.data.resolve_data_config({}, model=model)
        # print(cfg['input_size'][-1])
        # print(cfg['crop_pct'])
        input_size_cfg = cfg['input_size'][-1]
        crop_pct_cfg = cfg['crop_pct']

        stats = torchinfo.summary(model, input_size=(1, 3, input_size_csv, input_size_csv), verbose=0)
        # print(f"params (torchinfo): {stats.total_params}")
        # print(f"MACs (torchinfo): {stats.total_mult_adds}")
        params_tinfo = stats.total_params
        macs_tinfo = stats.total_mult_adds

        macs, params = ptflops.get_model_complexity_info(model, (3, input_size_csv, input_size_csv), as_strings=False, print_per_layer_stat=False, verbose=False, backend='pytorch')
        # print(f"macs: {macs}")
        # print(f"params: {params}")
        params_ptflops = params
        macs_ptflops = macs

        dummy_input = torch.randn(1, 3, input_size_csv, input_size_csv).cuda()
        total_activations = [0]

        def count_activations(name):
            def hook(model, input, output):
                num_elements = output.numel()
                total_activations[0] += num_elements
                del output
            return hook

        try:
            for name, layer in model.named_modules():  layer.register_forward_hook(count_activations(name))
            with torch.no_grad():  output = model(dummy_input)
        except Exception as e:  pass


        # print(f"Activations: {total_activations[0]}")
        activations = total_activations[0]

        data = f'{model_name},{input_size_csv},{input_size_cfg},{crop_pct_csv},{crop_pct_cfg},{params_csv},{params_tinfo},{macs_tinfo},{params_ptflops},{macs_ptflops},{activations}'
        print(data, flush=True)

        with open('source/model_stats.csv', 'a') as file:  file.write(data + '\n')

if __name__ == "__main__":
    main()