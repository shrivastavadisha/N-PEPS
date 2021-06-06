import os
import argparse
import json
import math
import statistics
import pandas as pd

def analyze(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    res = [json.loads(x) for x in lines]

    succeeded = [x for x in res if x['result'] != 'Failed']

    for x in res:
        if x['result'] == 'Failed':
            x['time'] = math.inf

    #times = sorted([x['time'] for x in res])

    #for ratio in [0.05, 0.1, 0.2,  0.4, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.99]:
        #print("%f: %f" % (ratio, times[int(ratio * len(times)) - 1]))

    #print("Total solved: %d\\%d - %f%%" % (len(succeeded), len(res), len(succeeded) / len(res) * 100.0))
    return len(succeeded) / len(res) * 100.0

def store_results(filenames, out_filename, success_ratio, result_dir):
    df = pd.DataFrame()
    splits, success_ratios, methods, gps_timeouts, peps_timeouts, agg_modes, agg_types, alphas, agg_inps, agg_models, machine_names, seeds = \
        [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(len(filenames)):
        file = filenames[i]
        parts = file.split("#")
        machine_names.append(parts[0])
        seeds.append(parts[-1])
        splits.append(parts[2])
        methods.append(parts[3])
        success_ratios.append(success_ratio[i])
        if parts[3] =='gps':
            gps_timeouts.append(parts[4])
            peps_timeouts.append("")
            agg_modes.append("")
            agg_inps.append("")
            agg_types.append("")
            alphas.append("")
            agg_models.append("")
        else:
            gps_timeouts.append(parts[4])
            peps_timeouts.append(parts[6])
            agg_inps.append(parts[7])
            agg_modes.append(parts[8])
            agg_types.append(parts[9])
            alphas.append(parts[10])
            agg_models.append(parts[11])

    df["splits"] = splits
    df["method"] = methods
    df["gps_timeout"] = gps_timeouts
    df["peps_timeout"] = peps_timeouts
    df["agg_mode"] = agg_modes
    df["agg_type"] = agg_types
    df['agg_models'] = agg_models
    df["agg_inps"] = agg_inps
    df["alpha"] = alphas
    df["machine_name"] = machine_names
    df["seed"] = seeds
    df["success_ratio"] = success_ratios

    df.to_csv(os.path.join(result_dir, out_filename))

def analyze_results(result_dir, out_filename):
    all_dirs = os.listdir(result_dir)
    res_dirs = []
    success_ratios = []
    for file in all_dirs:
        path = os.path.join(result_dir, file)
        success_ratios.append(analyze(path))
        res_dirs.append(file)
    store_results(res_dirs, out_filename, success_ratios, result_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results/E1/test/')
    parser.add_argument('--out_filename', type=str, default='combined_results.csv')
    args = parser.parse_args()

    analyze_results(args.dir, args.out_filename)


if __name__ == '__main__':
    main()
