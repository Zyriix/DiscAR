# Copyright (c) 2026 Bowen Zheng
# The Chinese University of Hong Kong, Shenzhen
#
# Licensed under the MIT License.

import sys
import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf

# Add project root to path so we can import evaluator
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", type=str, required=True)
    parser.add_argument("--sample_batch", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True
    
    try:
        # Initialize Evaluator
        evaluator = Evaluator(tf.Session(config=config), batch_size=args.batch_size)
        evaluator.warmup()

        # Compute FID
        # read_activations handles .npz files if path is passed
        ref_acts = evaluator.read_activations(args.ref_batch)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

       
        try:
            with np.load(args.ref_batch) as obj:
                keys = set(obj.files)
            if "arr_0" in keys and "mu" not in keys and "sigma" not in keys:
                print('overwrite ref_batch with statistics')
                np.savez_compressed(
                    args.ref_batch,
                    mu=ref_stats.mu,
                    sigma=ref_stats.sigma,
                    mu_s=ref_stats_spatial.mu,
                    sigma_s=ref_stats_spatial.sigma,
                )
        except Exception:
            print('ref_batch are already statistics')
            pass
        
        sample_acts = evaluator.read_activations(args.sample_batch)
        sample_stats, _ = evaluator.read_statistics(args.sample_batch, sample_acts)
        
        fid = sample_stats.frechet_distance(ref_stats)
        print(f"FID_RESULT:{fid}")
    except Exception as e:
        print(f"Error computing FID: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


