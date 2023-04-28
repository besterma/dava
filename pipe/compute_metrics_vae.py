# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script how to get started with research using disentanglement_lib.

To run the example, please change the working directory to the containing folder
and run:
>> python example.py

In this example, we show how to use disentanglement_lib to:
1. Train a standard VAE (already implemented in disentanglement_lib).
2. Train a custom VAE model.
3. Extract the mean representations for both of these models.
4. Compute the Mutual Information Gap (already implemented) for both models.
5. Compute a custom disentanglement metric for both models.
6. Aggregate the results.
7. Print out the final Pandas data frame with the results.
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse



from beta_tcvae.vae_quant import VAE
from disentanglement_lib.evaluation.metrics import (
    beta_vae,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    downstream_task,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    factor_vae,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    modularity_explicitness,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    reduced_downstream_task,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    sap_score,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    unsupervised_metrics,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import nmig
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.udr import evaluate as evaluate_udr
import disentanglement_lib.evaluation.udr.metrics.udr as udr
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.utils import results
import gin

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dsprites")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="dava")
    parser.add_argument("--dataset_path", type=str, required=False, default=".")
    parser.add_argument("--num_channels", type=int, required=False, default=1)
    parser.add_argument("--store_prefix", type=str, default="metrics")
    parser.add_argument("--store_path", type=str, default="")
    parser.add_argument("--limited_gt_factors", type=str, default=None)
    parser.add_argument("--pipe", action='store_true')
    parser.add_argument("--mig", action='store_true')
    parser.add_argument("--dci", action='store_true')
    parser.add_argument("--factor_vae", action='store_true')
    parser.add_argument("--fairness", action='store_true')
    parser.add_argument("--udr", action='store_true')
    parser.add_argument("--udr_start", type=int, default=0)
    parser.add_argument("--udr_end", type=int, default=0)
    args = parser.parse_args()
    # 0. Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    dataset = args.dataset
    type = args.model_type
    paths = [args.model_path]
    for base_path in paths:
        if dataset == "dsprites":
            dataset_name = "dataset.name='dsprites_full'"
            num_channels = "VAE.num_channels = 1"
        elif dataset == "abstract_dsprites":
            dataset_name = "dataset.name='abstract_dsprites'"
            num_channels = "VAE.num_channels = 3"
        elif dataset == "noisy_dsprites":
            dataset_name = "dataset.name='noisy_dsprites'"
            num_channels = "VAE.num_channels = 3"
        elif dataset == "shapes3d":
            dataset_name = "dataset.name='shapes3d'"
            num_channels = "VAE.num_channels = 3"
        elif dataset == "cars3d":
            dataset_name = "dataset.name='cars3d'"
            num_channels = "VAE.num_channels = 3"
        elif dataset == "smallnorb":
            dataset_name = "dataset.name='smallnorb'"
            num_channels = "VAE.num_channels = 1"
        elif dataset == "mpi3d_toy":
            dataset_name = "dataset.name='mpi3d_toy'"
            num_channels = "VAE.num_channels = 3"
        elif dataset == "mpi3d_real":
            dataset_name = "dataset.name='mpi3d_real'"
            num_channels = "VAE.num_channels = 3"
        elif dataset == "mpi3d_realistic":
            dataset_name = "dataset.name='mpi3d_realistic'"
            num_channels = "VAE.num_channels = 3"
        elif dataset == "numpy_array":
            dataset_name = "dataset.name='numpy_array_data'"
            num_channels = f"VAE.num_channels = {args.num_channels}"
        elif dataset == "auto":
            dataset_name = "dataset.name='auto'"
            num_channels = "VAE.num_channels = 0"
        else:
            raise Exception(f"Dataset {dataset} not yet initialized")
        dataset_path = f"numpy_array_data.data_array_path = '{args.dataset_path}'"
        base_gin_bindings = [dataset_name, num_channels, dataset_path]
        if args.limited_gt_factors is not None:
            factors = str.split(args.limited_gt_factors, ",")
            base_gin_bindings += [f"{dataset}.latent_factor_indices = {list(map(int, factors))}"]


        # needs to have tfhub/model.pth and results/gin/train.gin files in there

        # By default, we do not overwrite output directories. Set this to True, if you
        # want to overwrite (in particular, if you rerun this script several times).
        overwrite = True

        if args.store_path == "":
            store_path = base_path
        else:
            store_path = args.store_path

        print("Loading cached dataset")

        gin.parse_config_files_and_bindings([], base_gin_bindings)
        if dataset == "auto":
            if args.udr:
                gin_config_file = os.path.join(base_path, str(args.udr_start), "model", "results", "gin",
                                               "train.gin")
            else:
                gin_config_file = os.path.join(base_path, "results", "gin",
                                           "train.gin")
            gin_dict = results.gin_dict(gin_config_file)
            with gin.unlock_config():
                if gin.query_parameter("dataset.name") == "auto":
                    gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
                        "'", ""))
        dataset_cache = named_data.get_named_ground_truth_data()
        gin.clear_config()
        print("finished loading cached dataset")

        if args.mig:
            gin_bindings = base_gin_bindings + [
                "evaluation.evaluation_fn = @nmig",
                "evaluation.random_seed = 0",
                "nmig.num_train=50000",
                "nmig.batch_size=10",
                "discretizer.discretizer_fn = @histogram_discretizer",
                "discretizer.num_bins = 20",
            ]

            eval_path = os.path.join(store_path, args.store_prefix, "mean", "mig")
            evaluate.evaluate_with_gin(
                base_path, eval_path, overwrite, gin_bindings=gin_bindings, type=type,
                dataset_cache=dataset_cache
            )

        if args.dci:
            gin_bindings = base_gin_bindings + [
                "evaluation.evaluation_fn = @dci",
                "evaluation.random_seed = 0",
                "dci.num_train=10000",
                "dci.num_test=5000",
                "discretizer.discretizer_fn = @histogram_discretizer",
                "discretizer.num_bins = 20",
            ]

            eval_path = os.path.join(store_path, args.store_prefix, "mean", "dci")
            evaluate.evaluate_with_gin(
                base_path, eval_path, overwrite, gin_bindings=gin_bindings, type=type,
                dataset_cache=dataset_cache
            )

        if args.pipe:
            gin_bindings = [
                "evaluation.evaluation_fn = @pipe",
                "evaluation.random_seed = 0",
                "evaluation.unsupervised = True",
                "pipe.num_samples=50000",
                "pipe.uniform = True",
                "pipe.num_runs=1",
                "pipe.compute_fid=True",
                dataset_name,
            ]
            eval_path = os.path.join(store_path, args.store_prefix, "mean", "pipe")
            evaluate.evaluate_with_gin(
                base_path, eval_path, overwrite, gin_bindings=gin_bindings, type=type,
                dataset_cache=dataset_cache
            )

        if args.factor_vae:
            gin_bindings = base_gin_bindings + [
                "evaluation.evaluation_fn = @factor_vae_score",
                "evaluation.random_seed = 0",
                "factor_vae_score.num_variance_estimate = 10000",
                "factor_vae_score.num_train = 10000",
                "factor_vae_score.num_eval = 5000",
                "factor_vae_score.batch_size = 64",
                "prune_dims.threshold = 0.05",
            ]
            eval_path = os.path.join(store_path, args.store_prefix, "mean", "factor_vae")
            evaluate.evaluate_with_gin(
                base_path, eval_path, overwrite, gin_bindings=gin_bindings, type=type,
                dataset_cache=dataset_cache
            )

        if args.fairness:
            gin_bindings = base_gin_bindings + [
                "evaluation.evaluation_fn = @fairness",
                "evaluation.random_seed = 0",
                "fairness.num_train=10000",
                "fairness.num_test_points_per_class=100",
                "predictor.predictor_fn = @gradient_boosting_classifier",
            ]
            eval_path = os.path.join(store_path, args.store_prefix, "mean", "fairness")
            evaluate.evaluate_with_gin(
                base_path, eval_path, overwrite, gin_bindings=gin_bindings, type=type,
                dataset_cache=dataset_cache
            )

        if args.udr:
            gin_bindings = base_gin_bindings + [
                "udr_sklearn.batch_size = 256",
                "udr_sklearn.num_data_points = 51200",
            ]
            udr_start = args.udr_start
            udr_end = args.udr_end
            for start in range(udr_start, udr_end, 50):
                model_dirs = [os.path.join(base_path, str(i), "model") for i in range(start, start+5)]
                store_dir = os.path.join(store_path, str(start))
                gin.parse_config_files_and_bindings(None, gin_bindings)
                evaluate_udr.evaluate(model_dirs, store_dir, udr.compute_udr_sklearn, 0, dataset_cache=dataset_cache)
                gin.clear_config()

