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

"""Evaluation protocol to compute metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import time
import warnings

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import fairness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import stability  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import strong_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unified_scores  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import pipe_metric  # pylint: disable=unused-import
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from scipy import stats

device = 0 if tf.test.is_gpu_available() else "cpu"

import gin.tf


def evaluate_with_gin(model_dir,
                      output_dir,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None,
                      model_type="tf",
                      dataset_cache=None):
    """Evaluate a representation based on the provided gin configuration.

  This function will set the provided gin bindings, call the evaluate()
  function and clear the gin config. Please see evaluate() for required
  gin bindings.

  Args:
    model_dir: String with path to directory where the representation is saved.
    output_dir: String with the path where the evaluation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
    model_type: "tf" if a tensorflow from disentanglement_lib gets evaluated, "dava" for a pytorch model
    dataset_cache: cached GroundTruthData
  """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    evaluate(model_dir, output_dir, model_type, overwrite, dataset_cache=dataset_cache)
    gin.clear_config()


@gin.configurable(
    "evaluation", blacklist=["model_dir", "output_dir", "overwrite"])
def evaluate(model_dir,
             output_dir,
             model_type,
             overwrite=False,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             unsupervised=False,
             name="",
             dataset_cache=None):
    """Loads a representation TFHub module and computes disentanglement metrics.

  Args:
    model_dir: String with path to directory where the representation function
      is saved.
    output_dir: String with the path where the results should be saved.
    model_type: "tf" if a tensorflow from disentanglement_lib gets evaluated, "dava" for a pytorch model
    overwrite: Boolean indicating whether to overwrite output directory.
    evaluation_fn: Function used to evaluate the representation (see metrics/
      for examples).
    random_seed: Integer with random seed used for training.
    unsupervised: Flag to tell if evaluation_fn is unsupervised and needs decoder as input
    name: Optional string with name of the metric (can be used to name metrics).
    dataset_cache: cached GroundTruthData
  """
    # Delete the output directory if it already exists.
    if tf.gfile.IsDirectory(output_dir):
        if overwrite:
            tf.gfile.DeleteRecursively(output_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()

    # Automatically set the proper data set if necessary. We replace the active
    # gin config as this will lead to a valid gin config file where the data set
    # is present.

    # Obtain the dataset name from the gin config of the previous step.

    gin_config_file = os.path.join(model_dir, "results", "gin",
                                   "train.gin")
    gin_dict = results.gin_dict(gin_config_file)
    with gin.unlock_config():
        if gin.query_parameter("dataset.name") == "auto":
            gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
                "'", ""))
        if gin.query_parameter("dataset.name") == "reduced_dsprites_cont":
            gin.bind_parameter("reduced_dsprites_cont.seed",
                               int(gin_dict["reduced_dsprites_cont.seed"].replace(
                                   "'", "")))
            gin.bind_parameter("reduced_dsprites_cont.train_split",
                               float(gin_dict["reduced_dsprites_cont.train_split"].replace(
                                   "'", "")))
    print("start loading dataset")
    if dataset_cache is not None:
        dataset = dataset_cache
    else:
        dataset = named_data.get_named_ground_truth_data()
    print("dataset loaded")

    if model_type == "dava":
        z_dim = gin_dict["VAE.z_dim"].replace("'", "")
        if "%" in z_dim:
            z_dim = gin_dict["z_dim"].replace("'", "")
            gin.bind_parameter("pipe.z_dim", int(z_dim))
        num_channels = int(gin_dict["VAE.num_channels"].replace("'", ""))
        z_dim = int(z_dim)

        import torch
        import dava.dislibvae
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        checkpoint_vae = torch.load(os.path.join(model_dir, "model.pth"), map_location=device)

        use_batch_norm = False
        use_layer_norm = False
        use_instance_norm = False
        if "use_batch_norm" in checkpoint_vae.keys():
            use_batch_norm = checkpoint_vae["use_batch_norm"]
            use_layer_norm = checkpoint_vae["use_layer_norm"]
            use_instance_norm = checkpoint_vae["use_instance_norm"]
        elif "encoder.bn1.running_mean" in checkpoint_vae.keys() and checkpoint_vae["encoder.bn1.running_mean"].any():
            if "encoder.bn2.running_mean" in checkpoint_vae.keys():
                assert len(checkpoint_vae["bn1.bias"].shape) == 1
                use_batch_norm = True
            else:
                use_instance_norm = True
        elif "encoder.bn1.bias" in checkpoint_vae.keys() and checkpoint_vae["encoder.bn1.bias"].std() > 0:
            assert len(checkpoint_vae["bn1.bias"].shape) == 3
            use_layer_norm = True
        #  vae.decoder.use_batch_norm = True
        # We can load all models as BetaTCVAE as loss function doesn't matter and architecture is the same
        vae = dava.models.BetaTCVAE(z_dim, num_channels, 1, use_batch_norm=use_batch_norm,
                                    use_layer_norm=use_layer_norm, use_instance_norm=use_instance_norm)

        vae.load_state_dict(checkpoint_vae, strict=False)
        vae.to(device)
        vae.eval()

        if unsupervised:
            def _encode(x):
                """Computes representation vector for input images."""

                # change numpy array samples such that it is [batch, channels, x, y]
                x = np.moveaxis(x, 3, 1)
                x = torch.from_numpy(x).to(device)

                mean, logvar = vae.encode(x)

                return mean.cpu().detach().numpy(), logvar.cpu().detach().numpy()

            def _decode(latent_vectors):
                latent_vectors = torch.from_numpy(latent_vectors).to(device)
                latent_vectors = latent_vectors.type(dtype=torch.float32)
                xs = vae.decode(latent_vectors)
                xs = np.moveaxis(xs.cpu().detach().numpy(), 1, 3)
                return xs

            def _reconstruct(x):
                return _decode(_encode(x)[0])

            results_dict = evaluation_fn(
                dataset,
                _encode,
                _decode,
                _reconstruct,
                random_state=np.random.RandomState(random_seed)
            )
        else:
            def _encode(x):
                """Computes representation vector for input images."""

                # change numpy array samples such that it is [batch, channels, x, y]
                x = np.moveaxis(x, 3, 1)
                x = torch.from_numpy(x).to(device)

                mean, logvar = vae.encode(x)

                return mean.cpu().detach().numpy()

            results_dict = evaluation_fn(
                dataset,
                _encode,
                random_state=np.random.RandomState(random_seed))

    else:
        # Path to TFHub module of previously trained representation.
        module_path = os.path.join(model_dir, "tfhub")
        with hub.eval_function_for_module(module_path) as f:

            if unsupervised:
                def _encoder(x):
                    """Computes representation vector for input images."""
                    output_sampled = f(dict(images=x), signature="gaussian_encoder", as_dict=True)
                    return np.array(output_sampled["mean"]), np.array(output_sampled["logvar"])

                def sigmoid(x):
                    return stats.logistic.cdf(x)

                def _decoder(latent_vectors):
                    output = f(
                        dict(latent_vectors=latent_vectors),
                        signature="decoder",
                        as_dict=True)
                    return sigmoid(np.array(output["images"]))  # for some reason we need the activation function here

                def _reconstruct(x):
                    output = f(dict(images=x), signature="reconstructions", as_dict=True)
                    return sigmoid(np.array(output["images"]))

                results_dict = evaluation_fn(
                    dataset,
                    _encoder,
                    _decoder,
                    _reconstruct,
                    random_state=np.random.RandomState(random_seed)
                )
            else:
                def _representation_function(x):
                    """Computes representation vector for input images."""
                    output = f(dict(images=x), signature="representation", as_dict=True)
                    return np.array(output["default"])

                # Computes scores of the representation based on the evaluation_fn.
                if _has_kwarg_or_kwargs(evaluation_fn, "artifact_dir"):
                    artifact_dir = os.path.join(model_dir, "artifacts", name)
                    results_dict = evaluation_fn(
                        dataset,
                        _representation_function,
                        random_state=np.random.RandomState(random_seed),
                        artifact_dir=artifact_dir)
                else:
                    # Legacy code path to allow for old evaluation metrics.
                    warnings.warn(
                        "Evaluation function does not appear to accept an"
                        " `artifact_dir` argument. This may not be compatible with "
                        "future versions.", DeprecationWarning)

                    results_dict = evaluation_fn(
                        dataset,
                        _representation_function,
                        random_state=np.random.RandomState(random_seed))

    # Save the results (and all previous results in the pipeline) on disk.
    original_results_dir = os.path.join(model_dir, "results")
    results_dir = os.path.join(output_dir, "results")
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(results_dir, "evaluation", results_dict,
                                    original_results_dir)


def _has_kwarg_or_kwargs(f, kwarg):
    """Checks if the function has the provided kwarg or **kwargs."""
    # For gin wrapped functions, we need to consider the wrapped function.
    if hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    (args, _, kwargs, _, _, _, _) = inspect.getfullargspec(f)
    if kwarg in args or kwargs is not None:
        return True
    return False
