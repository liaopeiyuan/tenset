# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name

"""Cost model based on xgboost"""
from collections import defaultdict
import logging
import multiprocessing
import pickle
import time
import numpy as np

from tvm.autotvm.tuner.metric import max_curve
from tvm.auto_scheduler.compute_dag import ComputeDAG
from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.feature import (
    get_per_store_features_from_measure_pairs, get_per_store_features_from_states)
from tvm.auto_scheduler.measure_record import RecordReader
from tvm.auto_scheduler.workload_registry import workload_key_to_tensors
from .cost_model import PythonBasedModel
from .xgb_model import XGBModelInternal
from .lgbm_model import LGBModelInternal
from .mlp_model import MLPModelInternal
from .tabnet_model import TabNetModelInternal

xgb = None

logger = logging.getLogger("auto_scheduler")


class EnsembleModel(PythonBasedModel):
    """Model Ensemble in end2end search."""
    def __init__(self, few_shot_learning="base_only", verbose_eval=25,
                 num_warmup_sample=100, seed=None, disable_update=False, mode="arithmetic_mean"):
        super().__init__()

        self.num_warmup_sample = num_warmup_sample
        self.disable_update = disable_update
        self.models = [ XGBModelInternal(few_shot_learning=few_shot_learning,
                                verbose_eval=verbose_eval,
                                seed=seed),
                        LGBModelInternal(few_shot_learning=few_shot_learning,
                                      verbose_eval=verbose_eval,
                                      seed=seed),
                    ]
        self.names = ["_XGB", "_LGBM"]
        self.dataset = Dataset()
        self.mode = mode

    def update(self, inputs, results):
        if self.disable_update or len(inputs) <= 0:
            return
        tic = time.time()
        self.dataset.update_from_measure_pairs(inputs, results)
        for model in self.models:
            model.fit_base(self.dataset)
        logger.info("Ensemble Training time: %.2f s", time.time() - tic)

    def predict(self, task, states):
        features = get_per_store_features_from_states(states, task)
        if self.models is not None and len(self.models) > 0 and len(self.dataset) > self.num_warmup_sample:
            learning_task = LearningTask(task.workload_key, str(task.target))
            eval_dataset = Dataset.create_one_task(learning_task, features, None)
            preds = []
            for model in self.models:
                preds.append(model.predict(eval_dataset)[learning_task])
            ret = sum(preds) / len(preds)
        else:
            ret = np.random.uniform(0, 1, (len(states),))

        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float('-inf')

        return ret

    def update_from_file(self, file_name, n_lines=None):
        """Load measure records from a log file to update the cost model.
        This function can be used to pre-train the cost model with history log files.
        Parameters
        ----------
        file_name: str
            The filename
        n_lines: Optional[int]
            Only load first n lines of the log file
        """
        inputs, results = RecordReader(file_name).read_lines(n_lines)
        logger.info("EnsembleModel: Loaded %s measurement records from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        """Save the model to a file
        Parameters
        ----------
        file_name: str
            The filename
        """
        for model, name in zip(self.models, self.names):
            model.save(file_name + name)

    def load(self, file_name: str):
        """Load the model from a file
        Parameters
        ----------
        file_name: str
            The filename
        """
        if self.models is None:
            self.models = [ XGBModelInternal(),
                            LGBModelInternal(),
                        ]
            self.names = ["_XGB", "_LGBM"]
        
        for i, name in zip(range(len(self.models)), self.names):
            self.models[i].load(file_name + name)
        self.num_warmup_sample = -1
