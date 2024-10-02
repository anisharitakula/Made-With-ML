import ray.data
from predict import TorchPredictor
from ray.data import ActorPoolStrategy
import mlflow
from urllib.parse import urlparse
from ray.air import Result
import numpy as np
import torch

def get_best_checkpoint(run_id):
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path  # get path from mlflow
    results = Result.from_path(artifact_dir)
    return results.best_checkpoints[0][0]

def decode(indices, index_to_class):
    return [index_to_class[index] for index in indices]

class Predictor:
    def __init__(self,checkpoint):
        self.predictor=TorchPredictor.from_checkpoint(checkpoint)
        self.preprocessor=self.predictor.get_preprocessor()
    
    def __call__(self,batch):
        z=self.predictor(batch)['output']
        y_pred=np.stack(z)
        prediction = decode(y_pred,self.preprocessor.index_to_class)
        batch['prediction']=prediction
        return batch

if __name__=="__main__":
    
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    mlflow.set_tracking_uri("file:////tmp/mlflow")
    sorted_runs = mlflow.search_runs(search_all_experiments=True,order_by=["metrics.val_loss ASC"])
    best_run=sorted_runs.iloc[0]['run_id']
    best_checkpoint = get_best_checkpoint(run_id=best_run)
    predictor=TorchPredictor.from_checkpoint(best_checkpoint)



    ray.data.DatasetContext.get_current().execution_options.preserve_order = True  # deterministic
    HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
    test_ds = ray.data.read_csv(HOLDOUT_LOC)
    preprocessor=predictor.get_preprocessor()
    test_ds=preprocessor.transform(test_ds)
    
    test_ds=test_ds.map_batches(Predictor,batch_size=128,compute=ActorPoolStrategy(min_size=1, max_size=2),fn_constructor_kwargs={"checkpoint":best_checkpoint})

    pred=[]
    for row in test_ds.iter_rows():
        pred.append(row['prediction'])
    
    print(pred)


    # # Batch predict
    # predictions = test_ds.map_batches(
    # Predictor,
    # batch_size=128,
    # compute=ActorPoolStrategy(min_size=1, max_size=2),  # scaling
    # batch_format="pandas",
    # fn_constructor_kwargs={"checkpoint": best_checkpoint})

    # print("Finished the batch predict part")
    # Sample predictions



