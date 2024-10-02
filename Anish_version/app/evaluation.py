import json
import datetime
import ray
from predict import TorchPredictor,get_best_checkpoint
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict
from collections import OrderedDict


import numpy as np
import typer
from typing_extensions import Annotated
from config import logger
import utils

# Initialize Typer CLI app
app = typer.Typer()



def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:  # pragma: no cover, eval workload
    """Get overall performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.

    Returns:
        Dict: overall metrics.
    """
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    overall_metrics = {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }
    return overall_metrics

def get_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_to_index: Dict) -> Dict:  # pragma: no cover, eval workload
    """Get per class performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        class_to_index (Dict): dictionary mapping class to index.

    Returns:
        Dict: per class metrics.
    """
    per_class_metrics = {}
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(class_to_index):
        per_class_metrics[_class] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    sorted_per_class_metrics = OrderedDict(sorted(per_class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True))
    return sorted_per_class_metrics



@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    dataset_loc: Annotated[str, typer.Option(help="dataset (with labels) to evaluate on")] = None,
    results_fp: Annotated[str, typer.Option(help="location to save evaluation results to")] = None,
) -> Dict:  # pragma: no cover, eval workload
    """Evaluate on the holdout dataset.

    Args:
        run_id (str): id of the specific run to load from. Defaults to None.
        dataset_loc (str): dataset (with labels) to evaluate on.
        results_fp (str, optional): location to save evaluation results to. Defaults to None.

    Returns:
        Dict: model's performance metrics on the dataset.
    """
    # Load
    ds = ray.data.read_csv(dataset_loc)
    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    # y_true
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    values = preprocessed_ds.select_columns(cols=["targets"]).take_all()
    y_true = np.stack([item["targets"] for item in values])

    # y_pred
    predictions = preprocessed_ds.map_batches(predictor).take_all()
    y_pred = np.array([d["output"] for d in predictions])

    # Metrics
    metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "overall": get_overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": get_per_class_metrics(y_true=y_true, y_pred=y_pred, class_to_index=preprocessor.class_to_index),
    }
    logger.info(json.dumps(metrics, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d=metrics, path=results_fp)
    return metrics


if __name__ == "__main__":  # pragma: no cover, checked during evaluation workload
    app()

