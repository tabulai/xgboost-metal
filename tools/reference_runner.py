"""Run XGBoost CPU compatibility checks in a separate process."""
from pathlib import Path
import argparse
import json
import numpy as np
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", type=Path)
args = parser.parse_args()

reference = Path(__file__).resolve().parents[1] / "ref/xgboost"
assert Path(xgb.__file__).resolve().is_relative_to(reference), xgb.__file__
assert xgb.__version__ == "3.5.0-dev"
data = np.load(args.data)
config = json.loads(args.config.read_text())
def dataframe(values, schema):
    if schema is None:
        return values
    import pandas as pd
    columns = {}
    for i, name in enumerate(schema["columns"]):
        column = schema["categories"][i]
        if column is None:
            columns[name] = values[:, i]
        else:
            codes = np.where(np.isnan(values[:, i]), -1, values[:, i]).astype(np.int64)
            columns[name] = pd.Categorical.from_codes(codes, pd.Index(column["names"], dtype=column["dtype"]))
    return pd.DataFrame(columns)

meta = {k: data[k] for k in ("label", "weight", "base_margin", "label_lower_bound", "label_upper_bound", "group", "qid", "feature_weights") if k in data}
feature_types = config.get("feature_types")
matrix_class = xgb.QuantileDMatrix if config.get("quantile") else xgb.DMatrix
matrix_options = {"max_bin": config["params"].get("max_bin", 256)} if config.get("quantile") else {}
matrix_options["missing"] = config.get("missing", np.nan)
train_types = feature_types
if config.get("category_reference"):
    assert args.model is not None
    category_owner = xgb.Booster(model_file=args.model)
    train_types = category_owner.get_categories()
dtrain = matrix_class(dataframe(data["X"], config.get("train_frame")), feature_types=train_types, enable_categorical=True, **matrix_options, **meta)
test_meta = {"base_margin": data["test_margin"]} if "test_margin" in data else {}
test_config = config.get("test_matrix") or {}
if "reference_feature_types_after" in test_config:
    dtrain.feature_types = test_config["reference_feature_types_after"]
test_class = xgb.QuantileDMatrix if test_config.get("quantile") else xgb.DMatrix
test_options = {"max_bin": test_config.get("max_bin", config["params"].get("max_bin", 256))} if test_config.get("quantile") else {}
test_options["missing"] = test_config.get("missing", config.get("missing", np.nan))
if test_config.get("ref"):
    test_options["ref"] = dtrain
for field in ("weight", "group", "qid"):
    if "test_" + field in data:
        test_meta[field] = data["test_" + field]
test_types = test_config.get("feature_types", feature_types)
if test_config.get("category_reference"):
    category_owner = xgb.Booster(model_file=args.model) if args.model is not None else dtrain
    test_types = category_owner.get_categories()
dtest = test_class(dataframe(data["test"] if "test" in data else data["X"], config.get("test_frame")),
                    feature_types=test_types, enable_categorical=True, **test_options, **test_meta)
history = {}
custom = (lambda prediction, d: (prediction - d.get_label(), np.ones_like(prediction))) if config.get("custom_square") else None
if "grad" in data:
    custom = lambda prediction, d: (data["grad"], data["hess"])
if "projection" in data:
    from xgboost.objective import TreeObjective
    base_custom = custom or (lambda prediction, d: (
        prediction - d.get_label().reshape(prediction.shape), np.ones_like(prediction)))
    class ReducedObjective(TreeObjective):
        def __call__(self, iteration, prediction, matrix):
            gradient, hessian = base_custom(prediction, matrix)
            return gradient.reshape(matrix.num_row(), -1), hessian.reshape(matrix.num_row(), -1)
        def split_grad(self, iteration, gradient, hessian):
            return gradient @ data["projection"], hessian @ np.square(data["projection"])
    custom = ReducedObjective()
class ParameterUpdates(xgb.callback.TrainingCallback):
    def before_iteration(self, model, epoch, evals_log):
        update = (config.get("param_updates") or {}).get(str(epoch))
        if update is not None:
            model.set_param(update)
        for field, values in (config.get("matrix_updates") or {}).get(str(epoch), {}).items():
            if field in ("feature_types", "feature_names"):
                setattr(dtrain, field, values)
            else:
                dtype = np.uint32 if field in ("group", "qid") else np.float32
                dtrain.set_info(**{field: np.asarray(values, dtype=dtype)})
        return False
callbacks = [ParameterUpdates()] if config.get("param_updates") or config.get("matrix_updates") else None
if args.model:
    model = xgb.Booster(model_file=args.model)
    if config.get("continue_rounds"):
        model = xgb.train({"tree_method": "hist", **config["params"]}, dtrain, config["continue_rounds"],
                          xgb_model=model, evals=[(dtrain, "train")], evals_result=history, verbose_eval=False, obj=custom, callbacks=callbacks)
else:
    params = {"tree_method": "hist", **config["params"]}
    model = xgb.train(params, dtrain, config.get("rounds", 5), evals=[(dtrain, "train")],
                      evals_result=history, verbose_eval=False, obj=custom, callbacks=callbacks)
if config.get("post_params"):
    model.set_param(config["post_params"])
if "feature_types_after" in test_config:
    dtest.feature_types = test_config["feature_types_after"]
args.output.mkdir(parents=True, exist_ok=True)
if config.get("cuts"):
    indptr, values = dtrain.get_quantile_cut()
    np.savez(args.output / "cuts.npz", indptr=indptr, values=values)
if config.get("matrix_data"):
    arrays, info = {}, {}
    for name, matrix in [("train", dtrain), ("test", dtest)]:
        csr = matrix.get_data()
        arrays.update({name + "_indptr": csr.indptr, name + "_indices": csr.indices, name + "_data": csr.data})
        info[name] = {"feature_types": matrix.feature_types, "feature_names": matrix.feature_names,
                      "nonmissing": matrix.num_nonmissing()}
        if isinstance(matrix, xgb.QuantileDMatrix) or name == "train" and config.get("rounds", 5):
            ptr, values = matrix.get_quantile_cut()
            arrays[name + "_cut_ptr"], arrays[name + "_cuts"] = ptr, values
    np.savez(args.output / "matrix-data.npz", **arrays)
    (args.output / "matrix-info.json").write_text(json.dumps(info))
if config.get("evaluation"):
    (args.output / "evaluation.txt").write_text(model.eval(dtrain))
predictions = dict(prediction=model.predict(dtest),
         margin=model.predict(dtest, output_margin=True),
         contributions=model.predict(dtest, pred_contribs=True))
if config.get("category_reference"):
    predictions["training_prediction"] = model.predict(dtrain)
    predictions["training_leaf"] = model.predict(dtrain, pred_leaf=True)
prediction_errors = {}
learner_config = json.loads(model.save_config())["learner"]
linear = learner_config["gradient_booster"]["name"] == "gblinear"
vector_strategy = learner_config["learner_train_param"].get("multi_strategy") == "multi_output_tree"
vector = False
if not linear:
    booster_model = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]
    tree_model = booster_model.get("gbtree", booster_model)["model"]
    vector = any(int(tree["tree_param"]["size_leaf_vector"]) > 1 for tree in tree_model["trees"])
for name, request in {
    "approximate": {"pred_contribs": True, "approx_contribs": True},
    "leaf": {"pred_leaf": True},
    "interactions": {"pred_interactions": True},
    "approximate_interactions": {"pred_interactions": True, "approx_contribs": True},
}.items():
    try:
        predictions[name] = model.predict(dtest, **request)
    except xgb.core.XGBoostError as error:
        unsupported_vector = ((vector and name in ("approximate", "approximate_interactions")) or
                              (vector_strategy and name == "approximate_interactions")) and "multi-target" in str(error)
        if not linear and not unsupported_vector:
            raise
        prediction_errors[name] = str(error)
(args.output / "prediction_errors.json").write_text(json.dumps(prediction_errors))
for i, request in enumerate(config.get("prediction_requests") or []):
    predictions[f"request_{i}"] = model.predict(dtest, **request)
for i, step in enumerate(config.get("prediction_steps") or []):
    if step.get("reset"):
        model.reset()
    if "params" in step:
        model.set_param(step["params"])
    if "feature_types" in step:
        dtest.feature_types = step["feature_types"]
    if "base_margin" in step:
        dtest.set_base_margin(np.asarray(step["base_margin"], np.float32))
    predictions[f"step_{i}"] = model.predict(dtest, **step.get("predict", {}))
np.savez(args.output / "predictions.npz", **predictions)
model.save_model(args.output / "model.json")
model.save_model(args.output / "model.ubj")
(args.output / "config.json").write_text(model.save_config())
try:
    dump = model.get_dump(dump_format="json", with_stats=True)
except xgb.core.XGBoostError as error:
    changed_types = any("feature_types" in update for update in (config.get("matrix_updates") or {}).values())
    if not changed_types or "in feature map is" not in str(error) or "but the tree node is" not in str(error):
        raise
    (args.output / "dump-error.txt").write_text(str(error))
else:
    (args.output / "dump.json").write_text(json.dumps(dump))
(args.output / "history.json").write_text(json.dumps(history))
