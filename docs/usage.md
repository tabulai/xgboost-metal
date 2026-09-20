# Usage

Import the package as `xgboost`. The examples below assume that training and
test arrays have already been prepared. Use `device="metal"` and
`tree_method="hist"` to make the execution target explicit.

## Regression and classification

```python
import xgboost as xgb

regressor = xgb.XGBRegressor(
    device="metal", tree_method="hist", n_estimators=100, max_depth=6
)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

classifier = xgb.XGBClassifier(
    device="metal", tree_method="hist", n_estimators=100, max_depth=6
)
classifier.fit(X_train, class_labels)
probabilities = classifier.predict_proba(X_test)
```

The scikit-learn estimators require the `sklearn` extra. For direct control of
training and evaluation, use the core API:

```python
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_valid, label=y_valid)
booster = xgb.train(
    {"device": "metal", "tree_method": "hist", "objective": "reg:squarederror"},
    train,
    num_boost_round=200,
    evals=[(valid, "validation")],
    early_stopping_rounds=20,
    verbose_eval=False,
)
predictions = booster.predict(
    valid, iteration_range=(0, booster.best_iteration + 1)
)
```

## Data and categorical features

Dense NumPy arrays, SciPy sparse matrices, pandas dataframes, and Arrow inputs
are accepted. Missing values use `NaN` by default; pass `missing` to select a
different marker. Sparse inputs are expanded to dense storage during training,
so account for the dense size when planning memory use.

For pandas categorical columns, keep the column dtype as `category` and enable
categorical handling:

```python
model = xgb.XGBClassifier(
    device="metal", tree_method="hist", enable_categorical=True
)
model.fit(categorical_frame, class_labels)
model.save_model("categorical-model.ubj")
```

Use `DMatrix(..., weight=row_weights)` for sample weights. Ranking weights have
one value per query, rather than one value per row.

`QuantileDMatrix` accepts a training reference for validation or test data. Use
the same `max_bin` in the matrix and training parameters:

```python
train = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=256)
valid = xgb.QuantileDMatrix(X_valid, label=y_valid, max_bin=256, ref=train)
```

## Multiple outputs and quantiles

For a shared tree structure across regression outputs, supply a label matrix
with one column per target:

```python
model = xgb.XGBRegressor(multi_strategy="multi_output_tree", max_depth=6)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
```

For multiple quantiles, pass sorted levels:

```python
model = xgb.XGBRegressor(
    objective="reg:quantileerror", quantile_alpha=[0.1, 0.5, 0.9]
)
model.fit(X_train, y_train)
quantiles = model.predict(X_test)
```

`reg:expectileerror` supports ordered `expectile_alpha` levels. `reg:normal`
predicts a mean and log variance for a single response.

## Ranking and survival

For ranking, sort rows by query ID and provide `qid`:

```python
ranker = xgb.XGBRanker(objective="rank:ndcg", n_estimators=100)
ranker.fit(X_train, relevance, qid=query_ids)
scores = ranker.predict(X_test)
```

`rank:map` and `rank:pairwise` are also supported. NDCG uses integer relevance
levels from 0 through 31 by default; use `ndcg_exp_gain=False` for custom gains.

For accelerated failure time models, provide censoring bounds:

```python
survival = xgb.DMatrix(
    X_train, label_lower_bound=lower, label_upper_bound=upper
)
model = xgb.train(
    {"objective": "survival:aft", "aft_loss_distribution": "normal"},
    survival,
    num_boost_round=100,
)
```

Equal bounds identify observed events. An infinite upper bound represents right
censoring. Normal, logistic, and extreme-value distributions are supported.
For `survival:cox`, labels use positive event times and negative censored times.

## Models and continued training

JSON and UBJSON preserve the model for later loading:

```python
booster.save_model("model.ubj")
restored = xgb.Booster(model_file="model.ubj")
predictions = restored.predict(xgb.DMatrix(X_test))

continued = xgb.train(
    {"objective": "reg:squarederror"},
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=20,
    xgb_model=restored,
)
```

Ordinary `DMatrix` data can be saved with `save_binary()` and reloaded by
filename. Quantile and iterator matrices do not support binary saving.

Feature contributions and interactions are available through
`Booster.predict(..., pred_contribs=True)` and
`Booster.predict(..., pred_interactions=True)`. Contributions describe raw
margins; they sum to the raw prediction, including the bias term.

See [compatibility](compatibility.md) for booster-specific restrictions and
[benchmarking](benchmarking.md) for performance measurements.
