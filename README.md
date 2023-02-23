# A Collection of Knowledge Graph Embedding Methods (KGE).

## Quick Start
### Process datasets
1. First, run the following command to preprocess the datasets.
```
./run.sh configs/<dataset>.sh --process_data <gpu-ID>
```
`<dataset>` is one of datasets, such as: `wn18rr`, `fb15k-237`, and `nell995`.
`<gpu-ID>` is a non-negative integer number representing the GPU index.

2. For example:
```
./run.sh configs/wn18rr.sh --process_data 0
```

### Train models
1. The following commands can be used to train a KGE model. By default, dev set evaluation results will be printed when training terminates.

```
./run.sh configs/<dataset>-<KGE model>.sh --train <gpu-ID>
```
`<KGE model>` is one of KGE models, such as: `TransE`, `ConvE`, and `TuckER`.

2. For example:
```
./run.sh configs/wn18rr-convE.sh --train 0
```

### Evaluate models
To generate the evaluation results of a pre-trained model, simply change the `--train` flag in the commands above to `--inference`. 

For example, the following command performs inference and prints the evaluation results (on both dev and test sets).
```
./run.sh configs/<dataset>-<KGE model>.sh --inference <gpu-ID>
```

### Add new models
To add a new KGE model to the program, simple add the model's code to [models.py](src/models.py).

### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the [configuration files](configs) and [parse_args.py](src/parse_args.py).
