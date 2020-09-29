# PVPower Implementation

This repository supplements the paper *TODO* and can be used in order to verify our experiments. You are free to use this in your own experiments. However, please make sure to cite us correctly.


## Training

In order to use this, you need to download the dataset, which is publicly available under [doi:10.26165/JUELICH-DATA/TVWUUP](https://doi.org/10.26165/JUELICH-DATA/TVWUUP). Please make sure to place it in the `data` subfolder, such that the `data.csv` is available as `data/data.csv`.

We provide a common entrypoint `train.py` that can be used to conduct most of the experiments reported in the paper. The default hyperparameters are set such that they correspond to the ones reported in the paper. Hence, you can for example train the first fold of the crossvalidation by issuing:

```
python train.py --cv_fold_id=0 --jobname=cv_fold_0
```

Please refer to the documentation for further options:

```
python train.py --help
```


## Inference

We include our trained models as well and you may directly use them for inference on new data. Given, you've some sample images in `my_data`, you may run the model trained in the first fold of the crossvalidation as follows:

```
python inference.py --checkpoint_path=models_release/crossval/model_fold_0.ckpt --data_path=my_data --target_path=my_data/results
```

Or you can use the model that calculates the per-cell power loss using class activation maps like this:

```
python inference.py --checkpoint_path=models_release/cam/model.ckpt --data_path=my_data --target_path=my_data/results
```

For more options, please refer to

```
python inference.py --help
```


## References

*TODO*