## Dependencies

```shell
pip install -r requirements.txt
```

Please specify the CUDA and CuDNN version for jax explicitly. If you are using CUDA 10.2 or order, you would also need to manually choose an older version of tf2jax and neural-tangents.

## Quick Start

Train and evaluate models through `utils/main.py`. For example, to train our model on Split CIFAR-100 with a 500 fixed-size buffer, one could execute:
```shell
python utils/main.py --model cgr --load_best_args --dataset seq-cifar100 --buffer_size 1000 --E 4 --seed 0 --cgr_diag_log --cgr_diag_dir cgr_diag_logs
```

```shell
python analyze_cgr_diag.py --diag_dir cgr_diag_logs --E 4 --buffer_size 1000 --last_k_for_margin 5
```

## Acknowledgement

Our implementation is based on [Mammoth](https://github.com/aimagelab/mammoth). We also refer to [InfluenceCL](https://github.com/feifeiobama/InfluenceCL), [Example_Influence_CL](https://github.com/SSSunQing/Example_Influence_CL), and [rethinking_er](https://github.com/hastings24/rethinking_er)
