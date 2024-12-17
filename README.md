# Quant Strat Behavior Cloning

In this project, we generate trajectories from simple quantitative strategies (e.g. SMA, EWM, BBand, RSI, etc.) and train decision transformers to learn the behavior of these strategies. The goal is to learn the trading actions of these strategies and perform trajectory stitching to generate a comprehensive trading strategy.

## Data Generation

Please refer to another repository [GenStrat](https://github.com/AbnerTeng/GenStrat) to generate synthetic expert trajectories

## Training

To train the decision transformer, run the following command:

```bash
python -m src.main --expr univ_log --discrete --model edt --mode train --gpu 1
```

## Evaluation

Just change the `--mode train` to `--mode test` to evaluate the trained model.

