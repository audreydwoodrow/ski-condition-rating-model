# Ski condition rating model

A learning project built with Python and PyTorch: predict a skier's stated willingness-to-go rating from weather, resort conditions, and ticket price. The priority is understanding the calculations, from a hand-worked linear model to a small ReLU neural network.

This is a survey-based prototype, not a Tahoe run recommender. It does not predict closures, powder quality, safety, or actual enjoyment.

## Run a prediction

The repository includes a small trained model in `artifacts/relu_rating_model.json`. No dataset or retraining is needed to run it.

Tested with Python 3.13.7. From the repository folder on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python predict.py
```

If your environment is already set up, just activate it and run `python predict.py`.

Change the scenario using command-line arguments:

```bash
python predict.py --queue 5 --price 350 --temperature -5 --slopes-open 0.75 --weather SUN --wind "NO WIND" --day WEEKEND --period "REGULAR WEEK"
```

Run `python predict.py --help` for all options. Prices are Norwegian kroner, temperatures are Celsius, queue times are minutes, and slopes open is a fraction (0.75 means 75%). The survey included prices of 250–650 kroner, temperatures from −20 to +5°C, queues of 1/5/10 minutes, and slopes-open fractions of 0.5/0.75/1. Predictions for other values or unfamiliar combinations are extrapolations.

The output is a predicted rating on the survey's 0–100 scale, **not a probability or accuracy percentage**. The final layer is unbounded, so raw predictions can fall outside that scale; the program reports them without silently clipping.

## What happens during prediction?

1. Load the saved parameters and input-preparation rules.
2. Standardize four numeric inputs using saved training means and standard deviations.
3. Encode weather, wind, day type, and vacation period as 12 binary indicators.
4. Arrange the 16 numbers in the saved column order.
5. Perform one forward pass, with no gradients or parameter updates.

The model is `Linear(16, 8) → ReLU → Linear(8, 1)`: 145 learned parameters. Every hidden neuron sees all 16 inputs, using its own weights and bias. ReLU introduces bends; the output layer combines the eight hidden values into one rating.

## Data and held-out test set

Source: Erik Haugom, Iveta Malasevska, and Gudbrand Lien (2021), [The relative importance of ski resort- and weather-related characteristics when going alpine skiing: data from a rating-based conjoint survey](https://data.mendeley.com/datasets/6w4tzrs3yw/1), DOI 10.17632/6w4tzrs3yw.1. Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The model is derived from that dataset; cleaning, encoding, scaling, and training are this project's transformations. No author endorsement is implied.

Participants rated hypothetical skiing scenarios at Hafjell in Norway. These are individual stated preferences, not a record of real-day outcomes. Our downloaded spreadsheet has 7,182 rows. Excluding nonnumeric ratings leaves 6,490 ratings from 375 skiers; genuine zero ratings remain.

| Split | Skiers | Examples | Use |
|---|---:|---:|---|
| Training | 262 | 4,545 | Update parameters |
| Validation | 56 | 953 | Choose epochs and compare experiments |
| Test | 57 | 992 | Reserved for future final evaluation |

All responses from a skier stay in one split. A NumPy shuffle with seed 42 determines membership. Scaling is fitted only on training inputs. Test data has been prepared but **has not been used for training, model selection, or prediction-error evaluation**. Raw data and local split records stay out of Git.

To reproduce training, download `DATA.xlsx` from the source and place it in `data/DATA.xlsx`. The exact installed versions are listed in `requirements.txt`.

## Training and experiments

```bash
python worked_example.py       # Two-example walkthrough
python prepare_data.py         # Data preparation and linear-model training
python train_network.py        # Train the ReLU network
python investigate_factors.py  # 27 retraining comparisons
python export_model.py        # Replace the portable artifact with your trained network
```

Importing `prepare_data.py` prepares data and prints diagnostics, but does not train the linear model. Its name reflects the project's incremental teaching history; running it directly also trains the linear model.

Training uses batches of 32, mean squared error, and SGD. A batch's forward pass produces predictions; the backward pass calculates gradients; the optimizer updates parameters. There are 143 updates per epoch, including a final batch of one example. The ReLU experiment uses 100 epochs, learning rate 0.001, and seed 42. Hidden weights start randomly; the final bias starts at the training mean. The best validation snapshot is restored and verified after saving/reloading. This selects a checkpoint; it is not early stopping.

| Experiment | Best validation RMSE |
|---|---:|
| Constant training-mean prediction (45.85) | 30.5435 |
| Linear, learning rate 0.01, 10 epochs | 29.4598 |
| Linear, learning rate 0.001, 100 epochs | 29.5415 |
| Saved ReLU network, selected epoch 15 | 29.0250 |

RMSE is the square root of mean squared prediction error, expressed in rating points. Improvements are modest. Repeated validation selection is optimistic; these are not final test results. Training depends on initialization, shuffle sequence, numerical libraries, and hardware.

### Do all factors help?

We retrained with each factor made constant zero, hiding all columns for a categorical factor together. Three seeds (42, 43, 44) used the same architecture, matched initialization and batch order per seed, and the same split. The full-input mean best validation RMSE was 29.0389.

| Factor hidden | Mean increase in validation RMSE |
|---|---:|
| Temperature | +0.532 |
| Price | +0.356 |
| Weather | +0.307 |
| Slopes open | +0.217 |
| Queue time | +0.127 |
| Day type | +0.077 |
| Wind | +0.017 |
| Vacation period | +0.011 |

Positive means worse predictions without the factor. Wind and vacation changes were small and inconsistent across seeds. These comparisons measure additional predictive value given the other inputs, not causation or intrinsic importance. Related inputs can substitute for each other. No factors were permanently removed. The experiment resets its own shuffle generator, so its full-input runs differ from the original training script's sequence, which includes a batch-inspection pass.

## Saved files and stopping point

- `artifacts/relu_rating_model.json`: portable trained parameters, architecture, preprocessing, and validation metadata; included in Git so prediction works immediately.
- `models/`: local PyTorch checkpoints and experiment results; ignored by Git.
- `data/`: local survey spreadsheet and reserved split records; ignored by Git.
- `predict.py`: final command-line prediction program; never imports training data.

The JSON artifact contains learned parameters and aggregate preprocessing values, not individual survey responses. The test set is still reserved. Future work could evaluate the frozen model on that set, add more informative data, or inspect particular predictions. The model cannot explain all differences between people given conditions alone.
