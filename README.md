# Meta Learning

## Motivations

The goal of this project is to train a model that teaches another model some task.

The first model is learning how to update the weights each time it sees an example from the training set.
The second model is making predictions based.

At the end of training, we then have two models:
- a model that is able to make inference
- a meta-model that is able to update the weights of the first model when a new example is given to, allowing one-shot learning

## Models

### Prediction model

For the MVP, the model that predicts will be a very simple linear model.

### Meta model

The meta model is based on the Recurrent Neural Network architecture.
