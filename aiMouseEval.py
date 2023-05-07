import torch

from model import LSTM1
from targetWindow import run_eval

model_type = "lstm"


def evaluate(model, num_classes, input_size, hidden_size, num_layers, WIDTH, HEIGHT):
    if model is None:
        model = LSTM1(num_classes, input_size, hidden_size, num_layers, 1)
        model.load_state_dict(torch.load("mouseGUI.pt"))

    model.cpu()
    model.eval()

    run_eval(WIDTH, HEIGHT, model)


if __name__ == "__main__":
    input_size = 2
    hidden_size = 128
    num_layers = 1
    num_classes = 2

    evaluate(num_classes, input_size, hidden_size, num_layers)
