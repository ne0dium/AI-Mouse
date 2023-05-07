from aiMouseEval import evaluate
from aiMouseTrain import train
from targetWindow import run
import torch

if __name__ == "__main__":
    WIDTH, HEIGHT = 1920, 1080

    num_epochs = 400
    learning_rate = 0.001

    input_size = 2
    hidden_size = 128
    num_layers = 1
    num_classes = 2

    seq = 10

    circle_radius = 10
    circle_speed = 2

    pointsList = run(WIDTH, HEIGHT, circle_radius, circle_speed)

    model = train(pointsList, num_epochs, learning_rate, input_size, hidden_size, num_layers, num_classes, seq, WIDTH,
                  HEIGHT, model_type="lstm")

    evaluate(model, num_classes, input_size, hidden_size, num_layers, WIDTH, HEIGHT, circle_radius, circle_speed)

    # If you want to save the model
    torch.save(model.state_dict(), "mouse.pt")