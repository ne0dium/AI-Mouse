from aiMouseEval import evaluate
from aiMouseTrain import train
from targetWindow import run

if __name__ == "__main__":
    WIDTH, HEIGHT = 1920, 1080

    num_epochs = 400
    learning_rate = 0.001

    input_size = 2
    hidden_size = 128
    num_layers = 1
    num_classes = 2

    seq = 10

    pointsList = run(WIDTH, HEIGHT)

    model = train(pointsList, num_epochs, learning_rate, input_size, hidden_size, num_layers, num_classes, seq, WIDTH,
                  HEIGHT, model_type="lstm")

    evaluate(model, num_classes, input_size, hidden_size, num_layers, WIDTH, HEIGHT)
