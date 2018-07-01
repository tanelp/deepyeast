import keras
import argparse
import numpy as np
from sklearn import metrics

from deepyeast.dataset import load_data
from deepyeast.utils import preprocess_input

def parse_args():
    available_models = ['deepyeast', 'resnet', 'mobilenet', 'densenet']
    parser = argparse.ArgumentParser(description="Evaluate model's performance.")
    parser.add_argument('model', help="Specification of the model's structure",
                        choices=available_models)
    parser.add_argument('weights', help="Path to weights file.")
    parser.add_argument('--split', nargs='+', help='', default=['test', 'val', 'train'])
    args = parser.parse_args()
    return args

def evaluate(split):
    print('Loading {} set...'.format(split))
    x, y_true = load_data(split)
    x = preprocess_input(x)
    y_pred = model.predict(x)
    y_pred = y_pred.argmax(axis=1)
    print("{} set statistics:".format(split))
    print("Top-1-accuracy: {:.4f}".format(np.mean(y_true==y_pred)))
    print(metrics.classification_report(y_true, y_pred))

def load_model(model_name):
    if model_name == 'deepyeast':
        from deepyeast.models import DeepYeast
        model = DeepYeast()
    elif model_name == 'resnet':
        from deepyeast.models import ResNet50
        model = ResNet50()
    elif model_name == 'mobilenet':
        from deepyeast.models import MobileNet
        model = MobileNet()
    elif model_name == 'densenet':
        from deepyeast.models import DenseNet40_BC
        model = DenseNet40_BC()
    return model

if __name__ == '__main__':
    num_classes = 12

    args = parse_args()
    print(args)

    print('Loading model...')
    model = load_model(args.model)
    model.load_weights(args.weights)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy', 'accuracy'])

    if 'train' in args.split:
        evaluate('train')
    if 'val' in args.split:
        evaluate('val')
    if 'test' in args.split:
        evaluate('test')
