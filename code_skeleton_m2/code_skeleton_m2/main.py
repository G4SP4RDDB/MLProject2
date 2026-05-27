import argparse
import numpy as np

from src.activations import SoftMax
from src.losses import CrossEntropy
from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import Sigmoid, ReLU
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, onehot_to_label, label_to_onehot
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )



    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.
    means = np.mean(train_features, axis=0)
    stds = np.std(train_features, axis=0)

    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        val_size = int(0.2 * train_features.shape[0])
        x_tr = train_features[val_size:]
        x_val = train_features[:val_size]


        if args.task == "regression":
            y_tr = train_labels_reg[val_size:].reshape(-1, 1)
            y_val = train_labels_reg[:val_size].reshape(-1, 1)
        else:
            y_tr = label_to_onehot(train_labels_classif[val_size:], C=3)
            y_val = label_to_onehot(train_labels_classif[:val_size], C=3)


    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        ### WRITE YOUR CODE HERE
        pass

    elif args.method == "mlp":
        if args.task == "regression":
            method_obj = MLP(dimensions = (13, 64, 32, 1), activations = (ReLU, Sigmoid, ReLU))
        if args.task == "classification":
            method_obj = MLP(dimensions = (13, 64 ,32, 3), activations = (ReLU, Sigmoid, SoftMax))
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        if args.method == "mlp":
            method_obj.fit(x_tr, y_tr, loss=CrossEntropy, epochs= 400, batch_size=32, learning_rate=args.lr)
        else:
            pass # to add for K-means



    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method" #Only MLP for regression
        method_obj.fit(x_tr, y_tr,  loss = MSE, epochs = 400, batch_size = 32, learning_rate=args.lr)

    prediction = method_obj.predict(x_tr)

    print("Loss on TRAIN split: {}".format(MSE.loss(prediction, y_tr)))
    if(args.task == "classification"):
        pred_classes = onehot_to_label(prediction)
        true_classes = train_labels_classif[val_size:]
        print("Accuracy on TRAIN split: {}".format(accuracy_fn(pred_classes, true_classes)))

    print("------------------------")

    prediction = method_obj.predict(x_val)
    print("Loss on VALIDATION split: {}".format(MSE.loss(prediction, y_val)))
    if(args.task == "classification"):
        pred_classes = onehot_to_label(prediction)
        true_classes = train_labels_classif[:val_size]
        print("Accuracy on VALIDATION split: {}".format(accuracy_fn(pred_classes, true_classes)))



    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
