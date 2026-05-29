import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, SoftMax, Linear
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, onehot_to_label, label_to_onehot
from src.dataVisualisation import *
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
    PCA(train_features,2,train_labels_classif)

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        val_size = int(0.2 * train_features.shape[0])
        x_tr = train_features[val_size:]
        x_val = train_features[:val_size]


        if args.task == "regression":
            y_mean = train_labels_reg[val_size:].mean()
            y_std = train_labels_reg[val_size:].std()
            y_tr = ((train_labels_reg[val_size:] - y_mean) / y_std).reshape(-1, 1)
            y_val = ((train_labels_reg[:val_size] - y_mean) / y_std).reshape(-1, 1)
        else:
            y_tr = label_to_onehot(train_labels_classif[val_size:], C=3)
            y_val = label_to_onehot(train_labels_classif[:val_size], C=3)

    else:
        x_tr = train_features
        x_val = test_features

        if args.task == "regression":
            y_mean = train_labels_reg.mean()
            y_std = train_labels_reg.std()
            y_tr = ((train_labels_reg - y_mean) / y_std).reshape(-1, 1)
            y_val = ((test_labels_reg - y_mean) / y_std).reshape(-1, 1)
        else:
            y_tr = label_to_onehot(train_labels_classif, C=3)
            y_val = label_to_onehot(test_labels_classif, C=3)







    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        if args.task == "regression":
            if args.configuration == "simple":
                method_obj = MLP(dimensions = (13, 64, 32, 1), activations=(ReLU, Sigmoid, Linear))

            elif args.configuration == "deep":
                method_obj = MLP(dimensions=(13, 64, 32, 16, 1), activations=(ReLU, Sigmoid, ReLU, Linear))

            elif args.configuration == "wide":
                method_obj = MLP(dimensions=(13, 128, 64, 3), activations=(ReLU, Sigmoid, Linear))

        if args.task == "classification":
            if args.configuration == "simple":
                method_obj = MLP(dimensions = (13, 64 ,32, 3), activations = (ReLU, ReLU, SoftMax))

            elif args.configuration == "deep":
                method_obj = MLP(dimensions=(13, 64, 32, 16, 3), activations=(ReLU, ReLU, ReLU, SoftMax))

            elif args.configuration == "wide":
                method_obj = MLP(dimensions=(13, 128, 64, 3), activations=(ReLU, ReLU, SoftMax))
    else:
        raise ValueError(f"Unknown method: {args.method}")



    ## 4. Train and evaluate the method

    if args.task == "classification":
        if args.method == "mlp":
            #cross entropy with weights for classes
            counts = np.bincount(train_labels_classif.astype(int))
            class_weights = 1.0 / np.sqrt(counts) # to make weights not that heavy
            class_weights = class_weights / class_weights.sum()
            method_obj.fit(x_tr, y_tr, loss=CrossEntropy, epochs= args.epochs, batch_size=32, learning_rate=args.lr, class_weights=class_weights)
        elif args.method == "kmeans":
            method_obj.fit(x_tr, train_labels_classif)
        else:
            pass

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method" #Only MLP for regression
        method_obj.fit(x_tr, y_tr,  loss = MSE, epochs = args.epochs, batch_size = 32, learning_rate=args.lr)
       

    prediction = method_obj.predict(x_tr)

    if(args.task == "regression"):#denormalizing data
        prediction = prediction*y_std + y_mean
        y_tr = y_tr * y_std + y_mean
        print("Loss on TRAIN split: {}".format(MSE.loss(prediction, y_tr)))

    if(args.task == "classification"):
        if args.method == "kmeans":
            pred_classes = prediction
        else:
            pred_classes = onehot_to_label(prediction)


        if args.test:
            true_classes = train_labels_classif
        else:
            true_classes = train_labels_classif[val_size:]

        print("Accuracy on TRAIN split: {}".format(accuracy_fn(pred_classes, true_classes)))
        print("F1-micro score on VALIDATION split: {}".format(macrof1_fn(pred_classes, true_classes)))


    print("-" * 34)

    prediction = method_obj.predict(x_val)
    if(args.task == "regression"):#denormalizing data
        prediction = prediction*y_std + y_mean
        y_val= y_val * y_std + y_mean
        print("Loss on VALIDATION split: {}".format(MSE.loss(prediction, y_val)))

    if(args.task == "classification"):
        if args.method == "kmeans":
            pred_classes = prediction
        else:
            pred_classes = onehot_to_label(prediction)


        if args.test:
            true_classes = test_labels_classif
        else:
            true_classes = train_labels_classif[:val_size]

        print("Accuracy on VALIDATION split: {}".format(accuracy_fn(pred_classes, true_classes)))
        print("F1-micro score on VALIDATION split: {}".format(macrof1_fn(pred_classes, true_classes)))


    ### Graphs and hyper parameters choosing

    if args.task == "classification" and args.method == "kmeans":
        print(f"{'K':>2} | {'avg val acc':>10}")
        print("-" * 34)

        for k in range(1, args.K):
            val_accs = []

            for _ in range(30):
                method_obj = KMeans(K=k, max_iters=args.max_iters)
                method_obj.fit(x_tr, train_labels_classif[val_size:])

                val_pred = method_obj.predict(x_val)
                val_acc = accuracy_fn(val_pred, train_labels_classif[:val_size])
                val_accs.append(val_acc)
        avg_val_acc = np.mean(val_accs)
        print(f"{k:2d} | {avg_val_acc:10.4f}")

    #graph for mlp losses by epochs
    if args.task == "classification" and args.method == "mlp":
        plt.figure()
        plt.plot(range(10, args.epochs + 1, 10), method_obj.train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig("loss_curve.png")
        plt.show()



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
        default=1e-3,
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
    parser.add_argument(
        "--configuration",
        type=str,
        default = "simple",
        help="simple / wide / deep",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default = 400,
        help="epochs for methods which are iterative",
    )


    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
