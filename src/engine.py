import argparse
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pandas as pd
import pickle
import tensorflow as tf
from ML_Pipeline.cluster import ClusterIntents
from ML_Pipeline.eda import ExploreData
from ML_Pipeline.evaluation import Eval
from ML_Pipeline.predict import Predict
from ML_Pipeline.prepare import PrepareData
from ML_Pipeline.train import ModelTrain
from ML_Pipeline.utils import parse_files, parse_data_csv
from tensorflow.keras.models import load_model

PATH = os.getcwd()
PARENT_PATH = os.path.dirname(PATH)
DATA_DIR = f'{PARENT_PATH}/input/chat_logs'
LABEL_FILE = f'{PARENT_PATH}/input/unsupervised_labeled_data.csv'
LABEL_FILE_POST = f'{PARENT_PATH}/input/supervised_labeled_data.csv'
INTENT_FILE = f'{PARENT_PATH}/input/intents.json'
MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
MODEL_PATH = f'{PARENT_PATH}/outputs/saved_model_keras'
MODEL_HISTORY = f'{PARENT_PATH}/outputs/HistoryDict'

RUN_TRAIN_PREDICT = False

parser = argparse.ArgumentParser()
parser.add_argument("--chat", action='store_true',
                    help="User chat")
parser.add_argument("--cluster", action='store_true',
                    help="Preprocess and cluster raw chat logs")
parser.add_argument("--train", action='store_true',
                    help="Set train option")

# eda args
parser.add_argument("--eda", action='store_true',
                    help="Enable to perform exploratory data analysis")
parser.add_argument("--eda_token_dist", action='store_true',
                    help="Get distribution of token frequency")
parser.add_argument("--eda_plot_token_dist", action='store_true',
                    help="Plot distribution of token frequency")
parser.add_argument("--eda_top_n_tokens", type=int,
                    help="Get n tokens ordered by highest frequency")
parser.add_argument("--eda_tokens_hist", action='store_true',
                    help="Plot token length histogram")
parser.add_argument("--eda_sent_hist", action='store_true',
                    help="Plot sentence length histogram")

# clustering args
parser.add_argument("--cluster_best_params", action='store_true',
                    help="View best tuned parameters")
parser.add_argument("--cluster_plot", action='store_true',
                    help="Plot cluster space")
parser.add_argument("--cluster_labels_summary", action='store_true',
                    help="Get slice of table with labels information")
parser.add_argument("--cluster_labeled_utts", action='store_true',
                    help="Get slice of table with utts with labels")

# training args
parser.add_argument("--train_search_summary", action='store_true',
                    help="Get summary of the hyperparameter tuning search space")
parser.add_argument("--train_results_summary", action='store_true',
                    help="Get results of hyperparameter tuning")
parser.add_argument("--train_model_summary", action='store_true',
                    help="Get summary of the model architecture")
parser.add_argument("--train_model_diagram", action='store_true',
                    help="Get diagram of model architecture")

# evaluation args
parser.add_argument("--eval", action='store_true',
                    help="Enable to perform model evaluation")
parser.add_argument("--eval_test_loss_acc", action='store_true',
                    help="Get loss and accuracy of test data")
parser.add_argument("--eval_acc_plot", action='store_true',
                    help="Plot training and test data accuracy over epochs")
parser.add_argument("--eval_loss_plot", action='store_true',
                    help="Plot training and test data loss over epochs")
parser.add_argument("--eval_comp_preds", action='store_true',
                    help="Compare predicted vs actual intents")
parser.add_argument("--eval_fscore", action='store_true',
                    help="Get f score of test data")
parser.add_argument("--eval_conf_matrix", action='store_true',
                    help="Get confusion matrix of predicted vs actual intents")
args = parser.parse_args()

if args.cluster:
    if os.path.isdir(DATA_DIR):
        # Preprocess chat logs
        preprocessed_data = parse_files(DATA_DIR)

        # do exploratory data analysis
        if not preprocessed_data.empty:
            print('Data is preprocessed into dataframe')

            if args.eda:
                # instantiate eda class object
                eda = ExploreData(preprocessed_data)

                if args.eda_token_dist:
                    # get token frequency distribution
                    eda.get_token_frequency_dist()

                if args.eda_plot_token_dist:
                    # get distribution plot
                    eda.plot_dist_curve()

                if args.eda_top_n_tokens:
                    # get top N tokens
                    eda.get_top_n_tokens(args.top_n_tokens)

                if args.eda_tokens_hist:
                    # get token length histogram
                    eda.get_token_length_visualisations()

                if args.eda_sent_hist:
                    # get sentence length histogram
                    eda.get_sent_length_visuals()

            # cluster intents
            clustered_intents = ClusterIntents(preprocessed_data, LABEL_FILE, MODULE_URL, sorted_labels=False)  # if sorted_labels is True, labelled data sorted by intent

            if os.path.exists(LABEL_FILE):

                if args.cluster_best_params:
                    # get best parameters
                    clustered_intents.get_model_best_params()

                if args.cluster_plot:
                    # get cluster plot
                    clustered_intents.get_cluster_plot()

                if args.cluster_labels_summary:
                    # get summary of label stats
                    clustered_intents.get_labels_summary(20)

                if args.cluster_labeled_utts:
                    # check labeled utts
                    clustered_intents.get_labeled_utts(20)
            else:
                print('Label file not created')
        else:
            print('Preprocessed data empty')
    else:
        print('Data directory not found')


if os.path.exists(INTENT_FILE):
    RUN_TRAIN_PREDICT = True
elif os.path.exists(LABEL_FILE_POST):
    INTENT_FILE = parse_data_csv(LABEL_FILE_POST)
    RUN_TRAIN_PREDICT = True

if RUN_TRAIN_PREDICT:
    # get intents
    with open(INTENT_FILE, 'r') as f:
        intent_data = json.load(f)

    intents = sorted(list(intent_data.keys()))

    if args.train:

        # prepare training data
        prep = PrepareData(intent_data, intents)

        # gets train, validation, and test inputs and outputs
        train_x, val_x, test_x, train_y, val_y, test_y = prep.train_x, prep.val_x, prep.test_x, prep.train_y, prep.val_y, prep.test_y

        # train intent recognition model
        model_train = ModelTrain(train_x, val_x, train_y, val_y, MODULE_URL)

        if args.train_search_summary:
            # get hyperparameter search space
            model_train.get_search_summary()

        if args.train_results_summary:
            # get hyperparameter tuning resuts
            model_train.get_results_summary()

        if args.train_model_summary:
            # get descrioption of model architecture
            model_train.get_model_summary()

        if args.train_model_diagram:
            # get diagram of model
            model_train.get_model_diagram()

    if os.path.exists(MODEL_PATH):

        # load model
        model = load_model(MODEL_PATH)

        if args.eval:
            # evaluate model
            evaluate = Eval(model, prep, intents, test_x, test_y)

            if args.eval_test_loss_acc:
                # test set loss and accuracy
                evaluate.get_test_loss_acc()

            if args.eval_acc_plot or args.eval_loss_plot:
                # load history
                history = pickle.load(open(MODEL_HISTORY, "rb"))

                if args.eval_acc_plot:
                    # get accouracy plot
                    evaluate.get_accuracy_plot(history)

                if args.eval_loss_plot:
                    # get loss plot
                    evaluate.get_loss_plot(history)

            if args.eval_comp_preds:
                # compare predicted vs actual intents
                evaluate.compare_predicted_intents()

            if args.eval_fscore:
                # get f score of predicted vs actual intents
                evaluate.get_fscore()

            if args.eval_conf_matrix:
                # get confusion matrix of predicted vs actual intents
                evaluate.get_confusion_matrix()

        # run
        if args.chat:
            bot_name = "Anjali"
            print("How may we help you? (type 'quit' to exit)")
            while True:
                sentence = input("You: ")
                if sentence == "quit":
                    break

                predict = Predict(model, intents, intent_data, sentence)
                print(f"{bot_name}: {predict.response}")

    else:
        print('Model not found, please verify location or train new model.')
else:
    print('Intents json file not found.')
