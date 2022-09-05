from configuration import *
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from qboost import QBoostClassifier


class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Init variables

        params: 
            X_train: the training dataset
            X_test: the test dataset
            y_train: the label of the training dataset
            y_test: the label of the test dataset
        """
        self.classifier_list = classificators_list
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset = '' # for CVE dataset [bow, tf, tfidf]
        self.instance_classifier = None # the classificator instance

    
    def set_dataset(self, dataset):
        """
        Set the dataset already in use. This is for CVE classification
        param:
            dataset: the dataset currently in use [bow, tf, tfidf]
        """
        self.dataset = dataset
    

    def randomforest_classifier(self):
        """
        Create an instance of the random forest

        return: instance of RandomForestClassifier
        """
        from sklearn.ensemble import RandomForestClassifier

        rfc = RandomForestClassifier(max_depth=randomforest_max_depth)
        return rfc.fit(self.X_train, self.y_train)


    def adaboost_classifier(self):
        """
        Create an instance of the adaboost

        return: instance of AdaBoostClassifier
        """
        from sklearn.ensemble import AdaBoostClassifier

        adabclassifier = AdaBoostClassifier()
        return adabclassifier.fit(self.X_train, self.y_train)

    

    def svc_classifier(self):
        """
        Create an instance of the SVC

        return: instance of SVC
        """
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
        clf.fit(self.X_train, self.y_train)

        return clf
    

    def qboost_classifier(self):
        """
        Create a classifier for QBoost
        """
        dwave_sampler = DWaveSampler()
        emb_sampler = EmbeddingComposite(dwave_sampler)
        lmd = 0.04
        dwave_sampler = DWaveSampler(token=DEVtoken)
        emb_sampler = EmbeddingComposite(dwave_sampler)
        lmd = 0.5
        qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
        qboost.fit(self.X_train, self.y_train, emb_sampler, lmd=lmd, **DW_PARAMS)

        return qboost
        
    def print_graphs(self, classifier, calculate_training=False, threshold=False):

        import matplotlib.pyplot as plt
        from scikitplot.metrics import plot_roc
        from scikitplot.metrics import plot_precision_recall
        from scikitplot.metrics import plot_cumulative_gain
        from scikitplot.metrics import plot_lift_curve
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        from numpy import argmax
        import numpy as np
        from os import getcwd

        y_label = None

        if calculate_training:
            y_score = self.instance_classifier.predict_proba(self.X_train)
            fpr0, tpr0, thresholds = roc_curve(self.y_train, y_score[:, 1])
            roc_auc0 = auc(fpr0, tpr0)
            y_label = self.y_train 
        else:
            y_score = self.instance_classifier.predict_proba(self.X_test)
            fpr0, tpr0, thresholds = roc_curve(self.y_test, y_score[:, 1])
            roc_auc0 = auc(fpr0, tpr0)
            y_label = self.y_test
        
        # Calculate the best threshold
        best_threshold = None
        if threshold:
            J = tpr0 - fpr0
            ix = argmax(J) # take the value which maximizes the J variable
            best_threshold = thresholds[ix]
            # adjust score according to threshold.
            y_score = np.array([[1, y[1]] if y[0] >= best_threshold else [0, y[1]] for y in y_score])

        # Plot metrics
        plot_roc(y_label, y_score)
        # plt.show()
        path = f'{getcwd()}/plot/plot_roc_{classifier}_{self.dataset}.png'
        plt.savefig(path)
        
        plot_precision_recall(y_label, y_score)
        # plt.show()
        path = f'{getcwd()}/plot/plot_precision_recall_{classifier}_{self.dataset}.png'
        plt.savefig(path)
        
        plot_cumulative_gain(y_label, y_score)
        # plt.show()
        path = f'{getcwd()}/plot/plot_cumulative_gain_{classifier}_{self.dataset}.png'
        plt.savefig(path)
        
        plot_lift_curve(y_label, y_score)
        # plt.show()
        path = f'{getcwd()}/plot/plot_lift_curve_{classifier}_{self.dataset}.png'
        plt.savefig(path)
        
        # return roc_auc0, fpr0, tpr0, best_threshold

    def metrics(self, y, prediction):
        """
        Calculate the metrics with specific y_dataset and prediction of the classifier
        that are in used.
        """
        from sklearn.metrics import classification_report
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score

        print('Precision score %s' % precision_score(y, prediction))
        print('Recall score %s' % recall_score(y, prediction))
        print('F1-score score %s' % f1_score(y, prediction))
        print('Accuracy score %s' % accuracy_score(y, prediction))

        return classification_report(y, prediction)


    def create_file_report(self, end_time, name_cls, report, time_performance):
        """
        Create report file
        """
        import os
        path = f'{os.getcwd()}/report_undersample/{name_cls}_{self.dataset}.txt'
        with open(path, 'a') as f:
            f.write(report)
            f.write(f"Time performance for {time_performance}: {end_time}")
            f.write("\n")
        f.close()


    def learner_classifier(self, classifier: str):
        """
        Call dinamically the function randomforest, adaboost, svc, and qboost.
        All the methods are in the form nameclassifier_classifier i.e randomforest_classifier.
        Calculate the time for the prediction and create a file text with the results for each classifier.
        """
        from sys import modules
        from timeit import default_timer as timer

        if classifier.lower() not in self.classifier_list:
            raise Exception('Please, define a correct classificator. randomforest, adaboost, svc, qboost')
        
        start = timer()
        self.instance_classifier = getattr(self, '%s_classifier' % classifier.lower())()
        end = timer()
        end_time = end - start

        return end_time
    

    def run_classification(self, not_training_dataset: bool, list_classifier=[]):
        """
        Run a classification and prediction for specific classifier.
        The classifier must be specified into configuration files.
        params:
            not_training_dataset: if False runs the prediction on training and testing dataset
                                  if True runs the prediction only on testing dataset
        """
        from timeit import default_timer as timer
        from sys import exit

        if len(list_classifier) > 0:
            self.classifier_list = list_classifier

        for clf in self.classifier_list:
            print(f'{clf} classification...')
            # create model for each classifier
            end_training_time = self.learner_classifier(clf)
            print(f'End {clf} training in: {end_training_time}')

            # prediction only on test set
            if not_training_dataset:
                print(f'Start prediction of {clf} for the test dataset...')
                # start prediction
                prediction_test = self.instance_classifier.predict(self.X_test)
                # plot qboost does not have the method prodict_proba
                # TODO: try to create a custom predict_prova also for QBoost
                if clf != 'qboost':
                    self.print_graphs(clf)
                # calculate matrics
                report = self.metrics(self.y_test, prediction_test)
                # create file report
                self.create_file_report(end_training_time, clf, report, 'test dataset')
                print(f'End prediction test dataset of {clf} and report created.')

            else:
                print(f'Start prediction of {clf} for the training dataset...')
                # prediction with train set
                prediction_train = self.instance_classifier.predict(self.X_train)

                # plot qboost does not have the method prodict_proba
                if clf != 'qboost':
                    self.print_graphs(clf, calculate_training=True)
                # calculate matrics
                report = self.metrics(self.y_train, prediction_train)
                # create file report
                self.create_file_report(end_training_time, clf, report, 'train dataset')
                print(f'End prediction of train dataset of {clf} and report created.')

                print(f'Start prediction of {clf} for the test dataset...')
                # start timer
                start = timer()
                # prediction with test set
                prediction_test = self.instance_classifier.predict(self.X_test)
                end = timer()
                # end timer
                end_testing_time = end - start
                
                # plot qboost does not have the method prodict_proba
                if clf != 'qboost':
                    self.print_graphs(clf)
                # calculate matrics
                report = self.metrics(self.y_test, prediction_test)
                # create file report
                self.create_file_report(end_testing_time, clf, report, 'test dataset')
                print(f'End prediction of train dataset of {clf} and report created.')

