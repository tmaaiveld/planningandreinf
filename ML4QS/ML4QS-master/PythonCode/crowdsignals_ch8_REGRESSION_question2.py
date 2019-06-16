##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from statsmodels.tsa.stattools import adfuller
from pandas.tools.plotting import autocorrelation_plot
from Chapter7.FeatureSelection import FeatureSelectionRegression
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split


# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles/'

try:
    dataset = pd.read_csv(dataset_path + 'chapter5_result_tommy_500_edited_timestamps_final.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = dataset.index.to_datetime()

# Let us consider our second task, namely the prediction of the heart rate. We consider this as a temporal task.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'gyr_phone_x', '01/01/1970  00:00:00',
                                                                                   '01/01/1970  00:07:00', '01/01/1970  00:10:49')   # change split for 2nd time, now it should give proper pliot

# print 'Training set length is: ', len(train_X.index)
# print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

print 'Training set length is: ', len(train_X.index)
# print train_X.iloc[:, :4].tail(5)
print 'Test set length is: ', len(test_X.index)
# print train_X.iloc[:, :4].head(5)


# Select subsets of the features that we will consider:

basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','gyr_phone_y','gyr_phone_z',
                 'mag_phone_x','mag_phone_y','mag_phone_z']
pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7']
time_features = [name for name in dataset.columns if ('temp_' in name and not 'gyr_phone_x' in name)]
freq_features = [name for name in dataset.columns if ((('_freq' in name) or ('_pse' in name)) and not 'gyr_phone_x' in name)]
print '#basic features: ', len(basic_features)
print '#PCA features: ', len(pca_features)
print '#time features: ', len(time_features)
print '#frequency features: ', len(freq_features)
cluster_features = ['cluster']
print '#cluster features: ', len(cluster_features)
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

fs = FeatureSelectionRegression()
# First, let us consider the Pearson correlations and see whether we can select based on them.
# features, correlations = fs.pearson_selection(10, train_X[features_after_chapter_5], train_y)
# util.print_pearson_correlations(correlations)
best_features, ordered_features, ordered_scores = fs.forward_selection(10, train_X[features_after_chapter_5].fillna(method='ffill').fillna(method='bfill'), train_y)

print 'These were the selected_feautures returned by  Forward Selection: '
print best_features
print 'These were the ordered_feautures returned by  Forward Selection: '
print ordered_features
print 'These were the ordered_scores returned by  Forward Selection: '
print ordered_scores

'''
selected_features = ['pca_2', 'pca_7', 'pca_1', 'mag_phone_z_freq_0.3_Hz_ws_10', 'acc_phone_y_temp_mean_ws_30_freq_0.5_Hz_ws_10', 'pca_5_temp_std_ws_30_freq_0.2_Hz_ws_10', 'pca_2_freq_0.0_Hz_ws_10', 'pca_3_temp_std_ws_30_freq_weighted', 'mag_phone_z_freq_0.2_Hz_ws_10', 'acc_phone_z_temp_mean_ws_30_freq_0.0_Hz_ws_10']
'''
selected_features = ordered_features[0:10]
print '________'
print 'We will go ahead with this top 10 of features, which we define as selected_features in the remainder of this script'
print selected_features


possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

# Let us first study whether the time series is stationary and what the autocorrelations are.

dftest = adfuller(dataset['gyr_phone_x'], autolag='AIC')
print dftest

autocorrelation_plot(dataset['gyr_phone_x'])
plot.show()

# Now let us focus on the learning part.

learner = TemporalRegressionAlgorithms()
eval = RegressionEvaluation()

# We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.

repeats = 5

# we set a washout time to give the NN's the time to stabilize. We do not compute the error during the washout time.

washout_time = 10

scores_over_all_algs = []

for i in range(0, len(possible_feature_sets)):

    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_res = 0
    performance_tr_res_std = 0
    performance_te_res = 0
    performance_te_res_std = 0
    performance_tr_rnn = 0
    performance_tr_rnn_std = 0
    performance_te_rnn = 0
    performance_te_rnn_std = 0

    for repeat in range(0, repeats):
        print '----', repeat
        regr_train_y, regr_test_y = learner.reservoir_computing(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True, per_time_step=False)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        performance_tr_res += mean_tr
        performance_tr_res_std += std_tr
        performance_te_res += mean_te
        performance_te_res_std += std_te

        regr_train_y, regr_test_y = learner.recurrent_neural_network(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        performance_tr_rnn += mean_tr
        performance_tr_rnn_std += std_tr
        performance_te_rnn += mean_te
        performance_te_rnn_std += std_te


    # We only apply the time series in case of the basis features.

    if (feature_names[i] == 'initial set'):
        regr_train_y, regr_test_y = learner.time_series(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        overall_performance_tr_ts = mean_tr
        overall_performance_tr_ts_std = std_tr
        overall_performance_te_ts = mean_te
        overall_performance_te_ts_std = std_te
    else:
        overall_performance_tr_ts = 0
        overall_performance_tr_ts_std = 0
        overall_performance_te_ts = 0
        overall_performance_te_ts_std = 0


    overall_performance_tr_res = performance_tr_res/repeats
    overall_performance_tr_res_std = performance_tr_res_std/repeats
    overall_performance_te_res = performance_te_res/repeats
    overall_performance_te_res_std = performance_te_res_std/repeats
    overall_performance_tr_rnn = performance_tr_rnn/repeats
    overall_performance_tr_rnn_std = performance_tr_rnn_std/repeats
    overall_performance_te_rnn = performance_te_rnn/repeats
    overall_performance_te_rnn_std = performance_te_rnn_std/repeats

    scores_with_sd = [(overall_performance_tr_res, overall_performance_tr_res_std, overall_performance_te_res,
                       overall_performance_te_res_std),
                      (overall_performance_tr_rnn, overall_performance_tr_rnn_std, overall_performance_te_rnn,
                       overall_performance_te_rnn_std),
                      (overall_performance_tr_ts, overall_performance_tr_ts_std, overall_performance_te_ts,
                       overall_performance_te_ts_std)]
    print scores_with_sd
    util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index),
                                                 len(selected_test_X.index), scores_with_sd)
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_regression(['Reservoir', 'RNN', 'Time series'], feature_names, scores_over_all_algs)

regr_train_y, regr_test_y = learner.reservoir_computing(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['gyr_phone_x'], test_X.index, test_y, regr_test_y['gyr_phone_x'], 'gyr_phone_x')
regr_train_y, regr_test_y = learner.recurrent_neural_network(train_X[basic_features], train_y, test_X[basic_features], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['gyr_phone_x'], test_X.index, test_y, regr_test_y['gyr_phone_x'], 'gyr_phone_x')
regr_train_y, regr_test_y = learner.time_series(train_X[basic_features], train_y, test_X[features_after_chapter_5], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['gyr_phone_x'], test_X.index, test_y, regr_test_y['gyr_phone_x'], 'gyr_phone_x')
