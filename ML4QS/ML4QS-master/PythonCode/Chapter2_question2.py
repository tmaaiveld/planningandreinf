from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
import copy
import pandas as pd

#
'''        Erik data            '''
erik_path = '/Users/whitn/OneDrive/Documenten/Groupwork_TommyErik/ML4QS/ML4QS-master/datasets/phyphox_erik_thurs/'
# result_dataset_path = './intermediate_datafiles/'

# Chapter 2: Initial exploration of the dataset.

# Set a granularity (i.e. how big are our discrete time steps). We start very
# coarse grained, namely one measurement per minute, and secondly use four measurements
# per second

granularities = [250]
datasets = []

for milliseconds_per_instance in granularities:

    # Create an initial dataset object with the base directory for our data and a granularity
    DataSet = CreateDataset(erik_path, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values/
    DataSet.add_numerical_dataset('Accelerometer.csv', 'timestamps', ['x',
                                                                      'y',
                                                                      'z'], 'avg', 'acc_phone_')
    # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values/
    DataSet.add_numerical_dataset('Gyroscope.csv', 'timestamps', ['x',
                                                                  'y',
                                                                  'z'], 'avg', 'gyr_phone_')
    # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    DataSet.add_numerical_dataset('Magnetometer.csv', 'timestamps',['x',
                                                                    'y',
                                                                    'z'], 'avg', 'mag_phone_')
    # Get the resulting pandas data table

    erik = DataSet.data_table


'''              Crowdsignals data                     '''


dataset_path = '/Users/whitn/OneDrive/Documenten/Groupwork_TommyErik/ML4QS/ML4QS-master/PythonCode/intermediate_datafiles/chapter2_result.csv'

data = pd.read_csv(dataset_path)
df1= data.iloc[7494:8760, ] ### walking + nothing
df2 = data.iloc[8760:9910, ]  # df2 = data.iloc[8930:9910, ]

columns_to_drop  =  ['labelOnTable', 'labelSitting', 'labelWashingHands', 'labelDriving', 'labelEating', 'labelRunning','light_phone_lux',
                     'press_phone_pressure',
                     'acc_watch_x',
                     'acc_watch_y','acc_watch_z',   'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z' , 'mag_watch_x',
                     'mag_watch_y', 'mag_watch_z'

                     , 'hr_watch_rate',
                     
                     'labelWalking', 'labelStanding'
                     ]
dataset =pd.concat([df1, df2]).drop(columns_to_drop, axis=1)

 ################################################3

# print dataset.head()
# print  erik.head()

# Plot the data
DataViz = VisualizeDataset()

# Boxplot
# DataViz.plot_dataset_boxplot(dataset, ['acc_phone_x',
#                                        'acc_phone_y',
#                                        'acc_phone_z'])

# Plot all data
DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'mag_'], ['like', 'like', 'like','like'], ['line', 'line', 'line','points'])

# print dataset.head()
# print ' _____________ '
# print erik.head()

# And print a summary of the dataset
util.print_statistics(dataset.iloc[:, 1:])
print ' _____________ '
util.print_statistics(erik)
print ' _____________ '

# datasets.append(copy.deepcopy(dataset))

# And print the table that has been included in the book
util.print_latex_table_statistics_two_datasets(dataset.iloc[:, 1:], erik)
# util.print_latex_table_statistics_two_datasets(datasets[2], datasets[3])

# Finally, store the last dataset we have generated (250 ms).
#dataset.to_csv(result_dataset_path + 'chapter2_result_notebook.csv')
