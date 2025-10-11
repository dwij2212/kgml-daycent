The setting is as follows:

10 scenarios were randomly chosen from 10000. For this experiment, these were - `['2034', '2600', '1616', '2729', '1000', '3932', '3503', '1390', '4880', '1099']`

For training following scenarios were used:

scenario_1000
scenario_1099
scenario_1390
scenario_1616
scenario_2034
scenario_2600
scenario_2729

For validation:

scenario_3503
scenario_3932

For testing:
scenario_4880

There was also holdout for the points. This was done quadrant wise. We chose Q1 (SW) for training.

The rest all points were not used in train time at all. they are used only to evaluate the results.

The model was standard daycent model. 