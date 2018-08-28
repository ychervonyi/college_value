"""

Student model:

============================================
Coefficients:
 [[ 22900.75770752 -39526.81071915  52229.04112271]]
============================================
R squared: 0.4323
Features: ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID']
Number of examples: 8015

College model:

============================================
Coefficients:
 [[ 0.01029673  0.62380574  0.0756638  -0.30289539 -0.07458631  0.18591655
  -0.08744855  0.03828624  0.04916231]]
============================================
R squared: 0.1447
Features: ['ADM_RATE_ALL', 'AVGFACSAL', 'TUITIONFEE_IN', 'TUITIONFEE_OUT', 'PFTFAC', 'GRADS', 'CONTROL_0', 'CONTROL_1', 'CONTROL_2']
Number of examples: 6536


"""


from model import Model
from data_tools import Feature, process_data, get_data
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import argparse


def compute_college_scores(model_type, x, y):
    # Load student model
    print("Loading model...")
    student_model = Model(model_type=model_type, model_name='student')
    student_model.load()

    assert student_model is not None, "Can't load a model"

    # Actual earnings
    y_raw = y

    # Predicted earnings
    y_predicted = student_model.predict(x)

    # College score
    college_score = np.divide(y_raw, y_predicted)
    return college_score


def train_model(x, y, feature_names, model_name, batch=16, n_epochs=300,
                learning_rate=0.1, model_type='sklearn', save=False):
    # x, y, feature_names = process_data(features_all=features_all,
    #                                    features_model=features_model,
    #                                    path=path)
    # Make test and train set
    train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=0)

    params = {'batch': batch, 'epochs': n_epochs, 'learning_rate': learning_rate}
    print("Training %s model..." % model_name)
    model = Model(model_type=model_type, model_name=model_name, input_shape=train_X.shape[1], params=params)
    model.create()
    model.train(train_X, train_y, test_X, test_y)
    if save:
        model.save()

    predictions = model.predict(test_X)
    r_squared = r2_score(test_y, predictions)

    model.print()

    print("R squared: %.4f" % r_squared)
    print("Features: %s" % feature_names)
    print("Number of examples: %d" % len(y))


def train_student_model(features_all, features_model, model_name, path, batch=16, n_epochs=300,
                learning_rate=0.1, model_type='sklearn', save=False):
    # Merge data over multiple years
    print("Reading data...")
    dataset = get_data(features_all, path=path)

    df, x, y, feature_names = process_data(dataset=dataset,
                                       features_model=features_model)

    train_model(x=x, y=y, model_name=model_name, feature_names=feature_names, batch=batch,
                n_epochs=n_epochs, learning_rate=learning_rate, model_type=model_type, save=save)


def generate_ranking(features_all, features_model, path, model_type='sklearn'):
    # Merge data over multiple years
    print("Reading data...")
    dataset = get_data(features_all, path=path)

    df, x, y, feature_names = process_data(dataset=dataset,
                                           features_model=features_model)

    # College score
    college_score = compute_college_scores(model_type, x, y)

    college_name = df['INSTNM'].values
    college_name = college_name.reshape((-1, 1))

    # Concatenate college scores and names
    scores = np.concatenate((college_name, college_score), axis=1)

    # We might have data over several years, so let's create a hash table and compute average over years
    scores_dict = {}
    for i in range(scores.shape[0]):
        name = scores[i, 0]
        if scores[i, 1] not in scores_dict:
            scores_dict[name] = [float(scores[i, 1])]
        else:
            scores_dict[name].append(float(scores[i, 1]))
    for key, value in scores_dict.items():
        scores_dict[key] = np.mean(value)

    # Write scores into a file
    scores_sorted = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    scores_file = 'scores.csv'
    with open(scores_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in scores_sorted:
            writer.writerow(row)
    print("Scores are saved in %s" % scores_file)


def train_college_model(features_all, features_student, features_model, model_name, path, batch=16, n_epochs=300,
                learning_rate=0.1, model_type='sklearn', save=False):
    # Merge data over multiple years
    print("Reading data...")
    dataset = get_data(features_all, path=path)

    df, x, y, feature_names = process_data(dataset=dataset,
                                           features_model=features_student)
    y = compute_college_scores(model_type, x, y)

    df, x, _, feature_names = process_data(dataset=dataset,
                                           features_model=features_model)

    train_model(x=x, y=y, model_name=model_name, feature_names=feature_names, batch=batch,
                n_epochs=n_epochs, learning_rate=learning_rate, model_type=model_type, save=save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    args = parser.parse_args()

    features_earnings = [Feature(name='MN_EARN_WNE_MALE1_P6'),
                         Feature(name='COUNT_WNE_MALE1_P6'),
                         Feature(name='MN_EARN_WNE_MALE0_P6'),
                         Feature(name='COUNT_WNE_MALE0_P6')]

    features_student = [Feature(name='SAT_AVG_ALL'),
                        Feature(name='SATVRMID'),
                        Feature(name='SATMTMID')]

    features_college = [Feature(name='INSTNM', datatype='str', replace_with='')]

    features_all = features_earnings + features_student + features_earnings

    if args.task == 'student':
        train_student_model(features_all, features_student,
                            model_name='student',
                            model_type='sklearn',
                            path='/CollegeScorecard_Raw_Data/*.csv',
                            save=True)
    elif args.task == 'scores':
        generate_ranking(features_all, features_student, model_type='sklearn',
                         path='/CollegeScorecard_Raw_Data/*.csv')
    elif args.task == 'college':
        features_college = [
            Feature(name='ADM_RATE_ALL'),
            Feature(name='AVGFACSAL',),
            Feature(name='TUITIONFEE_IN'),
            Feature(name='TUITIONFEE_OUT'),
            Feature(name='PFTFAC'),
            Feature(name='GRADS'),
            Feature(name='CONTROL', onehot=True)
        ]
        features_all = features_earnings + features_student + features_college
        train_college_model(features_all, features_student, features_college,
                            model_name='college',
                            model_type='sklearn',
                            path='/CollegeScorecard_Raw_Data/*.csv',
                            save=False)
    else:
        print("No task is specified...")