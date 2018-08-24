from model import Model
from data_tools import Feature, process_data
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import csv


def train_model(features_all, features_model, model_name, path, batch=16, n_epochs=300,
                learning_rate=0.1, model_type='keras', save=False):
    X, Y, feature_names = process_data(features_all=features_all,
                                                    features_model=features_model,
                                                    path=path)
    # Make test and train set
    train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)

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
    print("Number of examples: %d" % len(Y))


def generate_ranking(features_all, features_model, model_name, path, batch=16, n_epochs=300,
                learning_rate=0.1, model_type='keras'):
    df, X, Y, feature_names = process_data(features_all=features_all,
                                                    features_model=features_model,
                                                    path=path)
    # Load student model
    print("Loading student model...")
    student_model = Model(model_type=model_type, model_name='student')
    student_model.load()

    assert student_model is not None, "Can't load a model"

    Y_raw = Y

    Y_student_predicted = student_model.predict(X)
    Y = np.divide(Y_raw, Y_student_predicted)

    school_name = df['INSTNM']

    con = np.concatenate((school_name.reshape((-1, 1)), Y), axis=1)

    scores_dict = {}
    for i in range(con.shape[0]):
        name = con[i, 0]
        if con[i, 1] not in scores_dict:
            scores_dict[name] = [float(con[i, 1])]
        else:
            scores_dict[name].append(float(con[i, 1]))
    for key, value in scores_dict.items():
        scores_dict[key] = np.mean(value)

    scores_sorted = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    with open('scores.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in scores_sorted:
            writer.writerow(row)


if __name__ == '__main__':
    earnings_features = [Feature(name='MN_EARN_WNE_MALE1_P6'),
                         Feature(name='COUNT_WNE_MALE1_P6'),
                         Feature(name='MN_EARN_WNE_MALE0_P6'),
                         Feature(name='COUNT_WNE_MALE0_P6')]
    features_student = [Feature(name='SAT_AVG_ALL'),
                        Feature(name='SATVRMID'),
                        Feature(name='SATMTMID')]
    college_features = [Feature(name='INSTNM', datatype='str', replace_with='')]
    features_all = earnings_features + features_student + college_features
    # train_model(features_all, features_student, model_name='student',
    #             model_type='sklearn',
    #             path='/CollegeScorecard_Raw_Data/*.csv',
    #             save=True)
    generate_ranking(features_all, features_student, model_name='student',
                model_type='sklearn',
                path='/CollegeScorecard_Raw_Data/*.csv')