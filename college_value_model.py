from data_tools import get_data
import numpy as np
import pandas
from model import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_tools import Feature


def train_model(features_all, features_model, model_name, path, batch=16, n_epochs=300,
                learning_rate=0.1, model_type='keras'):
    assert model_type in ('keras', 'sklearn'), "Model type should be keras or sklearn"
    params = {'batch': batch, 'epochs': n_epochs, 'learning_rate': learning_rate}

    # Merge data over multiple years
    print("Reading data...")
    dataset = get_data(features_all, path=path)

    dfs = []
    for key, value in dataset.items():
        dfs.append(value)
    df = pandas.concat(dfs)

    Y = df['EARNINGS'].values.reshape((-1, 1))

    X = []
    feature_names = []
    for feature in features_model:
        # Normalize only floats
        if feature.datatype != 'float64':
            continue
        # Normalize - min max normalization
        print("Normalizing...")
        feature_name = feature.name
        feature_names.append(feature_name)
        col = df[feature_name].values
        col_max, col_min = np.amax(col), np.amin(col)
        print("Feature: %s, max: %.4f, min: %.4f" % (feature_name, col_max, col_min))
        X.append((col - col_min)/(col_max - col_min))

        # col = data[:, c]
        # col_max, col_min = np.amax(col), np.amin(col)
        # print("After. Feature: %s, max: %.4f, min: %.4f" % (feature_name, col_max, col_min))

    X = np.asarray(X).transpose()
    # Make test and train set
    train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)

    print("Training %s model..." % model_name)
    model = Model(model_type=model_type, model_name=model_name, input_shape=X.shape[1], params=params)
    model.create()
    model.train(train_X, train_y, test_X, test_y)
    # model.save()

    predictions = model.predict(test_X)
    r_squared = r2_score(test_y, predictions)

    model.print()

    print("R squared: %.4f" % r_squared)
    print("Features: %s" % feature_names)
    print("Number of examples: %d" % len(Y))


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
    train_model(features_all, features_student, model_name='student',
                model_type='sklearn', path='/CollegeScorecard_Raw_Data/*.csv')