# This script contains functions that assess the performance of models eg ROC curves

def roc_compare(trainX, testX, trainY, testY, model):

    # Random Guessing
    ns_probs = [0 for _ in range(len(testY))]

    # predict probabilities
    lr_probs = model.predict_proba(testX)

    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testY, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testY, lr_probs)

    return ns_fpr, ns_tpr, lr_fpr, lr_tpr


def accuracy_test(testX, testY, model):

    resultY = model.predict(testX)

    accuracy = sum(resultY == np.ravel(testY)) / len(resultY)

    round(accuracy * 100, 1)

    return(accuracy)

def roc_auc(testX, testY, model):

    # predict probabilities
    test_probs = model.predict_proba(testX)

    test_probs = test_probs[:,1]

    auc = roc_auc_score(np.ravel(testY), test_probs)

    return(auc)
