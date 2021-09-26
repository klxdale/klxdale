# this code loops through several values of nodes/layers and looks at the accuracy of the model with these attributes
# the output is saved as a csv
# Validate Models Using 2019 Season, and train models using 2018 season
X_Valid = normalise_0_1(
    merge_data_all_2019[['KI', 'MK', 'HB', 'TK', 'CL', 'IF',
                         'CP', 'UP', 'CM', 'MI',
                         'Ladder Diff', 'Rolling Point Diff', 'Form Diff']].dropna())
# Output matrix contents
Y_Valid = categorise_margin(pd.DataFrame(merge_data_all_2019[['Margin']][9:198].dropna()))

# Let's test the amount of hidden layers and nodes to use
# Start with 3 layers
# Appropriate amount of hidden neurons is sqrt(27*1) ~ 5 OR sqrt(13,1) ~ 3.6
Layer_1 = list()
Layer_2 = list()
Layer_3 = list()
Layer_4 = list()
Layer_5 = list()
Accuracy_Train_List = list()
Accuracy_Valid_List = list()
Accuracy_WinLoss_Train_List = list()
Accuracy_WinLoss_Valid_List = list()
Accuracy_WinLossBlowout_Train_List = list()
Accuracy_WinLossBlowout_Valid_List = list()

for i in [3, 6, 9, 12, 15]:
    for j in [3, 6, 9, 12, 15]:
        for k in [3, 6, 9, 12, 15]:
            clf_Neural = MLPClassifier(hidden_layer_sizes=(i, j, k), learning_rate='adaptive',
                                       max_iter=10000)
            clf_Neural.fit(X_Train, np.ravel(Y_Train))
            print(i, j, k)

            ### Win/Loss categories ###
            # Training data
            Accuracy_Train = clf_Neural.score(X_Train, np.ravel(Y_Train), sample_weight=None)
            print('Neural Network Accuracy (Train):', round(Accuracy_Train * 100, 1), '%')
            # Validation data
            Home_Win = Y_Valid.reset_index()
            Predict_Win = pd.DataFrame({'Margin': clf_Neural.predict(X_Valid)})
            Accuracy_Valid = sum(Home_Win['Margin'] == Predict_Win['Margin']) / len(Home_Win)
            print('Neural Network Accuracy (Valid):', round(Accuracy_Valid * 100, 1), '%')

            ### Win/Loss ###
            # Training data
            Home_Win = category_to_win_loss(Y_Train.reset_index())
            Predict_Win = category_to_win_loss(pd.DataFrame({'Margin': clf_Neural.predict(X_Train)}))
            Accuracy_Train_WL = sum(Home_Win['Margin'] == Predict_Win['Margin']) / len(Home_Win)
            print('Neural Network Accuracy (Train):', round(Accuracy_Train_WL * 100, 1), '%')
            # Validation data
            Home_Win = category_to_win_loss(Y_Valid.reset_index())
            Predict_Win = category_to_win_loss(pd.DataFrame({'Margin': clf_Neural.predict(X_Valid)}))
            Accuracy_Valid_WL = sum(Home_Win['Margin'] == Predict_Win['Margin']) / len(Home_Win)
            print('Neural Network Accuracy (Valid):', round(Accuracy_Valid_WL * 100, 1), '%')

            ### Win/Loss blowout ###
            # Training data
            Home_Win = category_to_blowout_winloss(Y_Train.reset_index())
            Predict_Win = category_to_blowout_winloss(pd.DataFrame({'Margin': clf_Neural.predict(X_Train)}))
            Accuracy_Train_BO = sum(Home_Win['Margin'] == Predict_Win['Margin']) / len(Home_Win)
            print('Neural Network Accuracy (Train):', round(Accuracy_Train_BO * 100, 1), '%')
            # Validation data
            Home_Win = category_to_blowout_winloss(Y_Valid.reset_index())
            Predict_Win = category_to_blowout_winloss(pd.DataFrame({'Margin': clf_Neural.predict(X_Valid)}))
            Accuracy_Valid_BO = sum(Home_Win['Margin'] == Predict_Win['Margin']) / len(Home_Win)
            print('Neural Network Accuracy (Valid):', round(Accuracy_Valid_BO * 100, 1), '%')

            Layer_1.append(i)
            Layer_2.append(j)
            Layer_3.append(k)
            Accuracy_Train_List.append(Accuracy_Train)
            Accuracy_Valid_List.append(Accuracy_Valid)
            Accuracy_WinLoss_Train_List.append(Accuracy_Train_WL)
            Accuracy_WinLoss_Valid_List.append(Accuracy_Valid_WL)
            Accuracy_WinLossBlowout_Train_List.append(Accuracy_Train_BO)
            Accuracy_WinLossBlowout_Valid_List.append(Accuracy_Valid_BO)

Output = pd.DataFrame({'Layer_1': Layer_1,
                       'Layer_2': Layer_2,
                       'Layer_3': Layer_3,
                       'Accuracy_Train': Accuracy_Train_List,
                       'Accuracy_Validation': Accuracy_Valid_List,
                       'Accuracy_Train_WL': Accuracy_WinLoss_Train_List,
                       'Accuracy_Validation_WL': Accuracy_WinLoss_Valid_List,
                       'Accuracy_Train_BO': Accuracy_WinLossBlowout_Train_List,
                       'Accuracy_Validation_BO': Accuracy_WinLossBlowout_Valid_List
                       })

Output.to_csv("nn_layer_node/nn_output.csv")
