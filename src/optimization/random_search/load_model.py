# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, auc
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from itertools import cycle

# # Replace 'your_model_filename.joblib' with the actual filename of your saved model
# loaded_model = joblib.load('statlog_bst_rand.joblib')

# test_data_file = 'datasets/statlog+landsat+satellite/sat.tst'

# columns = [
#     'TopLeft1', 'TopLeft2', 'TopLeft3', 'TopLeft4',
#     'TopMiddle1', 'TopMiddle2', 'TopMiddle3', 'TopMiddle4',
#     'TopRight1', 'TopRight2', 'TopRight3', 'TopRight4',
#     'MiddleLeft1', 'MiddleLeft2', 'MiddleLeft3', 'MiddleLeft4',
#     'Center1', 'Center2', 'Center3', 'Center4',
#     'MiddleRight1', 'MiddleRight2', 'MiddleRight3', 'MiddleRight4',
#     'BottomLeft1', 'BottomLeft2', 'BottomLeft3', 'BottomLeft4',
#     'BottomMiddle1', 'BottomMiddle2', 'BottomMiddle3', 'BottomMiddle4',
#     'BottomRight1', 'BottomRight2', 'BottomRight3', 'BottomRight4',
#     'Class'
# ]
# df_tst = pd.read_csv(test_data_file, delim_whitespace=' ', names=columns)
# X_test, y_test = df_tst.drop('Class', axis=1), df_tst['Class']

# print("Unique number of classes: ", y_test.unique())
# y_pred = loaded_model.predict(X_test)

# unique_classes = np.unique(y_test)
# roc_auc_scores = []

# for cls in unique_classes:
#     # Check if there are positive samples for the current class
#     if np.sum(y_test == cls) > 0:
#         # Extract true labels and predicted probabilities for the current class
#         y_true_cls = (y_test == cls).astype(int)
#         if hasattr(loaded_model, "predict_proba"):
#             y_pred_cls = loaded_model.predict_proba(X_test)[:, cls]
#         else:
#             y_pred_cls = loaded_model.decision_function(X_test)[:, cls]

#         # Calculate ROC AUC for the current class
#         roc_auc_cls = roc_auc_score(y_true_cls, y_pred_cls)
#         roc_auc_scores.append(roc_auc_cls)
#     else:
#         roc_auc_scores.append(np.nan)

# # Print ROC AUC scores for each class
# for cls, roc_auc in zip(unique_classes, roc_auc_scores):
#     print(f"Class {cls} ROC AUC: {roc_auc}")

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.4f}')

# # F1-score
# f1 = f1_score(y_test, y_pred, average='weighted')  # 'binary' for binary classification
# print(f'F1-score: {f1:.4f}')

# # AUC-ROC
# # AUC-ROC for multi-class classification
# if hasattr(loaded_model, "predict_proba"):
#     y_score = loaded_model.predict_proba(X_test)
# else:
#     y_score = loaded_model.decision_function(X_test)

# n_classes = y_score.shape[1]  # Assuming each column represents a class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes):
#     try:
#         fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     except ValueError as e:
#         # Handle the case where there are no positive samples for the current class
#         print(f"Skipping class {i} due to no positive samples. Error: {e}")
#         roc_auc[i] = np.nan

# # Micro-average ROC curve and AUC
# try:
#     fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# except ValueError as e:
#     print(f"Micro-average calculation failed. Error: {e}")
#     roc_auc["micro"] = np.nan

# # Macro-average ROC curve and AUC
# fpr_grid = np.linspace(0.0, 1.0, 1000)
# mean_tpr = np.zeros_like(fpr_grid)

# for i in range(n_classes):
#     try:
#         mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
#     except ValueError as e:
#         print(f"Skipping class {i} for macro-average calculation. Error: {e}")

# mean_tpr /= n_classes

# fpr["macro"] = fpr_grid
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Print Macro-averaged One-vs-Rest ROC AUC score
# print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")

# # Plotting the ROC curves
# fig, ax = plt.subplots(figsize=(8, 8))

# # Micro-average ROC curve
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

# # Macro-average ROC curve
# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

# # Individual class ROC curves
# colors = cycle(["aqua", "darkorange", "cornflowerblue"])
# for class_id, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[class_id],
#         tpr[class_id],
#         label=f"ROC curve for Class {class_id} (AUC = {roc_auc[class_id]:.2f})",
#         color=color,
#         linestyle="-",
#         linewidth=2,
#     )

# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Extension of Receiver Operating Characteristic to One-vs-Rest multiclass")
# plt.legend()
# plt.savefig('statp2_auc_roc_ovr_random.png')
# plt.show()





# '''# Replace 'your_model_filename.joblib' with the actual filename of your saved model
# loaded_model = joblib.load('student_bst_rand.joblib')

# dfile_path = 'datasets/student_dropout/student_data.csv'

# df = pd.read_csv(dfile_path, delimiter=';')

# le = LabelEncoder()

# X = df.drop('Target', axis=1)
# y = le.fit_transform(df['Target'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)

# y_pred = loaded_model.predict(X_test)

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.4f}')

# # F1-score
# f1 = f1_score(y_test, y_pred, average='weighted')  # 'binary' for binary classification
# print(f'F1-score: {f1:.4f}')

# # AUC-ROC
# # AUC-ROC for multi-class classification
# y_score = loaded_model.predict_proba(X_test)
# n_classes = len(le.classes_)
# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Micro-average ROC curve and AUC
# fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Macro-average ROC curve and AUC
# fpr_grid = np.linspace(0.0, 1.0, 1000)
# mean_tpr = np.zeros_like(fpr_grid)

# for i in range(n_classes):
#     mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

# mean_tpr /= n_classes

# fpr["macro"] = fpr_grid
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Print Macro-averaged One-vs-Rest ROC AUC score
# print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")

# # Plotting the ROC curves
# fig, ax = plt.subplots(figsize=(8, 8))

# # Micro-average ROC curve
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

# # Macro-average ROC curve
# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

# # Individual class ROC curves
# colors = cycle(["aqua", "darkorange", "cornflowerblue"])
# for class_id, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[class_id],
#         tpr[class_id],
#         label=f"ROC curve for {le.classes_[class_id]} (AUC = {roc_auc[class_id]:.2f})",
#         color=color,
#         linestyle="-",
#         linewidth=2,
#     )

# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Extension of Receiver Operating Characteristic to One-vs-Rest multiclass")
# plt.legend()
# plt.savefig('student_auc_roc_ovr_random.png')
# '''

# '''import joblib
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,classification_report,roc_curve
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from matplotlib import pyplot as plt
# from sklearn.metrics import plot_roc_curve
# # Replace 'your_model_filename.joblib' with the actual filename of your saved model
# loaded_model = joblib.load('student_bst_rand.joblib')

# dfile_path = 'datasets/student_dropout/student_data.csv'

# df = pd.read_csv(dfile_path,delimiter=';')

# le =LabelEncoder()

# X = df.drop('Target',axis = 1)
# y = le.fit_transform(df['Target'])



# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=69,stratify=y)

# y_pred = loaded_model.predict(X_test)

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.4f}')

# # F1-score
# f1 = f1_score(y_test, y_pred, average='weighted')  # 'binary' for binary classification
# print(f'F1-score: {f1:.4f}')

# # AUC-ROC
# # Note: AUC-ROC is applicable for binary classification; for multi-class, you can use 'ovr' strategy
# # AUC-ROC for multi-class classification
# auc_roc = roc_auc_score(y_test, loaded_model.predict_proba(X_test), multi_class='ovr')
# print(f'AUC-ROC: {auc_roc:.4f}')


# print("Classification report: \n", classification_report(y_test,y_pred))
# roc_auc = roc_auc_score(y_test, loaded_model.predict_proba(X_test), multi_class='ovr')

# # Plotting the ROC curve
# fig, ax = plt.subplots(figsize=(8, 8))
# plot_roc_curve(loaded_model, X_test, y_test, ax=ax, name=f'AUC = {roc_auc:.2f}')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.show()'''