# save features:
# --images-list classes.csv --features-only  --save-features "~\Examen"
# les fonctionnalités sont sauvegardées au format pickle
# on charge la liste d'images à partir du fichier classes.csv
# il y a 3 colonnes path , class et title(du livre)
# on réduit à 12x16 pixels et utiliser le vecteur de valeurs de 64 pixels comme caractéristiques.
# X_train/y_trainet X_test/y_test sont crées avec train_test_split
# les résultats avec svm sont moins bons par rapport à KNN et logistic Regression 
# j'aurais voulu essayé avec Xgboost pour voir si je peux obtenir plus que 0.95 


# KNN:
#--load-features save_features.pickle --nearest-neighbors 1
# 1 is the best KNN neighbor according to the result on the valid set
# accuracy = 0,93

# LogisticRegression
# --load-features save_features.pickle  --logistic-regression

 2018-01-15 15:25:47,032 - INFO - Train set size is (7322, 64)
2018-01-15 15:25:47,033 - INFO - Test set size is (1831, 64)
2018-01-15 15:25:47,033 - INFO - Use the logisitic model to classify the MINST data
2018-01-15 15:26:16,375 - INFO - ------------------------------------TRAIN SET--------------------------------
2018-01-15 15:26:21,876 - INFO - accuracy = 0.9582081398524993
2018-01-15 15:26:21,914 - INFO - confusion matrix:
 [[ 460    0    0    3    0   30    0]
 [   0   27    4    0    0    8    1]
 [   0    1  217    1    1   74    1]
 [   5    0    0  289    1   22    0]
 [   1    0    3   10  153   12    0]
 [   7    0    0   36    0 5788    5]
 [   3    0    2    1    0   74   82]]
2018-01-15 15:26:21,996 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       0.97      0.93      0.95       493
        miniature       0.96      0.68      0.79        40
miniature + texte       0.96      0.74      0.83       295
      pageblanche       0.85      0.91      0.88       317
          reliure       0.99      0.85      0.92       179
            texte       0.96      0.99      0.98      5836
texte + miniature       0.92      0.51      0.65       162

      avg / total       0.96      0.96      0.96      7322

2018-01-15 15:26:21,996 - INFO - ------------------------------------TEST SET--------------------------------
2018-01-15 15:26:23,483 - INFO - accuracy = 0.926815947569634
2018-01-15 15:26:23,492 - INFO - confusion matrix:
 [[ 104    0    0    5    0   10    0]
 [   0    5    3    0    0    3    0]
 [   0    0   33    0    0   33    0]
 [   3    0    0   69    0   10    0]
 [   2    0    0    6   43    2    0]
 [   7    0    3   13    0 1427    1]
 [   0    0    0    0    0   33   16]]
2018-01-15 15:26:23,502 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       0.90      0.87      0.89       119
        miniature       1.00      0.45      0.62        11
miniature + texte       0.85      0.50      0.63        66
      pageblanche       0.74      0.84      0.79        82
          reliure       1.00      0.81      0.90        53
            texte       0.94      0.98      0.96      1451
texte + miniature       0.94      0.33      0.48        49

      avg / total       0.93      0.93      0.92      1831

2018-01-15 15:26:23,502 - INFO - ------------------------------------TRAIN SET--------------------------------
2018-01-15 15:26:23,517 - INFO - accuracy = 0.8628789948101612
2018-01-15 15:26:23,550 - INFO - confusion matrix:
 [[ 167    0    1   13    0  312    0]
 [   0    8    1    1    0   30    0]
 [   6    1  109    0    8  171    0]
 [  11    0    1  145    2  158    0]
 [  10    0    3    5  123   37    1]
 [  34    0   12   24    4 5761    1]
 [   4    0    4    0    0  149    5]]
2018-01-15 15:26:23,601 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       0.72      0.34      0.46       493
        miniature       0.89      0.20      0.33        40
miniature + texte       0.83      0.37      0.51       295
      pageblanche       0.77      0.46      0.57       317
          reliure       0.90      0.69      0.78       179
            texte       0.87      0.99      0.93      5836
texte + miniature       0.71      0.03      0.06       162

      avg / total       0.85      0.86      0.84      7322

2018-01-15 15:26:23,601 - INFO - ------------------------------------TEST SET--------------------------------
2018-01-15 15:26:23,605 - INFO - accuracy = 0.8607318405243036
2018-01-15 15:26:23,613 - INFO - confusion matrix:
 [[  48    0    0    5    0   66    0]
 [   0    0    0    0    1   10    0]
 [   2    1   18    0    3   41    1]
 [   4    0    0   42    0   36    0]
 [   4    1    5    1   37    5    0]
 [  12    1    6    5    0 1427    0]
 [   1    0    0    0    0   44    4]]
2018-01-15 15:26:23,629 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       0.68      0.40      0.51       119
        miniature       0.00      0.00      0.00        11
miniature + texte       0.62      0.27      0.38        66
      pageblanche       0.79      0.51      0.62        82
          reliure       0.90      0.70      0.79        53
            texte       0.88      0.98      0.93      1451
texte + miniature       0.80      0.08      0.15        49

      avg / total       0.84      0.86      0.84      1831
# LearningCurve with no limit samples
# --load-features save_features.pickle --learning-curve 
#--limit-sample 30000


# TestingCurve with no limit samples
# --load-features save_features.pickle --testing-curve 
#--limit-sample 30000


# SVM kernel linear 
# --load-features save_features.pickle --kernel linear --limit-sample 2000

2018-01-15 15:48:43,677 - INFO - Train set size is (1600, 64)
2018-01-15 15:48:43,677 - INFO - Test set size is (400, 64)
2018-01-15 15:48:43,677 - INFO - Use the svm model to classify the classes data
2018-01-15 15:48:43,677 - INFO - we'll use a linear kernel
2018-01-15 15:48:43,678 - INFO - we'll continue
2018-01-15 15:52:33,996 - INFO - we'll get there
2018-01-15 15:52:34,083 - INFO - accuracy = 0.913125
2018-01-15 15:52:34,088 - INFO - confusion matrix:
 [[419   0   0   0   0  54   0]
 [  0  13   0   0   0   0   0]
 [  5   0  71   0   0   9   0]
 [  0   0   0 124   0   3   0]
 [  0   0   0   0  85   0   0]
 [ 62   0   2   3   1 716   0]
 [  0   0   0   0   0   0  33]]
2018-01-15 15:52:34,100 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       0.86      0.89      0.87       473
        miniature       1.00      1.00      1.00        13
miniature + texte       0.97      0.84      0.90        85
      pageblanche       0.98      0.98      0.98       127
          reliure       0.99      1.00      0.99        85
            texte       0.92      0.91      0.91       784
texte + miniature       1.00      1.00      1.00        33

      avg / total       0.91      0.91      0.91      1600

2018-01-15 15:52:34,100 - INFO - Almost done
2018-01-15 15:52:34,122 - INFO - accuracy = 0.7625
2018-01-15 15:52:34,124 - INFO - confusion matrix:
 [[ 87   0   0   8   0  18   0]
 [  0   3   0   0   0   1   0]
 [  1   1   9   3   1   4   1]
 [  3   0   0  32   2   3   0]
 [  1   0   2   3  20   1   0]
 [ 22   0   3   7   1 153   4]
 [  3   0   0   0   0   2   1]]
2018-01-15 15:52:34,128 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       0.74      0.77      0.76       113
        miniature       0.75      0.75      0.75         4
miniature + texte       0.64      0.45      0.53        20
      pageblanche       0.60      0.80      0.69        40
          reliure       0.83      0.74      0.78        27
            texte       0.84      0.81      0.82       190
texte + miniature       0.17      0.17      0.17         6

      avg / total       0.77      0.76      0.76       400

2018-01-15 15:52:34,128 - INFO - Finally

# SVM kernel RBF 
# --load-features save_features.pickle --kernel RBF --limit-sample 2000

2018-01-15 15:53:22,807 - INFO - Train set size is (1600, 64)
2018-01-15 15:53:22,807 - INFO - Test set size is (400, 64)
2018-01-15 15:53:22,807 - INFO - Use the svm model to classify the classes data
2018-01-15 15:53:22,807 - INFO - we'll use a RBF kernel
2018-01-15 15:53:22,807 - INFO - we'll continue
2018-01-15 15:53:24,228 - INFO - we'll get there
2018-01-15 15:53:24,660 - INFO - accuracy = 1.0
2018-01-15 15:53:24,668 - INFO - confusion matrix:
 [[473   0   0   0   0   0   0]
 [  0  13   0   0   0   0   0]
 [  0   0  85   0   0   0   0]
 [  0   0   0 127   0   0   0]
 [  0   0   0   0  85   0   0]
 [  0   0   0   0   0 784   0]
 [  0   0   0   0   0   0  33]]
2018-01-15 15:53:24,679 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       1.00      1.00      1.00       473
        miniature       1.00      1.00      1.00        13
miniature + texte       1.00      1.00      1.00        85
      pageblanche       1.00      1.00      1.00       127
          reliure       1.00      1.00      1.00        85
            texte       1.00      1.00      1.00       784
texte + miniature       1.00      1.00      1.00        33

      avg / total       1.00      1.00      1.00      1600

2018-01-15 15:53:24,679 - INFO - Almost done
2018-01-15 15:53:24,782 - INFO - accuracy = 0.5075
2018-01-15 15:53:24,784 - INFO - confusion matrix:
 [[  9   0   0   0   0 104   0]
 [  0   0   0   0   0   4   0]
 [  0   0   0   0   0  20   0]
 [  0   0   0   1   0  39   0]
 [  0   0   0   0   3  24   0]
 [  0   0   0   0   0 190   0]
 [  0   0   0   0   0   6   0]]
C:\Users\Ramez Aissaoui\lab-python-raissaoui\venv\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
2018-01-15 15:53:24,787 - INFO - Classification report: 
                    precision    recall  f1-score   support

       calendrier       1.00      0.08      0.15       113
        miniature       0.00      0.00      0.00         4
miniature + texte       0.00      0.00      0.00        20
      pageblanche       1.00      0.03      0.05        40
          reliure       1.00      0.11      0.20        27
            texte       0.49      1.00      0.66       190
texte + miniature       0.00      0.00      0.00         6

      avg / total       0.68      0.51      0.37       400

2018-01-15 15:53:24,787 - INFO - Finally



