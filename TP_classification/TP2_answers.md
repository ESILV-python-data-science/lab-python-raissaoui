# params pour sauvergader les features (première étape) Question 1,2,3:
#  --images-list MNIST_all.csv --features-only  --save-features "~\TP_classficiation"
# Question 1:

# Charger une liste d'images à partir du fichier CSV en MNIST_all.csv utilisant pd.read_csv:
# region Answer 1
    file_list = pd.read_csv(args.images_list, header=None, names=['filename', 'class'], engine='python')
# endregion

# Question 2:

# La fonction calcule une image sous-résolution 8x8 et renvoie un vecteur de caractéristiques avec des valeurs de pixels.
# 0 doit correspondre au blanc et 255 au noir.
# region Answer 2
    # convert color images to grey level

    gray_img = img.convert('L')
    # find the min dimension to rotate the image if needed
    min_size = min(img.size)
    if img.size[1] == min_size:
        # convert landscape  to portrait
        rotated_img = gray_img.rotate(90, expand=1)
    else:
        rotated_img = gray_img

    # reduce the image to a given size
    reduced_img = rotated_img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]
# endregion
    
# Question 3:
# region Answer 3
        # Convert the features to dataframe with pd.DataFrame 
        # and save to a file in pickle format with to_pickle
        if args.save_features:
            # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle
            df_features = pd.DataFrame(X)
            df_features['class'] = y
            df_features.to_pickle('save_features.pickle')
        # activate the --load-features option to read the features 
        # from the pickle file instead of calculating the features from the images.   
        if args.load_features:
            df_features = pd.read_pickle(args.load_features)
            y = df_features['class']
            X = df_features.drop('class', axis=1)
# endregion            

# params pour  créer les k-nearest neighbour Question 4,5, 6
#  --load-features save_features.pickle --nearest-neighbors 1 --limit-samples 10000

# Question 4 and 5:
# region Answer 4 and 5

        def nearestNeighbors_train(X_train, y_train, k=MISSING, limit=10):
            # s'entrainer sur une serie de KNN
            # elle renvoie une liste contenant tout les modèles
            # meme si on utilise qu'une seule valeur  (si K is not MISSING), il faut renvoyer une liste
            # pour avoir une facon unique d'évaleur le modèle  (cf. nearestNeighbors_test)
            t0 = time.time()
            knns = []
            if k is MISSING:
                # train with multiple values
                for i in range(1, limit + 1):
                    knns.append(knn_train(i, X_train, y_train))
            else:
                # train with one value only
                knns.append(knn_train(k, X_train, y_train))
            logger.info("Training  done in %0.3fs" % (time.time() - t0))
            return knns
            
        /***********/
        
        def nearestNeighbors_test(knns, X_train, y_train, X_test, y_test, X_valid=MISSING, y_valid=MISSING):
            # tester les knn modèles sur leurs bases VALID (pas test) et recupérer leurs accuracy
            # Si knns.length > 1 :  plotter leurs accuracy en fonction de K
            # Si knns.length == 1 : évaluer son accuracy et s'arreter la .
            t0 = time.time()
            k = 1
            accuracies = []
            for knn in knns:
                logger.info("Testing KNN with k={}".format(k))
                accuracies.append(test_summary(knn, X_train, y_train, X_test, y_test, X_valid, y_valid))
                k += 1
            logger.info("Testing  done in %0.3fs" % (time.time() - t0))
            if k != 2:  # when it's more than one element in the accuracies' list
                plt.plot(range(1, k), accuracies, marker='o', linestyle='--', color='r')
                plt.axis([1, k, min(accuracies) - 0.005, max(accuracies) + 0.005])
                plt.xlabel('KNN neighbors')
                plt.ylabel('Accuracy')
                plt.title('Accuracies with different K in KNN algorithm')
                plt.show()
        
        /****************/
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            logger.info("Train set size is {}".format(X_train.shape))
            logger.info("Test set size is {}".format(X_test.shape))
            knnForOneNeighbor = nearestNeighbors_train(X_train, y_train, k=1)
            # train with one value for k only (ici 1) mais peut-être remplacé par k=args.nearest_neighbors
            #  pour définir le nombre de cluster par les paramètres
            nearestNeighbors_test(knnForOneNeighbor, X_train, y_train, X_test, y_test)
# endregion

# Resultat 4 and 5:
Accuracies with different k in KNN algorithm.png
# endregion
            
# Question 5':
# region Answer 5'
    def test(LearningObject, X, y, other_options=False):
        predictions = LearningObject.predict(X)
        accuracy = accuracy_score(y, predictions)
        if other_options:
            logger.info(
                "accuracy = {}".format(accuracy))
            logger.info(
                "confusion matrix:\n {}".format(metrics.confusion_matrix(y, predictions)))
            logger.info(
                "Classification report: \n {}".format(classification_report(y, predictions)))
        return accuracy
# endregion         

# Question 6:
# region Answer 6
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=0)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))
        logger.info("Valid set size is {}".format(X_valid.shape))
        knnsForNeighbors = nearestNeighbors_train(X_train, y_train,
                                                  limit=20)  # train with different values from 1 to (limit can be changed!)
        nearestNeighbors_test(knnsForNeighbors, X_train, y_train, X_test, y_test, X_valid, y_valid)
        ## 3 is the best KNN neighborging according to the result on the valid set!!
# endregion

# Resultat 4, 5 and 6
    2017-12-31 12:30:28,106 - INFO - Use kNN classifier with k = 1
    2017-12-31 12:30:28,110 - INFO - Train set size is (8000, 64)
    2017-12-31 12:30:28,110 - INFO - Test set size is (2000, 64)
    2017-12-31 12:30:28,248 - INFO - Training  done in 0.139s
    2017-12-31 12:30:28,249 - INFO - Testing KNN with k=1
    2017-12-31 12:30:28,249 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:30:31,692 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:30:33,200 - INFO - Testing  done in 4.951s
    2017-12-31 12:30:33,205 - INFO - Train set size is (6400, 64)
    2017-12-31 12:30:33,205 - INFO - Test set size is (2000, 64)
    2017-12-31 12:30:33,205 - INFO - Valid set size is (1600, 64)
    2017-12-31 12:30:34,632 - INFO - Training  done in 1.426s
    2017-12-31 12:30:34,632 - INFO - Testing KNN with k=1
    2017-12-31 12:30:34,632 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:30:36,712 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:30:37,849 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:30:38,738 - INFO - accuracy = 0.995625
    2017-12-31 12:30:38,740 - INFO - confusion matrix:
     [[661   2]
     [  5 932]]
    2017-12-31 12:30:38,741 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       1.00      1.00      1.00      1600
    
    2017-12-31 12:30:38,741 - INFO - Testing KNN with k=2
    2017-12-31 12:30:38,741 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:30:42,754 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:30:43,975 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:30:44,939 - INFO - accuracy = 0.995625
    2017-12-31 12:30:44,940 - INFO - confusion matrix:
     [[663   0]
     [  7 930]]
    2017-12-31 12:30:44,941 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       1.00      1.00      1.00      1600
    
    2017-12-31 12:30:44,941 - INFO - Testing KNN with k=3
    2017-12-31 12:30:44,941 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:30:49,566 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:30:50,808 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:30:51,835 - INFO - accuracy = 0.998125
    2017-12-31 12:30:51,837 - INFO - confusion matrix:
     [[662   1]
     [  2 935]]
    2017-12-31 12:30:51,838 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       1.00      1.00      1.00       663
              9       1.00      1.00      1.00       937
    
    avg / total       1.00      1.00      1.00      1600
    
    2017-12-31 12:30:51,839 - INFO - Testing KNN with k=4
    2017-12-31 12:30:51,839 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:30:56,620 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:30:57,911 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:30:59,567 - INFO - accuracy = 0.995625
    2017-12-31 12:30:59,570 - INFO - confusion matrix:
     [[662   1]
     [  6 931]]
    2017-12-31 12:30:59,571 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       1.00      1.00      1.00      1600
    
    2017-12-31 12:30:59,571 - INFO - Testing KNN with k=5
    2017-12-31 12:30:59,571 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:06,282 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:31:08,388 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:31:09,902 - INFO - accuracy = 0.995625
    2017-12-31 12:31:09,905 - INFO - confusion matrix:
     [[661   2]
     [  5 932]]
    2017-12-31 12:31:09,906 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       1.00      1.00      1.00      1600
    
    2017-12-31 12:31:09,906 - INFO - Testing KNN with k=6
    2017-12-31 12:31:09,907 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:14,112 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:31:15,421 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:31:16,485 - INFO - accuracy = 0.994375
    2017-12-31 12:31:16,487 - INFO - confusion matrix:
     [[661   2]
     [  7 930]]
    2017-12-31 12:31:16,488 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:31:16,488 - INFO - Testing KNN with k=7
    2017-12-31 12:31:16,488 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:21,306 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:31:22,671 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:31:23,838 - INFO - accuracy = 0.994375
    2017-12-31 12:31:23,840 - INFO - confusion matrix:
     [[661   2]
     [  7 930]]
    2017-12-31 12:31:23,842 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:31:23,842 - INFO - Testing KNN with k=8
    2017-12-31 12:31:23,842 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:28,242 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:31:30,026 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:31:31,188 - INFO - accuracy = 0.994375
    2017-12-31 12:31:31,190 - INFO - confusion matrix:
     [[661   2]
     [  7 930]]
    2017-12-31 12:31:31,191 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:31:31,191 - INFO - Testing KNN with k=9
    2017-12-31 12:31:31,191 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:35,840 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:31:37,237 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:31:38,391 - INFO - accuracy = 0.994375
    2017-12-31 12:31:38,393 - INFO - confusion matrix:
     [[660   3]
     [  6 931]]
    2017-12-31 12:31:38,394 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:31:38,394 - INFO - Testing KNN with k=10
    2017-12-31 12:31:38,394 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:42,897 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:31:44,564 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:31:45,772 - INFO - accuracy = 0.99375
    2017-12-31 12:31:45,774 - INFO - confusion matrix:
     [[660   3]
     [  7 930]]
    2017-12-31 12:31:45,775 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:31:45,775 - INFO - Testing KNN with k=11
    2017-12-31 12:31:45,775 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:50,573 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:31:52,055 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:31:53,300 - INFO - accuracy = 0.994375
    2017-12-31 12:31:53,303 - INFO - confusion matrix:
     [[660   3]
     [  6 931]]
    2017-12-31 12:31:53,304 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      1.00       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:31:53,304 - INFO - Testing KNN with k=12
    2017-12-31 12:31:53,304 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:31:58,560 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:00,178 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:01,656 - INFO - accuracy = 0.99375
    2017-12-31 12:32:01,663 - INFO - confusion matrix:
     [[660   3]
     [  7 930]]
    2017-12-31 12:32:01,666 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:01,666 - INFO - Testing KNN with k=13
    2017-12-31 12:32:01,666 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:32:07,017 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:08,583 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:09,827 - INFO - accuracy = 0.99375
    2017-12-31 12:32:09,829 - INFO - confusion matrix:
     [[660   3]
     [  7 930]]
    2017-12-31 12:32:09,830 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:09,830 - INFO - Testing KNN with k=14
    2017-12-31 12:32:09,830 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:32:14,594 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:16,062 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:17,208 - INFO - accuracy = 0.99375
    2017-12-31 12:32:17,210 - INFO - confusion matrix:
     [[660   3]
     [  7 930]]
    2017-12-31 12:32:17,211 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:17,211 - INFO - Testing KNN with k=15
    2017-12-31 12:32:17,211 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:32:21,946 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:23,422 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:24,574 - INFO - accuracy = 0.99375
    2017-12-31 12:32:24,576 - INFO - confusion matrix:
     [[660   3]
     [  7 930]]
    2017-12-31 12:32:24,577 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:24,577 - INFO - Testing KNN with k=16
    2017-12-31 12:32:24,577 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:32:29,122 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:30,843 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:32,231 - INFO - accuracy = 0.993125
    2017-12-31 12:32:32,233 - INFO - confusion matrix:
     [[660   3]
     [  8 929]]
    2017-12-31 12:32:32,234 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:32,234 - INFO - Testing KNN with k=17
    2017-12-31 12:32:32,234 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:32:37,193 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:38,656 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:39,953 - INFO - accuracy = 0.993125
    2017-12-31 12:32:39,955 - INFO - confusion matrix:
     [[660   3]
     [  8 929]]
    2017-12-31 12:32:39,956 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:39,956 - INFO - Testing KNN with k=18
    2017-12-31 12:32:39,956 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:32:47,291 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:49,646 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:51,382 - INFO - accuracy = 0.993125
    2017-12-31 12:32:51,385 - INFO - confusion matrix:
     [[660   3]
     [  8 929]]
    2017-12-31 12:32:51,387 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:51,387 - INFO - Testing KNN with k=19
    2017-12-31 12:32:51,387 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:32:56,890 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:32:58,473 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:32:59,668 - INFO - accuracy = 0.993125
    2017-12-31 12:32:59,669 - INFO - confusion matrix:
     [[660   3]
     [  8 929]]
    2017-12-31 12:32:59,670 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:32:59,671 - INFO - Testing KNN with k=20
    2017-12-31 12:32:59,671 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:33:06,905 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:33:09,240 - INFO - ------------------------------------VALID SET--------------------------------
    2017-12-31 12:33:10,866 - INFO - accuracy = 0.993125
    2017-12-31 12:33:10,868 - INFO - confusion matrix:
     [[660   3]
     [  8 929]]
    2017-12-31 12:33:10,869 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       663
              9       1.00      0.99      0.99       937
    
    avg / total       0.99      0.99      0.99      1600
    
    2017-12-31 12:33:10,869 - INFO - Testing  done in 156.238s
    
     K = 3 est la meilleur valeur car la précision (Accuracy) est bien élevée et aussi pocède une error minimale entre le training and testing
# endregion

# params pour  créer les regressions logisitques  Question 7
#  --load-features save_features.pickle  --logistic-regression
# Question 7:
    
# region Answer 7
    elif args.logistic_regression:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))
        logger.info('Use the logisitic model to classify the MINST data')
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        knn = knn_train(3, X_train, y_train)  # 3 is the best KNN value
        test_summary(knn, X_train, y_train, X_test, y_test)
        test_summary(logreg, X_train, y_train, X_test, y_test)
# endregion

# Resultat 7
    2017-12-31 12:37:36,538 - INFO - Train set size is (48000, 64)
    2017-12-31 12:37:36,538 - INFO - Test set size is (12000, 64)
    2017-12-31 12:37:36,538 - INFO - Use the logisitic model to classify the MINST data
    2017-12-31 12:40:22,549 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:46:48,755 - INFO - accuracy = 0.96125
    2017-12-31 12:46:48,799 - INFO - confusion matrix:
     [[4697    8    3    3    0    6   16    0   17    0]
     [   1 5348   11    7    3    2    3   13    7   10]
     [  17   21 4680   22    8    5   11   41   19    7]
     [  17   32   48 4634    3   50    4   30   52   32]
     [  14   27    4    2 4462    1   24   29    4  102]
     [  20   11    6   65   11 4134   28    5   29   21]
     [  11   11    6    0    6   12 4674    0    7    0]
     [   7   35   13    8   46    2    0 4770    5  104]
     [  35   73   19   54   20   52   19   29 4279   42]
     [  11   24    3   27   73   27    1  130   16 4462]]
    2017-12-31 12:46:48,819 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.97      0.99      0.98      4750
              1       0.96      0.99      0.97      5405
              2       0.98      0.97      0.97      4831
              3       0.96      0.95      0.95      4902
              4       0.96      0.96      0.96      4669
              5       0.96      0.95      0.96      4330
              6       0.98      0.99      0.98      4727
              7       0.95      0.96      0.95      4990
              8       0.96      0.93      0.94      4622
              9       0.93      0.93      0.93      4774
    
    avg / total       0.96      0.96      0.96     48000
    
    2017-12-31 12:46:48,820 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:48:20,414 - INFO - accuracy = 0.92725
    2017-12-31 12:48:20,424 - INFO - confusion matrix:
     [[1150    0    3    0    1    8    6    1    4    0]
     [   2 1313    5    2    2    2    3    2    3    3]
     [   8   16 1063    6    1    2    4   17    7    3]
     [   8   15   24 1080    1   29    0   20   28   24]
     [   3   10    5    0 1084    0    3   21    1   46]
     [   7    5    4   32    1 1000   20    1   14    7]
     [   5    6    1    0    1    8 1163    0    6    1]
     [   3   18    5    3   22    2    0 1160    2   60]
     [  19   35    6   21   11   28    9    9 1057   34]
     [   6   12    1    8   25    5    0   58    3 1057]]
    2017-12-31 12:48:20,429 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.95      0.98      0.96      1173
              1       0.92      0.98      0.95      1337
              2       0.95      0.94      0.95      1127
              3       0.94      0.88      0.91      1229
              4       0.94      0.92      0.93      1173
              5       0.92      0.92      0.92      1091
              6       0.96      0.98      0.97      1191
              7       0.90      0.91      0.90      1275
              8       0.94      0.86      0.90      1229
              9       0.86      0.90      0.88      1175
    
    avg / total       0.93      0.93      0.93     12000
    
    2017-12-31 12:48:20,430 - INFO - ------------------------------------TRAIN SET--------------------------------
    2017-12-31 12:48:20,471 - INFO - accuracy = 0.8604791666666667
    2017-12-31 12:48:20,510 - INFO - confusion matrix:
     [[4516    6   16   27   14   37   54    1   74    5]
     [   1 5131   38   34   17   70   13    4   81   16]
     [  48  107 4151  101   79   47   57   69  146   26]
     [  40  114  210 3985   12  196   28   66  153   98]
     [  19   65   18    9 4093   20   53   43  104  245]
     [  68   32   48  267   84 3343  133   19  251   85]
     [  39   46   25    2   23   83 4465   17   26    1]
     [  22   72   60   63  120   18    1 4312   15  307]
     [  55  186   83  231   23  289   41   38 3579   97]
     [  59  130   29   53  176   55    2  453   89 3728]]
    2017-12-31 12:48:20,531 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.93      0.95      0.94      4750
              1       0.87      0.95      0.91      5405
              2       0.89      0.86      0.87      4831
              3       0.84      0.81      0.82      4902
              4       0.88      0.88      0.88      4669
              5       0.80      0.77      0.79      4330
              6       0.92      0.94      0.93      4727
              7       0.86      0.86      0.86      4990
              8       0.79      0.77      0.78      4622
              9       0.81      0.78      0.79      4774
    
    avg / total       0.86      0.86      0.86     48000
    
    2017-12-31 12:48:20,531 - INFO - ------------------------------------TEST SET--------------------------------
    2017-12-31 12:48:20,540 - INFO - accuracy = 0.8588333333333333
    2017-12-31 12:48:20,550 - INFO - confusion matrix:
     [[1112    1    6    3    1    8   23    1   17    1]
     [   1 1265    8    7    3   20    6    0   26    1]
     [   8   31  957   23   22    6   19   20   33    8]
     [  13   27   50  985    4   49   10   26   31   34]
     [  11   23    6    4 1027    4    8   19   23   48]
     [  17   11    8   69   18  871   35    2   41   19]
     [   5    9    6    1    3   25 1127    5   10    0]
     [   5   20   15   11   37    4    0 1089   13   81]
     [  16   53   23   68    8   82   13    3  937   26]
     [  13   28    4    8   35   12    0  120   19  936]]
    2017-12-31 12:48:20,556 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.93      0.95      0.94      1173
              1       0.86      0.95      0.90      1337
              2       0.88      0.85      0.87      1127
              3       0.84      0.80      0.82      1229
              4       0.89      0.88      0.88      1173
              5       0.81      0.80      0.80      1091
              6       0.91      0.95      0.93      1191
              7       0.85      0.85      0.85      1275
              8       0.81      0.76      0.79      1229
              9       0.81      0.80      0.80      1175
    
    avg / total       0.86      0.86      0.86     12000

# params pour  créer les regressions logisitques  Question 8
#  --load-features save_features.pickle --learning-curve --limit-sample 10000

# Question 8:

# region Answer 8:

        def knn_train(k, X_train, y_train):
            # knn_train() il permet juste de lancer un model d'apprentissage knn en lui donnant le K souhaité
            knnObject = KNeighborsClassifier(n_neighbors=k)
            knnObject.fit(X_train, y_train)
            return knnObject
            
            /*******/
            
        elif args.learning_curve:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            logger.info("Initial train set size is {}".format(X_train.shape))
            logger.info("Test set size is {}".format(X_test.shape))
            percents = [1, 10, 20, 40, 50, 80, 100]
            logisiticsTestAccuracies = []
            logisiticsTrainAccuracies = []
            knnsTestAccuracies = []
            knnsTrainAccuracies = []
            knnsToTest = [1,2,3]
            for i in knnsToTest: # for 3 KNNs
                knnsTestAccuracies.append([])
                knnsTrainAccuracies.append([])
            for percent in percents:
                logger.info("using {}% of the training set ".format(percent))
                Nrows = round(percent * len(X_train.index) / 100)
                current_X_train = X_train.head(Nrows)
                current_y_train = y_train.head(Nrows)
                logger.info("Train set size is {}".format(current_X_train.shape))
                #logistic
                logreg = LogisticRegression()
                logreg.fit(current_X_train, current_y_train)
                logisiticsTrainAccuracies.append(test(logreg, current_X_train, current_y_train))
                logisiticsTestAccuracies.append(test(logreg, X_test, y_test))
                counter = 0
                for i in knnsToTest:
                    knn = knn_train(i,current_X_train, current_y_train)
                    knnsTrainAccuracies[counter].append(test(knn, current_X_train, current_y_train))
                    knnsTestAccuracies[counter].append(test(knn, X_test, y_test))
                    counter+=1

        temp= logisiticsTestAccuracies + logisiticsTrainAccuracies
        plt.plot(percents, logisiticsTrainAccuracies, marker='o', linestyle='--', color='r', label="Train accuracy")
        plt.plot(percents, logisiticsTestAccuracies, marker='o', linestyle='--', color='b', label="Test accuracy")
        plt.axis([1, 100, (min(temp)) - 0.005, (max(temp) + 0.005)])
        plt.xlabel('Training set size (percent)')
        plt.ylabel('Accuracy')
        plt.title('Accuracies using logistic model with different training set size')
        plt.legend()
        plt.show()
        counter = 0
        for i in range(0, 3):
            temp = knnsTestAccuracies[i] + knnsTrainAccuracies[i]
            plt.plot(percents, knnsTrainAccuracies[i], marker='o', linestyle='--', color='r', label="Train accuracy")
            plt.plot(percents, knnsTestAccuracies[i], marker='o', linestyle='--', color='b', label="Test accuracy")
            plt.axis([1, 100, (min(temp)) - 0.005, (max(temp) + 0.005)])
            plt.xlabel('Training set size (percent)')
            plt.ylabel('Accuracy')
            plt.title('Accuracies using KNN{} model with different training set size'.format(knnsToTest[counter]))
            plt.legend()
            plt.show()
            counter+=1
# endregion
 # 1300 est un echantillon tres petit, puisque en plus on prend que 1% de valeurs dans le train donc le train ne s'entraine pas suffisament et créer des erruers
# c'est pourquoi j'ai préféré  travailler avec 10K, ce qui est plus representatif et montre la difference de performance

# Resultat 8
    2017-12-31 12:52:00,572 - INFO - Initial train set size is (8000, 64)
    2017-12-31 12:52:00,572 - INFO - Test set size is (2000, 64)
    2017-12-31 12:52:00,572 - INFO - using 1% of the training set 
    2017-12-31 12:52:00,572 - INFO - Train set size is (80, 64)
    2017-12-31 12:52:00,646 - INFO - using 10% of the training set 
    2017-12-31 12:52:00,647 - INFO - Train set size is (800, 64)
    2017-12-31 12:52:01,337 - INFO - using 20% of the training set 
    2017-12-31 12:52:01,337 - INFO - Train set size is (1600, 64)
    2017-12-31 12:52:02,933 - INFO - using 40% of the training set 
    2017-12-31 12:52:02,933 - INFO - Train set size is (3200, 64)
    2017-12-31 12:52:07,142 - INFO - using 50% of the training set 
    2017-12-31 12:52:07,142 - INFO - Train set size is (4000, 64)
    2017-12-31 12:52:14,066 - INFO - using 80% of the training set 
    2017-12-31 12:52:14,067 - INFO - Train set size is (6400, 64)
    2017-12-31 12:52:28,903 - INFO - using 100% of the training set 
    2017-12-31 12:52:28,903 - INFO - Train set size is (8000, 64)
    
    
    Accuracies using KNN0 model with different training set size.png
    
    Accuracies using KNN1 model with different training set size.png
    
    Accuracies using KNN2 model with different training set size.png
    
    Accuracies using KNN3 model with different training set size.png

# params pour  créer les regressions logisitques  Question 9
#  --load-features save_features.pickle --svm


# --load-features save_features.pickle --testing-curve --limit-sample 1000
# Question 9
# region Answer 9

    if args.kernel == "linear" :
                # Answer 9
                logger.info("we'll use a linear kernel")
                clf = SVC(kernel='linear')
                logger.info("we'll continue")
                clf.fit(X_train, y_train)
                logger.info("we'll get there")
                test(clf, X_train, y_train, other_options=True)
                logger.info("Almost done")
                test(clf, X_test, y_test, other_options=True)
                logger.info("Finally")
            
# endregion

# Resultat 9 
    2017-12-31 13:20:34,489 - INFO - Train set size is (8000, 64)
    2017-12-31 13:20:34,489 - INFO - Test set size is (2000, 64)
    2017-12-31 13:20:34,489 - INFO - Use the svm model to classify the MINST data
    2017-12-31 13:20:34,489 - INFO - we'll use a linear kernel
    2017-12-31 13:20:34,489 - INFO - we'll continue
    2017-12-31 13:24:47,593 - INFO - we'll get there
    2017-12-31 13:24:47,699 - INFO - accuracy = 0.99425
    2017-12-31 13:24:47,708 - INFO - confusion matrix:
     [[3220   16]
     [  30 4734]]
    2017-12-31 13:24:47,711 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      3236
              9       1.00      0.99      1.00      4764
    
    avg / total       0.99      0.99      0.99      8000
    
    2017-12-31 13:24:47,711 - INFO - Almost done
    2017-12-31 13:24:47,735 - INFO - accuracy = 0.9905
    2017-12-31 13:24:47,738 - INFO - confusion matrix:
     [[ 803   12]
     [   7 1178]]
    2017-12-31 13:24:47,742 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.99      0.99      0.99       815
              9       0.99      0.99      0.99      1185
    
    avg / total       0.99      0.99      0.99      2000
# endregion 

# Question 10
# region Answer 10
        elif args.testing_curve:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Initial test set size is {}".format(X_test.shape))
        percents = [1, 10, 20, 40, 50, 80, 100]
        logisiticsTestAccuracies = []
        logisiticsTestAccuracies_std = []
        knnsTestAccuracies = []
        knnsTestAccuracies_std = []
        knnsToTest = [1, 2, 3]
        for i in knnsToTest:
            # for 3 KNNs
            knnsTestAccuracies.append([])
            knnsTestAccuracies_std.append([])

        # learning
        # logistic
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        # knns
        knnsModels = []
        for i in knnsToTest:
            knnsModels.append(knn_train(i, X_train, y_train))
        # test decreasing prop
        for percent in percents:
            logger.info("using {}% of the test set ".format(percent))
            Nrows = round(percent * len(X_test.index) / 100)
            logistictemp = []
            knnTemp = []
            for i in knnsToTest:  # for 3 KNNs
                knnTemp.append([])
            for i in range(0,10):
                current_X_test = X_test.sample(frac =percent/100, replace=True)
                current_y_test = y_test.sample(frac = percent/100, replace=True)
                logger.info("Test set size is {}".format(current_X_test.shape))
                logistictemp.append(test(logreg, current_X_test, current_y_test))
                counter = 0
                for knnModel in knnsModels:
                    knnTemp[counter].append(test(knnModel, current_X_test, current_y_test))
                    counter += 1
            logisticTempNumpy = np.array(logistictemp)
            logisiticsTestAccuracies.append(logisticTempNumpy.mean())
            logisiticsTestAccuracies_std.append(logisticTempNumpy.std())
            counter = 0
            for i in knnTemp:
                knnNumpy = np.array(i)
                knnsTestAccuracies[counter].append(knnNumpy.mean())
                knnsTestAccuracies_std[counter].append(knnNumpy.std())
                counter += 1

        plt.errorbar(percents, logisiticsTestAccuracies,logisiticsTestAccuracies_std, linestyle='None', marker='^')
        plt.xlabel('Testing set size (percent)')
        plt.ylabel('Accuracy')
        plt.title('Accuracies using logistic model with different testing set size')
        plt.show()
        counter = 0
        for i in knnsToTest:
            plt.errorbar(percents, knnsTestAccuracies[counter], knnsTestAccuracies_std[counter], linestyle='None', marker='^')
            plt.xlabel('Testing set size (percent)')
            plt.ylabel('Accuracy')
            plt.title('Accuracies using knn{} model with different testing set size'.format(counter))
            plt.show()
            counter += 1
# endregion

# Resultat 10 
    2017-12-31 13:01:36,333 - INFO - Train set size is (8000, 64)
    2017-12-31 13:01:36,333 - INFO - Initial test set size is (2000, 64)
    2017-12-31 13:01:37,302 - INFO - using 1% of the test set 
    2017-12-31 13:01:37,310 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,370 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,427 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,481 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,542 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,592 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,645 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,698 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,751 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,806 - INFO - Test set size is (20, 64)
    2017-12-31 13:01:37,865 - INFO - using 10% of the test set 
    2017-12-31 13:01:37,866 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:38,380 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:38,882 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:39,372 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:39,897 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:40,450 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:40,975 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:41,470 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:41,960 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:42,459 - INFO - Test set size is (200, 64)
    2017-12-31 13:01:42,950 - INFO - using 20% of the test set 
    2017-12-31 13:01:42,951 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:43,959 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:44,968 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:46,049 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:47,045 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:48,035 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:49,074 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:50,124 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:51,391 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:52,713 - INFO - Test set size is (400, 64)
    2017-12-31 13:01:54,020 - INFO - using 40% of the test set 
    2017-12-31 13:01:54,022 - INFO - Test set size is (800, 64)
    2017-12-31 13:01:56,888 - INFO - Test set size is (800, 64)
    2017-12-31 13:01:59,619 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:02,283 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:04,873 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:07,528 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:10,319 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:12,765 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:15,123 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:17,400 - INFO - Test set size is (800, 64)
    2017-12-31 13:02:19,679 - INFO - using 50% of the test set 
    2017-12-31 13:02:19,680 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:22,931 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:25,961 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:29,080 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:32,144 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:35,204 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:38,344 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:41,472 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:44,724 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:48,001 - INFO - Test set size is (1000, 64)
    2017-12-31 13:02:50,977 - INFO - using 80% of the test set 
    2017-12-31 13:02:50,978 - INFO - Test set size is (1600, 64)
    2017-12-31 13:02:55,656 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:00,052 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:05,898 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:12,676 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:17,135 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:22,191 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:27,216 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:32,478 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:37,639 - INFO - Test set size is (1600, 64)
    2017-12-31 13:03:42,754 - INFO - using 100% of the test set 
    2017-12-31 13:03:42,754 - INFO - Test set size is (2000, 64)
    2017-12-31 13:03:50,101 - INFO - Test set size is (2000, 64)
    2017-12-31 13:03:57,095 - INFO - Test set size is (2000, 64)
    2017-12-31 13:04:03,619 - INFO - Test set size is (2000, 64)
    2017-12-31 13:04:10,189 - INFO - Test set size is (2000, 64)
    2017-12-31 13:04:16,828 - INFO - Test set size is (2000, 64)
    2017-12-31 13:04:23,418 - INFO - Test set size is (2000, 64)
    2017-12-31 13:04:29,841 - INFO - Test set size is (2000, 64)
    2017-12-31 13:04:36,466 - INFO - Test set size is (2000, 64)
    2017-12-31 13:04:43,423 - INFO - Test set size is (2000, 64)
    
    
    Accuracies using KNN0 model with different testing set size.png
    
    Accuracies using KNN1 model with different testing set size.png
    
    Accuracies using KNN2 model with different testing set size.png
    
    Accuracies using KNN3 model with different testing set size.png
    
# endregion


# Question 11
# region Answer 11
    logger.info("we'll use a RBF kernel")
                clf = SVC(kernel='rbf')
                logger.info("we'll continue")
                clf.fit(X_train, y_train)
                logger.info("we'll get there")
                test(clf, X_train, y_train, other_options=True)
                logger.info("Almost done")
                test(clf, X_test, y_test, other_options=True)
                logger.info("Finally")
# endregion

# Resultat 11
    2017-12-31 13:27:37,929 - INFO - Train set size is (8000, 64)
    2017-12-31 13:27:37,929 - INFO - Test set size is (2000, 64)
    2017-12-31 13:27:37,929 - INFO - Use the svm model to classify the MINST data
    2017-12-31 13:27:37,929 - INFO - we'll use a RBF kernel
    2017-12-31 13:27:37,929 - INFO - we'll continue
    2017-12-31 13:27:55,812 - INFO - we'll get there
    2017-12-31 13:28:05,198 - INFO - accuracy = 1.0
    2017-12-31 13:28:05,206 - INFO - confusion matrix:
     [[3236    0]
     [   0 4764]]
    2017-12-31 13:28:05,210 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       1.00      1.00      1.00      3236
              9       1.00      1.00      1.00      4764
    
    avg / total       1.00      1.00      1.00      8000
    
    2017-12-31 13:28:05,210 - INFO - Almost done
    2017-12-31 13:28:07,684 - INFO - accuracy = 0.5925
    2017-12-31 13:28:07,686 - INFO - confusion matrix:
     [[   0  815]
     [   0 1185]]
    C:\Program Files (x86)\Python36-32\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    2017-12-31 13:28:07,695 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.00      0.00      0.00       815
              9       0.59      1.00      0.74      1185
    
    avg / total       0.35      0.59      0.44      2000
    
    2017-12-31 13:28:07,695 - INFO - Finally
   
    
# endregion

# Question 11bis
# Answer 11bis
    logger.info("we'll use a RBF kernel, with tuned parameters")
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100, 1000]}]
            clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                               scoring='accuracy')
            clf.fit(X_train, y_train)
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print("Grid scores on development set:")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            test(clf, X_train, y_train, other_options=True)
            test(clf, X_test, y_test, other_options=True)
# endregion
# Resultat 12
    2017-12-31 20:31:52,678 - INFO - Train set size is (8000, 64)
    2017-12-31 20:31:52,678 - INFO - Test set size is (2000, 64)
    2017-12-31 20:31:52,678 - INFO - Use the svm model to classify the MINST data
    2017-12-31 20:31:52,678 - INFO - we'll use a RBF kernel, with tuned parameters
    Best parameters set found on development set:
    {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
    Grid scores on development set:
    0.596 (+/-0.000) for {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}
    0.947 (+/-0.013) for {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.596 (+/-0.000) for {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
    0.949 (+/-0.013) for {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.596 (+/-0.000) for {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    0.949 (+/-0.013) for {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.596 (+/-0.000) for {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
    0.949 (+/-0.013) for {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}
    2017-12-31 20:39:58,202 - INFO - accuracy = 1.0
    2017-12-31 20:39:58,209 - INFO - confusion matrix:
     [[3236    0]
     [   0 4764]]
    2017-12-31 20:39:58,213 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       1.00      1.00      1.00      3236
              9       1.00      1.00      1.00      4764
    
    avg / total       1.00      1.00      1.00      8000
    
    2017-12-31 20:39:59,358 - INFO - accuracy = 0.9435
    2017-12-31 20:39:59,360 - INFO - confusion matrix:
     [[ 814    1]
     [ 112 1073]]
    2017-12-31 20:39:59,361 - INFO - Classification report: 
                  precision    recall  f1-score   support
    
              0       0.88      1.00      0.94       815
              9       1.00      0.91      0.95      1185
    
    avg / total       0.95      0.94      0.94      2000
#  endregion 
# Question 12
# region Answer 12

# endregion
