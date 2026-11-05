
clc;
clear;
close all;

% Locate project folders
projectRoot = fileparts(fileparts(mfilename('fullpath')));
datasetFolder = fullfile(projectRoot, 'Dataset');
resultsFolder = fullfile(projectRoot, 'Results');

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% Load red wine dataset
redFile = fullfile(datasetFolder, 'winequality-red.csv');
redOpts = detectImportOptions(redFile);
redWine = readtable(redFile, redOpts);

% Load white wine dataset
whiteFile = fullfile(datasetFolder, 'winequality-white.csv');
whiteOpts = detectImportOptions(whiteFile);
whiteWine = readtable(whiteFile, whiteOpts);

% Display dataset information
disp('RED WINE DATASET');
disp(size(redWine));
disp(redWine.Properties.VariableNames);

disp('WHITE WINE DATASET');
disp(size(whiteWine));
disp(whiteWine.Properties.VariableNames);

% Display quality distribution
disp('RED WINE QUALITY COUNTS');
tabulate(redWine.quality);

disp('WHITE WINE QUALITY COUNTS');
tabulate(whiteWine.quality);

%% DUPLICATE ANALYSIS BEFORE REMOVAL

redNumericOriginal = redWine{:,1:12};
whiteNumericOriginal = whiteWine{:,1:12};

redDuplicates = height(redWine) - ...
    size(unique(redNumericOriginal, 'rows'), 1);

whiteDuplicates = height(whiteWine) - ...
    size(unique(whiteNumericOriginal, 'rows'), 1);

commonRecords = intersect(redNumericOriginal, ...
    whiteNumericOriginal, 'rows');

%% REMOVE EXACT DUPLICATES WITHIN EACH WINE TYPE

[~, redIdx] = unique(redNumericOriginal, 'rows', 'stable');
redWine = redWine(redIdx,:);

[~, whiteIdx] = unique(whiteNumericOriginal, 'rows', 'stable');
whiteWine = whiteWine(whiteIdx,:);


%% COMBINE DATASETS

redWine.WineType = repmat("Red", height(redWine), 1);
whiteWine.WineType = repmat("White", height(whiteWine), 1);

wineData = [redWine; whiteWine];

%% RECREATE THREE QUALITY CLASSES

classLabel = strings(height(wineData), 1);

classLabel(wineData.quality <= 4) = "Low";
classLabel(wineData.quality >= 5 & wineData.quality <= 6) = "Medium";
classLabel(wineData.quality >= 7) = "High";

wineData.Class = categorical(classLabel);

disp('DATASET AFTER DUPLICATE REMOVAL');
disp(size(wineData));

disp('UPDATED CLASS DISTRIBUTION');
tabulate(wineData.Class);




%% VISUALIZE CLASS DISTRIBUTION

figure;

classCounts = countcats(wineData.Class);
classNames = categories(wineData.Class);

bar(classCounts);

set(gca, 'XTick', 1:numel(classNames), ...
    'XTickLabel', classNames);

xlabel('Wine Quality Class');
ylabel('Number of Samples');
title('Distribution of Wine Quality Classes');
grid on;

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'Class_Distribution.png'));


%% DATA QUALITY CHECK

numericData = wineData{:, 1:12};

% Check missing values
missingCount = sum(ismissing(numericData), 'all');

% Check duplicate records
[~, uniqueIndices] = unique(numericData, 'rows', 'stable');

duplicateCount = size(numericData, 1) - numel(uniqueIndices);

disp('DATA QUALITY CHECK');
fprintf('Missing values: %d\n', missingCount);
fprintf('Duplicate records: %d\n', duplicateCount);

%% DUPLICATE ANALYSIS

disp('DUPLICATE ANALYSIS');
fprintf('Red wine duplicates: %d\n', redDuplicates);
fprintf('White wine duplicates: %d\n', whiteDuplicates);
fprintf('Records common to both datasets: %d\n', ...
    size(commonRecords, 1));


%% FINAL CLEANED DATASET

disp('FINAL CLEANED DATASET');
fprintf('Total samples: %d\n', height(wineData));

disp('Final class distribution');
tabulate(wineData.Class);

disp('Wine type distribution');
tabulate(wineData.WineType);


%% STATISTICAL PATTERN ANALYSIS

% Extract 11 chemical features
X = wineData{:, 1:11};

featureNames = wineData.Properties.VariableNames(1:11);

% Calculate statistical measures
featureMean = mean(X);
featureVariance = var(X);
featureStd = std(X);

% Display feature statistics
featureStats = table(featureNames', featureMean', ...
    featureVariance', featureStd', ...
    'VariableNames', {'Feature', 'Mean', ...
    'Variance', 'StandardDeviation'});

disp('STATISTICAL FEATURE ANALYSIS');
disp(featureStats);

% Calculate covariance matrix
covMatrix = cov(X);

disp('COVARIANCE MATRIX');
disp(covMatrix);

% Calculate correlation matrix
corrMatrix = corrcoef(X);

disp('CORRELATION MATRIX');
disp(corrMatrix);

%% CORRELATION HEATMAP

figure;

heatmap(featureNames, featureNames, corrMatrix);

title('Correlation Between Wine Chemical Features');

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'Correlation_Heatmap.png'));


%% TRAINING AND TESTING DATA SPLIT

rng(42);

% Extract features and labels
X = wineData{:,1:11};
Y = wineData.Class;

% Stratified 80-20 split
cv = cvpartition(Y, 'HoldOut', 0.20);

XTrainRaw = X(training(cv), :);
XTestRaw = X(test(cv), :);

YTrain = Y(training(cv));
YTest = Y(test(cv));

% Standardize using training data only
trainMean = mean(XTrainRaw);
trainStd = std(XTrainRaw);

trainStd(trainStd == 0) = 1;

XTrain = (XTrainRaw - trainMean) ./ trainStd;
XTest = (XTestRaw - trainMean) ./ trainStd;

% Display split information
disp('TRAINING AND TESTING SPLIT');

fprintf('Training samples: %d\n', size(XTrain,1));
fprintf('Testing samples: %d\n', size(XTest,1));

disp('Training class distribution');
tabulate(YTrain);

disp('Testing class distribution');
tabulate(YTest);


%% PRINCIPAL COMPONENT ANALYSIS (PCA)

% Calculate covariance matrix using training data
pcaCov = cov(XTrain);

% Eigenvalue decomposition
[eigenVectors, eigenValues] = eig(pcaCov);

% Sort eigenvalues in descending order
[eigenValues, idx] = sort(diag(eigenValues), 'descend');
eigenVectors = eigenVectors(:, idx);

% Calculate explained variance
explainedVariance = 100 * eigenValues / sum(eigenValues);
cumulativeVariance = cumsum(explainedVariance);

% Select components retaining at least 95% variance
numPC = find(cumulativeVariance >= 95, 1);

% Select principal component directions
pcaDirections = eigenVectors(:, 1:numPC);

% Project training and testing data
XTrainPCA = XTrain * pcaDirections;
XTestPCA = XTest * pcaDirections;

% Display PCA results
disp('PCA RESULTS');
fprintf('Original features: %d\n', size(XTrain,2));
fprintf('Selected components: %d\n', numPC);
fprintf('Variance retained: %.2f%%\n', cumulativeVariance(numPC));

% Plot explained variance
figure;
plot(1:11, cumulativeVariance, '-o', 'LineWidth', 1.5);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('PCA - Cumulative Explained Variance');
grid on;

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'PCA_Explained_Variance.png'));


%% FISHER DISCRIMINANT ANALYSIS (FDA)

classes = categories(YTrain);
numFeatures = size(XTrain, 2);
numClasses = numel(classes);

overallMean = mean(XTrain);

Sw = zeros(numFeatures);
Sb = zeros(numFeatures);

for i = 1:numClasses

    classData = XTrain(YTrain == classes{i}, :);

    classMean = mean(classData);
    classCount = size(classData, 1);

    % Within-class scatter
    centeredData = classData - classMean;
    Sw = Sw + centeredData' * centeredData;

    % Between-class scatter
    meanDifference = classMean - overallMean;

    Sb = Sb + classCount * ...
        (meanDifference' * meanDifference);

end

% Regularize within-class scatter matrix
lambda = 1e-6;
SwRegularized = Sw + lambda * eye(numFeatures);

% Generalized eigenvalue decomposition
[eigVectors, eigValues] = eig(Sb, SwRegularized);

[eigValues, idx] = sort(real(diag(eigValues)), 'descend');
eigVectors = real(eigVectors(:, idx));

% Three classes allow maximum two FDA directions
numFDA = min(numClasses - 1, numFeatures);

fdaDirections = eigVectors(:, 1:numFDA);

% Project training and testing data
XTrainFDA = XTrain * fdaDirections;
XTestFDA = XTest * fdaDirections;

disp('FDA RESULTS');
fprintf('Original features: %d\n', numFeatures);
fprintf('Discriminant directions: %d\n', numFDA);

% Visualize FDA projection
figure;

gscatter(XTrainFDA(:,1), XTrainFDA(:,2), YTrain);

xlabel('FDA Direction 1');
ylabel('FDA Direction 2');
title('Wine Quality Classes - FDA Projection');
grid on;

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'FDA_Projection.png'));


%% KNN CLASSIFICATION COMPARISON

k = 5;

% Train KNN using original features
knnOriginal = fitcknn(XTrain, YTrain, 'NumNeighbors', k);
predOriginal = predict(knnOriginal, XTest);

% Train KNN using PCA features
knnPCA = fitcknn(XTrainPCA, YTrain, 'NumNeighbors', k);
predPCA = predict(knnPCA, XTestPCA);

% Train KNN using FDA features
knnFDA = fitcknn(XTrainFDA, YTrain, 'NumNeighbors', k);
predFDA = predict(knnFDA, XTestFDA);

% Calculate classification accuracy
accOriginal = mean(predOriginal == YTest) * 100;
accPCA = mean(predPCA == YTest) * 100;
accFDA = mean(predFDA == YTest) * 100;

% Display results
disp('KNN CLASSIFICATION RESULTS');

fprintf('Original Features Accuracy: %.2f%%\n', accOriginal);
fprintf('PCA Features Accuracy: %.2f%%\n', accPCA);
fprintf('FDA Features Accuracy: %.2f%%\n', accFDA);

% Plot comparison
figure;

accuracyValues = [accOriginal, accPCA, accFDA];

bar(accuracyValues);

set(gca, 'XTickLabel', ...
    {'Original', 'PCA', 'FDA'});

ylabel('Accuracy (%)');
title('KNN Classification Comparison');
ylim([0 100]);
grid on;

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'KNN_Comparison.png'));

% Confusion matrices
figure;
confusionchart(YTest, predOriginal);
title('KNN - Original Features');

figure;
confusionchart(YTest, predPCA);
title('KNN - PCA Features');

figure;
confusionchart(YTest, predFDA);
title('KNN - FDA Features');


%% CLASSIFICATION PERFORMANCE METRICS

predictions = {predOriginal, predPCA, predFDA};
methodNames = {'Original', 'PCA', 'FDA'};

classOrder = categories(YTest);
numClasses = numel(classOrder);

for j = 1:3

    C = confusionmat(YTest, predictions{j}, ...
        'Order', categorical(classOrder));

    precision = zeros(numClasses,1);
    recall = zeros(numClasses,1);
    f1 = zeros(numClasses,1);

    for i = 1:numClasses

        TP = C(i,i);
        FP = sum(C(:,i)) - TP;
        FN = sum(C(i,:)) - TP;

        if TP + FP > 0
            precision(i) = TP / (TP + FP);
        end

        if TP + FN > 0
            recall(i) = TP / (TP + FN);
        end

        if precision(i) + recall(i) > 0
            f1(i) = 2 * precision(i) * recall(i) / ...
                (precision(i) + recall(i));
        end

    end

    macroF1 = mean(f1) * 100;
    balancedAccuracy = mean(recall) * 100;

    fprintf('\n%s CLASSIFICATION METRICS\n', methodNames{j});

    for i = 1:numClasses
        fprintf('%s: Precision %.2f%%, Recall %.2f%%, F1 %.2f%%\n', ...
            classOrder{i}, precision(i)*100, ...
            recall(i)*100, f1(i)*100);
    end

    fprintf('Macro-F1: %.2f%%\n', macroF1);
    fprintf('Balanced Accuracy: %.2f%%\n', balancedAccuracy);

end


%% RANDOM OVERSAMPLING FOR CLASS BALANCING

classes = categories(YTrain);
classCounts = countcats(YTrain);

targetCount = max(classCounts);

balancedIdx = [];

for i = 1:numel(classes)

    classIdx = find(YTrain == classes{i});

    sampledIdx = classIdx(randi( ...
        numel(classIdx), targetCount, 1));

    balancedIdx = [balancedIdx; sampledIdx];

end

% Shuffle balanced training samples
balancedIdx = balancedIdx(randperm(numel(balancedIdx)));

% Create balanced datasets
XTrainBalanced = XTrain(balancedIdx,:);
XTrainPCABalanced = XTrainPCA(balancedIdx,:);
XTrainFDABalanced = XTrainFDA(balancedIdx,:);

YTrainBalanced = YTrain(balancedIdx);

disp('BALANCED TRAINING DATA');

tabulate(YTrainBalanced);

% Train balanced KNN classifiers
knnOriginalBal = fitcknn(XTrainBalanced, ...
    YTrainBalanced, 'NumNeighbors', 5);

knnPCABal = fitcknn(XTrainPCABalanced, ...
    YTrainBalanced, 'NumNeighbors', 5);

knnFDABal = fitcknn(XTrainFDABalanced, ...
    YTrainBalanced, 'NumNeighbors', 5);

% Predict on original untouched test data
predOriginalBal = predict(knnOriginalBal, XTest);
predPCABal = predict(knnPCABal, XTestPCA);
predFDABal = predict(knnFDABal, XTestFDA);

% Calculate accuracy
fprintf('\nBALANCED KNN RESULTS\n');

fprintf('Original: %.2f%%\n', ...
    mean(predOriginalBal == YTest)*100);

fprintf('PCA: %.2f%%\n', ...
    mean(predPCABal == YTest)*100);

fprintf('FDA: %.2f%%\n', ...
    mean(predFDABal == YTest)*100);


%% BALANCED CLASSIFIER EVALUATION

balancedPredictions = {predOriginalBal, predPCABal, predFDABal};
methodNames = {'Original Balanced', 'PCA Balanced', 'FDA Balanced'};

classOrder = categories(YTest);

for j = 1:3

    C = confusionmat(YTest, balancedPredictions{j}, ...
        'Order', categorical(classOrder));

    precision = diag(C) ./ max(sum(C,1)', 1);
    recall = diag(C) ./ max(sum(C,2), 1);

    f1 = 2 .* precision .* recall ./ ...
        max(precision + recall, eps);

    macroF1 = mean(f1) * 100;
    balancedAccuracy = mean(recall) * 100;

    fprintf('\n%s\n', methodNames{j});

    for i = 1:numel(classOrder)
        fprintf('%s: Precision %.2f%%, Recall %.2f%%, F1 %.2f%%\n', ...
            classOrder{i}, precision(i)*100, ...
            recall(i)*100, f1(i)*100);
    end

    fprintf('Macro-F1: %.2f%%\n', macroF1);
    fprintf('Balanced Accuracy: %.2f%%\n', balancedAccuracy);

end


%% GAUSSIAN BAYESIAN CLASSIFICATION

% Original features
bayesOriginal = fitcnb(XTrain, YTrain, ...
    'DistributionNames', 'normal');

predBayesOriginal = predict(bayesOriginal, XTest);

% PCA features
bayesPCA = fitcnb(XTrainPCA, YTrain, ...
    'DistributionNames', 'normal');

predBayesPCA = predict(bayesPCA, XTestPCA);

% FDA features
bayesFDA = fitcnb(XTrainFDA, YTrain, ...
    'DistributionNames', 'normal');

predBayesFDA = predict(bayesFDA, XTestFDA);

% Calculate accuracy
bayesAcc = [
    mean(predBayesOriginal == YTest)*100;
    mean(predBayesPCA == YTest)*100;
    mean(predBayesFDA == YTest)*100
    ];

disp('BAYESIAN CLASSIFICATION RESULTS');

fprintf('Original Features: %.2f%%\n', bayesAcc(1));
fprintf('PCA Features: %.2f%%\n', bayesAcc(2));
fprintf('FDA Features: %.2f%%\n', bayesAcc(3));

% Comparison graph
figure;

bar(bayesAcc);

set(gca, 'XTickLabel', {'Original', 'PCA', 'FDA'});

ylabel('Accuracy (%)');
title('Bayesian Classification Comparison');
ylim([0 100]);
grid on;

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'Bayesian_Comparison.png'));


%% BAYESIAN CLASSIFICATION EVALUATION

bayesPredictions = {predBayesOriginal, ...
    predBayesPCA, predBayesFDA};

bayesMethods = {'Bayes Original', 'Bayes PCA', 'Bayes FDA'};

classOrder = categories(YTest);

for j = 1:3

    C = confusionmat(YTest, bayesPredictions{j}, ...
        'Order', categorical(classOrder));

    precision = diag(C) ./ max(sum(C,1)', 1);
    recall = diag(C) ./ max(sum(C,2), 1);

    f1 = 2 .* precision .* recall ./ ...
        max(precision + recall, eps);

    fprintf('\n%s METRICS\n', bayesMethods{j});

    for i = 1:numel(classOrder)
        fprintf('%s: Precision %.2f%%, Recall %.2f%%, F1 %.2f%%\n', ...
            classOrder{i}, precision(i)*100, ...
            recall(i)*100, f1(i)*100);
    end

    fprintf('Macro-F1: %.2f%%\n', mean(f1)*100);
    fprintf('Balanced Accuracy: %.2f%%\n', mean(recall)*100);

end


%% MAHALANOBIS DISTANCE CLASSIFIER

classes = categories(YTrain);
numClasses = numel(classes);

classMeans = zeros(numClasses, size(XTrain,2));
classCovariances = cell(numClasses,1);

for i = 1:numClasses

    classData = XTrain(YTrain == classes{i},:);

    classMeans(i,:) = mean(classData);

    classCovariances{i} = cov(classData) + ...
        1e-5 * eye(size(XTrain,2));

end

% Predict using Mahalanobis distance
mahalPred = strings(size(XTest,1),1);

for i = 1:size(XTest,1)

    distances = zeros(numClasses,1);

    for j = 1:numClasses

        difference = XTest(i,:) - classMeans(j,:);

        distances(j) = difference / classCovariances{j} ...
            * difference';

    end

    [~, bestClass] = min(distances);

    mahalPred(i) = classes{bestClass};

end

predMahalanobis = categorical(mahalPred, classes);

% Accuracy
mahalAccuracy = mean(predMahalanobis == YTest)*100;

fprintf('\nMAHALANOBIS CLASSIFICATION\n');
fprintf('Accuracy: %.2f%%\n', mahalAccuracy);

% Confusion matrix
figure;

confusionchart(YTest, predMahalanobis);

title('Mahalanobis Distance Classification');

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'Mahalanobis_Confusion.png'));

%% MAHALANOBIS PERFORMANCE EVALUATION

C = confusionmat(YTest, predMahalanobis, ...
    'Order', categorical(categories(YTest)));

precision = diag(C) ./ max(sum(C,1)', 1);
recall = diag(C) ./ max(sum(C,2), 1);

f1 = 2 .* precision .* recall ./ ...
    max(precision + recall, eps);

disp('MAHALANOBIS CLASSIFICATION METRICS');

for i = 1:numel(classes)
    fprintf('%s: Precision %.2f%%, Recall %.2f%%, F1 %.2f%%\n', ...
        classes{i}, precision(i)*100, ...
        recall(i)*100, f1(i)*100);
end

fprintf('Macro-F1: %.2f%%\n', mean(f1)*100);
fprintf('Balanced Accuracy: %.2f%%\n', mean(recall)*100);


%% FINAL CLASSIFIER COMPARISON

%% FINAL CLASSIFIER COMPARISON

Classifier = {'KNN'; 'Gaussian Naive Bayes'; 'Mahalanobis'};

allClassifierPredictions = {
    predOriginal;
    predBayesOriginal;
    predMahalanobis
    };

Accuracy = zeros(3,1);
MacroF1 = zeros(3,1);
BalancedAccuracy = zeros(3,1);

classOrder = categories(YTest);

for j = 1:3

    C = confusionmat(YTest, allClassifierPredictions{j}, ...
        'Order', categorical(classOrder));

    precision = diag(C) ./ max(sum(C,1)',1);
    recall = diag(C) ./ max(sum(C,2),1);

    f1 = 2 .* precision .* recall ./ ...
        max(precision + recall, eps);

    Accuracy(j) = mean(allClassifierPredictions{j} == YTest) * 100;
    MacroF1(j) = mean(f1) * 100;
    BalancedAccuracy(j) = mean(recall) * 100;

end

FinalResults = table(Classifier, Accuracy, ...
    MacroF1, BalancedAccuracy);

disp('FINAL CLASSIFIER COMPARISON');
disp(FinalResults);

writetable(FinalResults, ...
    fullfile(projectRoot, 'Results', ...
    'Final_Classifier_Comparison.csv'));


%% CLASSIFIER PERFORMANCE COMPARISON

figure;

comparisonData = [
    78.29 51.72 49.13;
    68.61 48.89 52.21;
    46.24 40.65 55.01
    ];

bar(comparisonData);

set(gca, 'XTickLabel', ...
    {'KNN', 'Bayesian', 'Mahalanobis'});

ylabel('Performance (%)');

title('Comparison of Classical Pattern Recognition Classifiers');

legend('Accuracy', 'Macro-F1', ...
    'Balanced Accuracy', 'Location', 'best');

ylim([0 100]);
grid on;

saveas(gcf, fullfile(projectRoot, 'Results', ...
    'Classifier_Performance_Comparison.png'));


%% COMPLETE EXPERIMENTAL SUMMARY

Method = {
    'KNN Original';
    'KNN PCA';
    'KNN FDA';
    'KNN Original Balanced';
    'KNN PCA Balanced';
    'KNN FDA Balanced';
    'Bayesian Original';
    'Bayesian PCA';
    'Bayesian FDA';
    'Mahalanobis Original'
    };

Accuracy = [
    78.29; 78.48; 75.47;
    64.29; 63.16; 57.14;
    68.61; 75.56; 75.94;
    46.24
    ];

MacroF1 = [
    51.72; 51.06; 43.18;
    49.49; 48.24; 41.90;
    48.89; 48.54; 45.38;
    40.65
    ];

BalancedAccuracy = [
    49.13; 48.42; 42.57;
    57.51; 56.10; 48.40;
    52.21; 46.82; 43.66;
    55.01
    ];

ExperimentalResults = table(Method, Accuracy, ...
    MacroF1, BalancedAccuracy);

disp('COMPLETE EXPERIMENTAL RESULTS');
disp(ExperimentalResults);

writetable(ExperimentalResults, ...
    fullfile(projectRoot, 'Results', ...
    'Complete_Experimental_Results.csv'));


%% 5-FOLD STRATIFIED CROSS-VALIDATION

rng(42);

% Original features and class labels
X = wineData{:,1:11};
Y = wineData.Class;

% Create stratified 5-fold partition
cv5 = cvpartition(Y, 'KFold', 5);

% Store performance for each fold
accuracyCV = zeros(5,3);
macroF1CV = zeros(5,3);
balancedAccCV = zeros(5,3);

methodNames = {'Original KNN','PCA KNN','FDA KNN'};
classNames = categories(Y);

for fold = 1:5

    fprintf('\nPROCESSING FOLD %d\n', fold);

    trainIdx = training(cv5,fold);
    testIdx = test(cv5,fold);

    XtrRaw = X(trainIdx,:);
    XteRaw = X(testIdx,:);

    Ytr = Y(trainIdx);
    Yte = Y(testIdx);

    % Standardization using training data only
    mu = mean(XtrRaw);
    sigma = std(XtrRaw);
    sigma(sigma == 0) = 1;

    Xtr = (XtrRaw-mu)./sigma;
    Xte = (XteRaw-mu)./sigma;

    %% PCA

    covarianceMatrix = cov(Xtr);

    [V,D] = eig(covarianceMatrix);

    [eigenValues,idx] = sort(diag(D),'descend');
    V = V(:,idx);

    explained = 100*eigenValues/sum(eigenValues);
    cumulative = cumsum(explained);

    numPC = find(cumulative >= 95,1);

    pcaDirections = V(:,1:numPC);

    XtrPCA = Xtr*pcaDirections;
    XtePCA = Xte*pcaDirections;

    %% FDA

    numFeatures = size(Xtr,2);
    numClasses = numel(classNames);

    overallMean = mean(Xtr);

    Sw = zeros(numFeatures);
    Sb = zeros(numFeatures);

    for c = 1:numClasses

        classData = Xtr(Ytr == classNames{c},:);

        classMean = mean(classData);
        classCount = size(classData,1);

        centered = classData-classMean;

        Sw = Sw + centered'*centered;

        difference = classMean-overallMean;

        Sb = Sb + classCount*(difference'*difference);

    end

    Sw = Sw + 1e-6*eye(numFeatures);

    [Vf,Df] = eig(Sb,Sw);

    [~,idx] = sort(real(diag(Df)),'descend');

    Vf = real(Vf(:,idx));

    fdaDirections = Vf(:,1:numClasses-1);

    XtrFDA = Xtr*fdaDirections;
    XteFDA = Xte*fdaDirections;

    %% KNN CLASSIFICATION

    predictions = cell(3,1);

    model1 = fitcknn(Xtr,Ytr,'NumNeighbors',5);
    predictions{1} = predict(model1,Xte);

    model2 = fitcknn(XtrPCA,Ytr,'NumNeighbors',5);
    predictions{2} = predict(model2,XtePCA);

    model3 = fitcknn(XtrFDA,Ytr,'NumNeighbors',5);
    predictions{3} = predict(model3,XteFDA);

    %% PERFORMANCE EVALUATION

    for m = 1:3

        C = confusionmat(Yte,predictions{m}, ...
            'Order',categorical(classNames));

        precision = diag(C)./max(sum(C,1)',1);
        recall = diag(C)./max(sum(C,2),1);

        f1 = 2*precision.*recall ./ ...
            max(precision+recall,eps);

        accuracyCV(fold,m) = ...
            mean(predictions{m} == Yte)*100;

        macroF1CV(fold,m) = mean(f1)*100;

        balancedAccCV(fold,m) = mean(recall)*100;

    end

end

%% CROSS-VALIDATION SUMMARY

disp('5-FOLD CROSS-VALIDATION RESULTS');

for m = 1:3

    fprintf('\n%s\n',methodNames{m});

    fprintf('Accuracy: %.2f +/- %.2f%%\n', ...
        mean(accuracyCV(:,m)),std(accuracyCV(:,m)));

    fprintf('Macro-F1: %.2f +/- %.2f%%\n', ...
        mean(macroF1CV(:,m)),std(macroF1CV(:,m)));

    fprintf('Balanced Accuracy: %.2f +/- %.2f%%\n', ...
        mean(balancedAccCV(:,m)),std(balancedAccCV(:,m)));

end

% Save cross-validation results
CVResults = table(methodNames', ...
    mean(accuracyCV)',std(accuracyCV)', ...
    mean(macroF1CV)',std(macroF1CV)', ...
    mean(balancedAccCV)',std(balancedAccCV)', ...
    'VariableNames',{'Method','MeanAccuracy','StdAccuracy', ...
    'MeanMacroF1','StdMacroF1', ...
    'MeanBalancedAccuracy','StdBalancedAccuracy'});

disp(CVResults);

writetable(CVResults,fullfile(projectRoot, ...
    'Results','Five_Fold_CV_Results.csv'));


%% 5-FOLD CV: BAYESIAN AND MAHALANOBIS

rng(42);

cv5 = cvpartition(Y,'KFold',5);

accuracyExtra = zeros(5,4);
macroF1Extra = zeros(5,4);
balancedExtra = zeros(5,4);

extraNames = {'Bayes Original','Bayes PCA', ...
    'Bayes FDA','Mahalanobis'};

classNames = categories(Y);

for fold = 1:5

    fprintf('\nEXTRA CV FOLD %d\n',fold);

    trainIdx = training(cv5,fold);
    testIdx = test(cv5,fold);

    XtrRaw = X(trainIdx,:);
    XteRaw = X(testIdx,:);

    Ytr = Y(trainIdx);
    Yte = Y(testIdx);

    % Standardization
    mu = mean(XtrRaw);
    sigma = std(XtrRaw);
    sigma(sigma == 0) = 1;

    Xtr = (XtrRaw-mu)./sigma;
    Xte = (XteRaw-mu)./sigma;

    %% PCA

    [V,D] = eig(cov(Xtr));
    [e,idx] = sort(diag(D),'descend');
    V = V(:,idx);

    explained = 100*e/sum(e);
    numPC = find(cumsum(explained)>=95,1);

    directionsPCA = V(:,1:numPC);

    XtrPCA = Xtr*directionsPCA;
    XtePCA = Xte*directionsPCA;

    %% FDA

    nc = numel(classNames);
    nf = size(Xtr,2);

    overallMean = mean(Xtr);
    Sw = zeros(nf);
    Sb = zeros(nf);

    for c = 1:nc

        classData = Xtr(Ytr==classNames{c},:);
        classMean = mean(classData);
        n = size(classData,1);

        centered = classData-classMean;
        Sw = Sw+centered'*centered;

        difference = classMean-overallMean;
        Sb = Sb+n*(difference'*difference);

    end

    Sw = Sw+1e-6*eye(nf);

    [Vf,Df] = eig(Sb,Sw);
    [~,idx] = sort(real(diag(Df)),'descend');

    Vf = real(Vf(:,idx));

    directionsFDA = Vf(:,1:nc-1);

    XtrFDA = Xtr*directionsFDA;
    XteFDA = Xte*directionsFDA;

    %% BAYESIAN CLASSIFICATION

    bayes1 = fitcnb(Xtr,Ytr,'DistributionNames','normal');
    p1 = predict(bayes1,Xte);

    bayes2 = fitcnb(XtrPCA,Ytr,'DistributionNames','normal');
    p2 = predict(bayes2,XtePCA);

    bayes3 = fitcnb(XtrFDA,Ytr,'DistributionNames','normal');
    p3 = predict(bayes3,XteFDA);

    %% MAHALANOBIS CLASSIFICATION

    means = zeros(nc,nf);
    covariances = cell(nc,1);

    for c = 1:nc

        classData = Xtr(Ytr==classNames{c},:);

        means(c,:) = mean(classData);

        covariances{c} = cov(classData)+ ...
            1e-5*eye(nf);

    end

    p4 = strings(size(Xte,1),1);

    for i = 1:size(Xte,1)

        distances = zeros(nc,1);

        for c = 1:nc

            difference = Xte(i,:)-means(c,:);

            distances(c) = difference / ...
                covariances{c} * difference';

        end

        [~,best] = min(distances);
        p4(i) = classNames{best};

    end

    p4 = categorical(p4,classNames);

    predictions = {p1,p2,p3,p4};

    %% EVALUATION

    for m = 1:4

        C = confusionmat(Yte,predictions{m}, ...
            'Order',categorical(classNames));

        precision = diag(C)./max(sum(C,1)',1);
        recall = diag(C)./max(sum(C,2),1);

        f1 = 2*precision.*recall ./ ...
            max(precision+recall,eps);

        accuracyExtra(fold,m) = ...
            mean(predictions{m}==Yte)*100;

        macroF1Extra(fold,m) = mean(f1)*100;

        balancedExtra(fold,m) = mean(recall)*100;

    end

end

%% EXTRA CROSS-VALIDATION SUMMARY

disp('BAYESIAN AND MAHALANOBIS 5-FOLD RESULTS');

for m = 1:4

    fprintf('\n%s\n',extraNames{m});

    fprintf('Accuracy: %.2f +/- %.2f%%\n', ...
        mean(accuracyExtra(:,m)),std(accuracyExtra(:,m)));

    fprintf('Macro-F1: %.2f +/- %.2f%%\n', ...
        mean(macroF1Extra(:,m)),std(macroF1Extra(:,m)));

    fprintf('Balanced Accuracy: %.2f +/- %.2f%%\n', ...
        mean(balancedExtra(:,m)),std(balancedExtra(:,m)));

end

%% SAVE EXTRA CV RESULTS

ExtraCVResults = table(extraNames', ...
    mean(accuracyExtra)',std(accuracyExtra)', ...
    mean(macroF1Extra)',std(macroF1Extra)', ...
    mean(balancedExtra)',std(balancedExtra)', ...
    'VariableNames',{'Method','MeanAccuracy','StdAccuracy', ...
    'MeanMacroF1','StdMacroF1', ...
    'MeanBalancedAccuracy','StdBalancedAccuracy'});

writetable(ExtraCVResults,fullfile(projectRoot, ...
    'Results','Bayesian_Mahalanobis_CV_Results.csv'));

disp(ExtraCVResults);


%% FINAL 5-FOLD OUT-OF-FOLD CONFUSION MATRICES

rng(42);

cvFinal = cvpartition(Y,'KFold',5);

classNames = categories(Y);

% Store predictions for every sample
oofOriginal = Y;
oofPCA = Y;
oofFDA = Y;

for fold = 1:5

    trainIdx = training(cvFinal,fold);
    testIdx = test(cvFinal,fold);

    XtrRaw = X(trainIdx,:);
    XteRaw = X(testIdx,:);

    Ytr = Y(trainIdx);

    % Standardization using training data
    mu = mean(XtrRaw);
    sigma = std(XtrRaw);
    sigma(sigma == 0) = 1;

    Xtr = (XtrRaw-mu)./sigma;
    Xte = (XteRaw-mu)./sigma;

    %% PCA

    [V,D] = eig(cov(Xtr));
    [e,idx] = sort(diag(D),'descend');
    V = V(:,idx);

    explained = 100*e/sum(e);
    numPC = find(cumsum(explained)>=95,1);

    directionsPCA = V(:,1:numPC);

    XtrPCA = Xtr*directionsPCA;
    XtePCA = Xte*directionsPCA;

    %% FDA

    nf = size(Xtr,2);
    nc = numel(classNames);

    overallMean = mean(Xtr);
    Sw = zeros(nf);
    Sb = zeros(nf);

    for c = 1:nc

        classData = Xtr(Ytr==classNames{c},:);
        classMean = mean(classData);
        n = size(classData,1);

        centered = classData-classMean;
        Sw = Sw+centered'*centered;

        difference = classMean-overallMean;
        Sb = Sb+n*(difference'*difference);

    end

    Sw = Sw+1e-6*eye(nf);

    [Vf,Df] = eig(Sb,Sw);
    [~,idx] = sort(real(diag(Df)),'descend');

    Vf = real(Vf(:,idx));
    directionsFDA = Vf(:,1:nc-1);

    XtrFDA = Xtr*directionsFDA;
    XteFDA = Xte*directionsFDA;

    %% KNN PREDICTIONS

    model1 = fitcknn(Xtr,Ytr,'NumNeighbors',5);
    model2 = fitcknn(XtrPCA,Ytr,'NumNeighbors',5);
    model3 = fitcknn(XtrFDA,Ytr,'NumNeighbors',5);

    oofOriginal(testIdx) = predict(model1,Xte);
    oofPCA(testIdx) = predict(model2,XtePCA);
    oofFDA(testIdx) = predict(model3,XteFDA);

end

%% CONFUSION MATRICES AND CLASS-WISE METRICS

allPredictions = {oofOriginal,oofPCA,oofFDA};

methodNames = {'Original KNN','PCA KNN','FDA KNN'};

for m = 1:3

    C = confusionmat(Y,allPredictions{m}, ...
        'Order',categorical(classNames));

    disp(['CONFUSION MATRIX: ',methodNames{m}]);
    disp(C);

    precision = diag(C)./max(sum(C,1)',1);
    recall = diag(C)./max(sum(C,2),1);

    fprintf('\n%s PER-CLASS METRICS\n',methodNames{m});

    for c = 1:numel(classNames)

        fprintf('%s: Precision %.2f%%, Recall %.2f%%\n', ...
            classNames{c},precision(c)*100,recall(c)*100);

    end

    figure;
    confusionchart(Y,allPredictions{m});

    title(['Five-Fold ',methodNames{m}]);

    fileName = ['CV_Confusion_', ...
        strrep(methodNames{m},' ','_'),'.png'];

    saveas(gcf,fullfile(projectRoot,'Results',fileName));

end

