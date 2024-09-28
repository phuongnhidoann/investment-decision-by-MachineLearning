# MACHINE LEARNING-BASED DECISION SUPPORT FOR FINANCIAL INVESTMENT IN NON-FINANCIAL VIETNAMESE COMPANIES LISTED ON THE STOCK EXCHANGE                                 

**Abstract:** To support the decision-making process of new investors, this paper aims to implement Machine Learning algorithms to generate investment indications, considering the in the case of Vietnamese non-financial companies. The dataset used was provided by WiGroup and Refinancing Rate in Vietnam was collected on the FocusEconomics. The results of the different algorithms were compared to each other using the following metrics: accuracy, precision, recall, and F1-score. The Random Forest was the algorithm that obtained the best classification metrics and an accuracy of 72%.

**Keywords:** financial investment; machine learning.

# 1. INTRODUCTION
- The financial investment business has been growing in recent years with new people interested in the subject, mainly by individuals. According to Vietnam Securities Depository (VSD), the number of domestic investor accounts increased by an additional 39,430 accounts in December 2023. In terms of structure, the increase in the number of securities accounts in the last month of the year was mainly driven by individual investors with 39,240 accounts. Meanwhile, the number of accounts held by institutional investors only increased by 190 accounts. Within this context, it can be noticed that new investors need some help to start their activities in the investment business. 
- This project will use Machine Learning algorithms to assist in the decision making process of new investors in the financial investment field, where it was sought to answer the following research questions: how to evaluate the financial situations of companies listed on the Vietnamese Stock Market to assist in the decision-making process of new investors and which Machine Learning algorithms have greater accuracy for this purpose?
- Machine Learning algorithms that serve as financial investment advisors for new investors may be a solution to these questions. Based on these needs, this paper has the following objectives: identify the financial data to make reliable investment recommendations; perform preprocessing; define algorithms that will be used; implement Machine Learning algorithms; conduct validation and performance tests with the used model.
# 2. THEORETICAL BACKGROUND
## 2.1. Financial market investment types
- Financial investments can take various forms, including stocks, savings accounts, commodities, and foreign currencies, among others. Despite the diversity of these applications, a financial investment is fundamentally defined as the allocation of capital into a financial instrument with the primary goal of generating future income (According to CARDOZO, T.).
- According to RT. Paiva et al, financial investments can be categorized into two main types: fixed-income and variable-income investments. Fixed-income investments provide the investor with a predetermined return at the end of the investment period, offering a sense of certainty about the profitability. Common examples include savings accounts and government bonds. On the other hand, variable-income investments do not guarantee a fixed return due to their fluctuating nature, making them riskier but potentially more lucrative. Stocks and mutual funds are prime examples of variable-income investments, where returns are influenced by market conditions.
## 2.2. Fundamental analysis
- According to Gava Gastaldo.N (2019), fundamentalist analysis determines the appropriate stock prices using the earnings and dividends of a given company, expectations of future interest rates, and risk assessment of the company.
- The basic objective of the valuation of a company is to obtain a fair value, which reflects the expected return on future performance projections consistent with the reality of the company evaluated. Since it is based on projections, the valuation is subject to errors and uncertainties, mainly since the analysis of external variables is not controlled by the company in question. Within this context, the result obtained through an evaluation is not an exact estimate of the value of the evaluated company.
- To be able to analyze companies, it is important to use variables and indicators, known as fundamentalist indicators, that impact a company. In this project, the most used indicators were used according to Bered and Rosa, having as a basis the theoretical foundation presented to obtain each indicator. The fundamentalist indicators that were used are: Earnings per Share (EPS), Price/Earnings (P/E), Book Value per Share (BVPS), Return on Equity (ROE) and EBITDA.
## 2.3. Machine Learning models used in project
### 2.3.1. Logistic Regression
- Binary Logistic Regression models the relationship between a set of independent variables and a binary dependent variable. The goal is to find the best model that describes the relationship between the dependent variable and multiple independent variables.
- The Logistic Regression’s dependent variable could be binary or categorical and the independent ones could be a mixture of continuous, categorical, and binary variables.
- The general form of Logistic Regression is as follows:
**y = alpha + beta1X1 + beta2X2 + beta3X3 + … + betamXm***
**P = 1/(1+e^(-y))**

where X1, X2, ..., Xm is the feature vector and y is a linear combination function of the features. The parameters beta1, beta2, ..., betam are the regression coefficients to be estimated. The output is between 0 and 1, and, usually, if the output is above the threshold of 0.5 the model predicts class 1 (positive) and otherwise class 0 (negative).

### 2.3.2. K-Nearest Neighbors (KNN)
The K-Nearest Neighbors algorithm is used in data mining. K-NN is a method to classify objects based on query points and all the objects in the training data. An object is classified based on its K neighbors. K is a positive integer that is determined before performing algorithms. Euclidean distance is often used to calculate the distance between objects.

### 2.3.3. Decision Tree
The Decision Tree is a popular model for classification and regression tasks, characterized by its tree-like structure comprising nodes and branches. It offers several advantages that contribute to its widespread usage. Firstly, Decision Trees can handle a variety of features, including both categorical and numerical variables, making them versatile for different types of data. Additionally, the model's inherent structure provides a clear representation of acquired knowledge, making it easily interpretable for users and stakeholders. Lastly, Decision Trees are known for their efficiency, as they can efficiently conduct the entire training and learning process, making them suitable for large datasets and time-sensitive applications.

### 2.3.4. Support Vector Machine (SVM)
Support Vector Machine is one of the oldest ML algorithms and aims to identify the decision boundaries as the maximum-margin hyperplane separating two classes. The hyperplane equation is given by equation:

**f (x) =  omega ^ T * x + b**

where is omega the normal vector and b the bias. The objective function of SVM can be expressed as:

**{1/2‖omega‖^2  s.t. ={yi- omega^T.xi-b≤ε.omega^T.xi+b-y ≤ε**

Where ε is the deviation between f(x) and the target yi.

### 2.3.5. Random Forest
- Random Forest is an ensemble learning algorithm developed by Breiman [39]. Ensemble learning is a way to combine different basic classifiers (“weak” classifiers) to compose a new one (strong learner), more complex, more efficient, and more precise. The weak classifiers should make independent errors in their predictions, and thus a strong classifier can be composed of different algorithms or if the same algorithm is used the models should be trained with different subsets of the training set.
- Random Forest is an ensemble bagging tree-based learning algorithm. In particular, the Random Forest Classifier is a set of decision trees that are trained using randomly selected subsets of the training set and randomly selected subsets of features.

# 3. PROJECT
## 3.1. Dataset
I collected financial data from WiGroup from 2014 to 2023 of Vietnamese non-financial companies listed on the Vietnamese Stock Market and Refinancing Rate in Vietnam was collected on the FocusEconomics. Companies belonging to the financial sector were excluded from the dataset. The exclusion was made because the companies in this sector present very distinct characteristics if compared to the companies in other sectors, impairing comparability and consequently the training of algorithms.

**Table 1.** The 9 financial investment features I used in this project.

|  | Variable | Description | 
|--------------|-------|------|
| X1 | EPS (Earnings per Share) | Indicates how much money a company makes for each share of its stock and is a widely used metric for estimating corporate value. |
| X2 | P/E (Price/Earnings) | Evaluates the relationship between the current market price of a stock and its earnings per share. |
| X3 | BVPS (Book value per share) | Indicates a firm’s net asset value (total assets - total liabilities) on a per-share basis. |
| X4 | ROE (Return on Equity) | Measures the company's profit potential relative to its equity capital. |
| X5 | EBITDA | Measure of company profitability used by investors. |

To use the data, it is necessary to classify each stock as either designated or not designated as a good financial investment, where “1” represents a stock classified as a good investment and “0” represents a stock not considered a good investment. Based on the research of Oliveira (2022) and with adjustments in the calculation ratio, this classification is performed by comparing the change in EPS ratio of each stock with the change of Refinancing Rate in Vietnam.

## 3.2. Validation
To compare the performance of different models, evaluation metrics derived from the Confusion Matrix were used. The metrics included Accuracy, Precision, Recall, and F1-score, providing a detailed assessment of the classification results of each algorithm. These metrics allowed for a thorough comparison, helping to identify the algorithm that delivered the best results:

***Accuracy*** = number of correct predictions/number of predictions made

***Precision*** = TP/(TP+FP)

***Recall*** = TP/(TP+FN)

***F1-Score*** = (2 x Recall x Precision)/(Recall+ Precision) 

where TP is true positive, FN is false negative, and FP is false positive.

## 3.3. Visualization
<img src="https://i.imgur.com/ocJ6D61.png">

***Figure 1:*** Correlation Matrix

<img src="https://i.imgur.com/guWYERu.png">

***Figure 2:*** Distribution of Investment Chart

# 4. DEVELOPMENT
This section outlines the steps undertaken to complete this work, which include: database preparation, Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Support Vector Machine (SVM) and Random Forest. The initial step involved preparing the database for the subsequent implementation of these algorithms. The data preparation was performed in these steps, namely: calculates financial ratios, handles missing values, removes outliers, sets up the conditions to determine whether a stock is classified as a good investment or not. After all the data preparation, the splitting of the training and test sets was performed using Sklearn’s “train_test_split” function, where the data were split into 30% for training and 70% for testing.
## 4.1. Logistic Regression
<img src="https://i.imgur.com/vrxfeoP.png">

***Figure 3:*** Logistic Regression Classification Metrics

This is a mediocre overall accuracy score with 71%, indicating that the model has room for improvement. Class 1 has a significantly lower performance, with a recall rate of only 0.56, indicating that the model is missing many important data points. In addition, the model might be biased towards Class 0, prioritizing accurate classification of that class while neglecting Class 1.

<img src="https://i.imgur.com/BlMVVSq.png">

***Figure 4:*** Confusion Matrix of the Logistic Regression

## 4.2. KNN
<img src="https://i.imgur.com/rEI4ByY.png">

***Figure 5:*** Accuracy vs K-value for KNN

After testing the value of k in the K-Nearest Neighbors (KNN) model and evaluating the accuracy of the model with each value of k, we see the accuracy ranges from 65% to 71% suggests that the KNN model has moderate performance.

<img src="https://i.imgur.com/ZVkZqVM.png">

***Figure 6:*** KNN Classification Metrics

From the results, the model has an overall accuracy of 70%, which shows relatively stable performance. However, there is a significant difference in the classification performance between class “0” and class “1”. The model has better ability to classify class “0” with higher accuracy and better recall. On the other hand, the model struggles more in classifying class “1”.

<img src="https://i.imgur.com/xGXlxb5.png">

***Figure 7:*** Confusion Matrix of the KNN

## 4.3. Decision Tree
The Decision Tree was implemented using Sklearn’s DecisionTreeClassifier, which is a Decision Tree that is intended to work with classification problems. After performing the training and testing using the 2014 – 2023 dataset, which obtained the classification metrics shown in Figure 8.

<img src="https://i.imgur.com/GCTk0aX.png">

***Figure 8:*** Decision Tree Classification Metrics

The model achieves an overall accuracy of 65% on the entire test dataset, which is considered a poor performance. The classification performance of class “0” appears to be better than class “1”, while class “1” faces more difficulties and has lower performance.

<img src="https://i.imgur.com/rJuDIHE.png">

***Figure 9:*** Confusion Matrix of the Decision Tree

## 4.4. SVM

<img src="https://i.imgur.com/vT4uYoq.png">

***Figure 10:*** SVM Classification Metrics

Overall, the model demonstrates reasonably good prediction capabilities with an overall accuracy of 70%. However, there is a significant difference in performance between class “0” and class “1”. Although the model has relatively good accuracy and F1-score, the low recall (53%) indicates that the model tends to miss many samples belonging to class “1”. Additionally, the model performs very well with class “0”, with a recall of up to 83% and an F1-score of 76%. This suggests that the model learns the features of class “0” better than class “1”.

<img src="https://i.imgur.com/qKVsn5Z.png">

***Figure 11:*** Confusion Matrix of the SVM

## 4.5. Random Forest

<img src="https://i.imgur.com/qybSwhL.png">

***Figure 12:*** Random Forest Classification Metrics

The overall accuracy achieved by the model on the test dataset is 72%. This indicates that around 72% of the predictions made by the model are correct, which is a reasonably good performance. The performance in classifying both class “0” and class “1” has improved, with higher recall and F1-scores. 

<img src="https://i.imgur.com/2uWPArs.png">

***Figure 13:*** Confusion Matrix of the Random Forest

# 5.RESULT
This section will present the results obtained with the algorithms implemented in this work. First, a comparison will be presented regarding the accuracy obtained with each algorithm using the 2014 - 2023 dataset, as shown in Figure 14.

<img src="https://i.imgur.com/MH8TmJK.png">

***Figure 14:*** Comparison of algorithm Accuracy

- In the graph shown in Figure 14, the Random Forest has the highest accuracy, with 71.85%. This is followed by 70.68% accuracy for the Logistic Regression, 70.48% for the KNN, 70.16% for the SVM and 65.15% for the Decision Tree. Best overall performer of Random Forest due to balanced and high metrics in both classes, leading to the highest overall accuracy. Suitable for applications requiring balanced and reliable classification performance.
- However, to analyze the results in a deeper and more adequate way, it is necessary to analyze the classification metrics, as presented in Figure 15 and Figure 16. 

<img src="https://i.imgur.com/5k9gJP6.png">

***Figure 15:*** Comparison of Metrics for Class “1” Classification

<img src="https://i.imgur.com/yw1de6K.png">

***Figure 16:*** Comparison of Metrics for Class “0” Classification

- The Random Forest model performs consistently well for both class “0” and class “1”. Its high precision and recall for class “0” (74% and 79%, respectively) suggest it is very effective at correctly identifying negative instances while maintaining a good balance of avoiding false negatives. This is complemented by balanced metrics for class “1”, leading to its overall highest accuracy. In addition, the balanced performance across both classes implies that Random Forest manages both false positives and false negatives effectively, which is crucial for maintaining high accuracy and F1-Scores.
- SVM’s high precision for class “1” (69%) and class “0” (71%) suggests it is selective and accurate in classifying instances, avoiding false positives. However, the lower recall for class “1” (53%) indicates it misses many actual positive instances, even though it performs exceptionally well in identifying negative instances (83% recall for class “0”). The disparity between precision and recall in class “1” might contribute to its slightly lower overall accuracy, even though it performs well in class “0”.
- Both KNN and Logistic Regression models show consistent performance with high recall and precision for class “0”, and balanced but slightly lower metrics for class “1”. The slightly better recall in class “0” and balanced metrics in class “1” explain their overall comparable accuracy (~70%).
- Decision Tree shows the lowest performance in both classes, with precision and recall metrics at 70% for class “0” and lower metrics for class “1”.
- Investing in a bad stock results in losses for the investor, whereas not investing in a good stock means missing out on potential profits, but without incurring any losses. Consequently, when choosing the best algorithm for this application, it is crucial to focus on the metrics related to class “0” classifications (stocks deemed bad for investment). This ensures that the chosen algorithm minimizes the risk of investing in poor-performing stocks. Thus, Random Forest was really the algorithm that presented the best values for precision, recall and F1-score for the application of this work.
# 6. CONCLUSION
- This project implements Machine Learning algorithms to serve as financial investment advisors for investors. The objectives were successfully achieved, including: calculating essential financial ratios to develop reliable investment proposals; perform preprocessing; define algorithms that will be used. implement Machine Learning algorithms; perform validation and performance tests.
- Fundamental analysis is used as a basis to evaluate and create investment guidance, utilizing the fundamental financial indicators of each stock as variables in the dataset. However, the fundamental indicators alone are insufficient for implementation. Each registered stock needs to be labeled as either good (class “1”) or bad (class “0”) for investment purposes. Therefore, additional financial indicators based on the research of Bered and Rosa need to be included. In addition, preparation was carried out in the dataset, handling empty values and handling outliers.
- The Machine Learning techniques used were: Logistic Regression, K-Nearest Neighbors, Decision Tree, Support Vector Machine and Random Forest. The implementation was completed in Python language, using Machine Learning library Sikit-learn.
- The results obtained with each algorithm were compared using the metrics: accuracy, precision, recall, and F1-score. The algorithm that obtained the best classification metrics was Random Forest. The accuracy obtained with Random Forest was 72%. The metrics for class “0” were accuracy of 0.74, recall of 0.79, and F1-score of 0.76. Metrics for class “1” were accuracy of 0.68, recall of 0.62, and F1-score of 0.65.
- The accuracy of the Random Forest model, while notable, may be limited by the inherent complexity of market behavior, especially during the tumultuous years of 2020 and 2021, which were significantly impacted by the COVID-19 pandemic. One key issue is the inadequate variance ratio of EPS, which suggests a potential need to replace it with the variation of each share's quotation, as proposed in Oliveira’s (2022) research. This discrepancy likely contributes to inaccuracies in classifying stocks as good or bad investments. Furthermore, the stock prices used to calculate the P/E ratios are not entirely accurate, as they adhere to legal regulations mandating a minimum par value of 10,000 VND per share, leading to potential errors in the model’s predictions.
- Based on the findings, several recommendations and limitations were identified. The model’s accuracy can be influenced by external market conditions and events, such as the COVID-19 pandemic and financial metrics are not yet consistent. Ensuring the accuracy and relevance of stock price data used for calculating financial ratios is crucial. Using more precise and updated stock prices for each company would improve the reliability of the predictions. Additionally, experimenting with more advanced machine learning models or ensemble methods is suggested to improve classification performance and robustness.  Exploring time series forecasting techniques like ARIMA or LSTMs that can capture temporal patterns in stock prices. These models can potentially learn from historical price movements to make predictions.



