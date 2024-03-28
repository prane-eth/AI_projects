# Credit Default Risk Prediction: Final Report


## Introduction
In the realm of banking and finance, the ability to predict loan defaults accurately is a critical component for risk management and operational efficiency. This project, focusing on a dataset from a German bank comprising historical loan information of 1000 customers, seeks to address this need. By leveraging machine learning algorithms, the organizations aim to unveil hidden patterns and factors that contribute to loan defaults.

This exploratory journey revolves around key questions: What characteristics distinguish defaulters from non-defaulters? How can machine learning models best predict future defaults based on the existing data? As we delve deeper, we also consider the impact of various customer attributes such as employment duration, existing loans, savings balance, and demographic details on their likelihood of defaulting.

This analysis is not just about model accuracy but also about understanding the underlying dynamics of financial behavior. Loan defaults pose significant challenges in banking, affecting millions of dollars in assets annually. This project aims to mitigate these risks through predictive analytics. In a global financial environment marked by volatility, accurately predicting loan defaults is more crucial than ever for financial stability.


## Methods and Materials
This methodology encompasses two primary segments: Exploratory Data Analysis (EDA) and Model Development. In the EDA phase, I conducted a thorough examination of the dataset, visualizing various aspects such as loan amounts, duration, and balance. This process involved cleaning the data, handling missing values, and creating new features to enhance the predictive power of the models. 

For model development, I employed a diverse set of machine learning algorithms, including Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machines. Each model was carefully tuned and evaluated based on accuracy, F1 score, and ROC AUC score. The selection of machine learning algorithms was based on their proven efficacy in classification tasks and their ability to handle diverse data types.

The analysis was performed using Python, specifically utilizing libraries such as Pandas, Scikit-Learn, and Matplotlib for data processing, model development, and visualization. Data cleaning involved standardizing categorical variables and handling missing values, while feature engineering focused on creating new features like creating a new `age_group` column using the `age` column.


**Exploratory Data Analysis (EDA)**:

Note: Observations are added in the Python notebook.

- **Savings Balance vs Default Status**

  - **Description**: Investigates how savings account balances correlate with defaulting on loans.

  - **Visualization**: Bar chart showing the count of defaulters and non-defaulters across different savings balance ranges.

    ![Savings Balance vs Default Status](images/Savings_Balance_vs_Default_Status.png)

- **Loan Duration vs Default Status**

  - **Description**: Examines the duration of loans in relation to the likelihood of default.

  - **Visualization**: Box plot comparing the loan duration of defaulters vs. non-defaulters.

    ![Loan Duration vs Default Status](images/Loan_Duration_vs_Default_Status.png)

- **Credit History vs Default Status**

  - **Description**: The association between the credit history categories of individuals and their default status on loans.

  - **Visualization**: Bar chart depicting the number of defaults across different credit history categories.

    ![Credit History vs Default Status](images/Credit_History_vs_Default_Status.png)

- **Correlation Heatmap**:

  - **Description**: Visual representation of how various financial variables correlate with the likelihood of loan default.

  - **Visualization**: Heatmap displaying correlation coefficients.
  
    ![Correlation Heatmap](images/Correlation_Matrix.png)



## Results
The analysis reveals that the CatBoost classifier excels among the various models, achieving a test accuracy of 80.00%, with a notable overfitting value of 0.10750. This model also shows a cross-validation accuracy of 75.7%, emphasizing its consistency and robustness in performance. Its F1-Score of 59.18% and ROC AUC Score of 83.25% further illustrate its strong capability in distinguishing between default and non-default cases. 

While the CatBoost model outperformed others in terms of accuracy and overfitting, models like Gradient Boosting and Random Forest excelled in aspects such as F1 score and ROC AUC, indicating their utility in certain scenarios.

The analysis consistently identified key attributes such as employment duration, loan amount, and age as significant predictors of default risk. This project illustrates the nuanced and multifaceted nature of predicting loan defaults, where no single model uniformly excels across all metrics.

**Model Performance results**:
    ![Model Performances](images/Model_Performances.png)

**Visualizations**:

- **Feature Importances**:
    ![Feature Importances](images/Feature_Importances.png)

- **Confusion Matrix**:
    ![Confusion Matrix](images/Confusion_Matrix.png)

- **ROC Curve**:
    ![ROC Curve](images/ROC_Curve.png)

- **Precision-Recall Curve**:
    ![Precision-Recall Curve](images/Precision_Recall_Curve.png)


## Discussion
The balance between accuracy and overfitting makes the CatBoost model a superior choice in this context. While ensemble methods like Gradient Boosting and Random Forest demonstrated commendable performance, the CatBoost model's balance between accuracy and overfitting made it the superior choice. Even simpler models, such as Logistic Regression, provided valuable insights despite lower accuracy, underscoring the complexity of credit risk assessment.

The most striking revelation from the study was the complex interplay of various factors in determining loan default risk. For instance, longer employment duration was generally associated with lower default rates, possibly reflecting greater financial stability. Meanwhile, higher loan amounts correlated with increased default risk, underscoring the challenges faced by customers in managing larger debts.

However, the study is not without limitations. The reliance on historical data may not fully capture the evolving economic conditions and customer behaviors. Future research could incorporate more dynamic models that adapt to changing market trends. Additionally, integrating alternative data sources, such as social media or transactional data, could further refine the predictions. These findings can assist financial institutions in refining their risk assessment algorithms and loan approval criteria, potentially reducing the incidence of loan defaults.


## Conclusions
In conclusion, this study offers valuable insights into the factors influencing loan defaults and demonstrates the potential of machine learning in financial risk assessment. While no single model proved infallible, the combination of different approaches provides a robust framework for predicting defaults. This project not only advances the understanding of credit risk but also lays the groundwork for more innovative and adaptive models in the future. Moving forward, incorporating real-time data and machine learning models into banking systems could revolutionize how financial institutions manage credit risk.