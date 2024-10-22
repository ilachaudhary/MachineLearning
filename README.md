**Predicting Loan Status Using Artificial Neural Network**

In this project, the goal is  to predict whether a loan applicant should be approved a loan or not based on 
predicted probabilities calculated as output from input features. 
 
**Dataset Overview**

The data file is attached (loan_approval_dataset.csv)

**Exploratory Data Analysis:**

Ordinal mapping is used to convert the string status to integer format to allow correlation among multiple features
Pearson Correlation is used to study the correlation between the loan status and input features and make decision of input fields
Sns Heatmap which is another map used to study the correlation between the loan status and input features
Sns Countplot to visualize the count of different classification in output

**Model Training in LoanApprovalPrediction.py**

The data is split into training and test sets (80-20 split). 
The model used was Feedforward neural network 
The model is run for 100 epochs and loss is shown for every 10th epoch

**Performance Metrics for test data:**

Loss: 0.06
Overall accuracy: 95% 

**Predictions on the Test Set:**

Generated predictions are stored in a CSV file(Loan_Prediction.csv).

**Conclusion:**

This project is a recommendation system to predict the Loan approval making it suitable for real-world applications in financial institutions.
