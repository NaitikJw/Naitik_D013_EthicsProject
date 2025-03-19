# **Ethics in Data Science – Gender Pay Gap Analysis**

## **Overview**
This project analyzes the **Gender Pay Gap** using salary data from **Glassdoor**, incorporating **ethical AI principles** such as **bias mitigation, transparency, privacy, and accountability** to ensure fairness in AI-driven decision-making.

## **Objectives**
- Identify gender-based salary disparities across different job roles, seniority levels, and education backgrounds.
- Implement ethical AI solutions to mitigate bias, improve transparency, and ensure accountability.
- Use **data visualization** to communicate key findings.
- Apply **machine learning models** to predict salaries while maintaining ethical considerations.

---

## **Dataset**
The dataset used in this project comes from **Glassdoor**, containing fields such as:
- `Gender`: Employee gender (Male/Female)
- `BasePay`: Base salary of the employee
- `Bonus`: Additional compensation
- `TotalPay`: Sum of BasePay and Bonus
- `JobTitle`: Employee’s role
- `Seniority`: Seniority level in the company
- `Education`: Highest education level attained

---

## **Project Workflow**
### **1️⃣ Data Preprocessing**
```python
import pandas as pd
# Load dataset
df = pd.read_csv('Glassdoor Gender Pay Gap.csv')
# Handle missing values
df.dropna(subset=['BasePay', 'Bonus'], inplace=True)
# Compute total salary
df['TotalPay'] = df['BasePay'] + df['Bonus']
df.head()
```
📌 **Ensures clean and structured data for analysis.**

### **2️⃣ Gender-Based Salary Analysis**
```python
gender_stats = df.groupby('Gender')[['BasePay', 'Bonus', 'TotalPay']].describe()
print(gender_stats)
```
📌 **Provides insights into salary differences between men and women.**

### **3️⃣ Data Visualization**
#### **Job Title Distribution**
```python
import matplotlib.pyplot as plt
title_counts = df['JobTitle'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
title_counts[:15].plot(kind='bar', color='skyblue')
plt.title('Top 10 Job Titles by Total Entries')
plt.xlabel('Job Title')
plt.ylabel('Total Entries')
plt.xticks(rotation=45)
plt.show()
```
📌 **Shows the most common job roles and potential gender disparities.**

#### **Seniority & Education Level by Gender**
```python
import plotly.graph_objs as go
import plotly.offline as py
# Create pie charts for gender representation
# (Code to generate interactive charts)
py.iplot(fig)
```
📌 **Visualizes how gender distribution varies by seniority and education level.**

---

## **Ethical AI Enhancements**

### **🔹 Bias Mitigation**
```python
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
df['GenderBinary'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
pay_gap_dataset = BinaryLabelDataset(df=df, label_names=['TotalPay'], protected_attribute_names=['GenderBinary'])
reweighing = Reweighing(unprivileged_groups=[{'GenderBinary': 0}], privileged_groups=[{'GenderBinary': 1}])
pay_gap_dataset_transf = reweighing.fit_transform(pay_gap_dataset)
```
📌 **Ensures gender fairness in salary predictions.**

### **🔹 Transparency & Explainability**
```python
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```
📌 **Explains model predictions and identifies salary-influencing factors.**

### **🔹 Privacy & Accountability**
```python
import logging
logging.basicConfig(filename='audit.log', level=logging.INFO)
logging.info("Salary predictions logged for audit trail.")
```
📌 **Maintains transparency by logging salary decisions.**

---

## **Final Outcomes & Insights**
- ✅ **Gender Pay Gap Identified:** Statistical analysis and visualizations highlighted pay disparities.
- ✅ **Ethical AI Solutions Implemented:** Bias mitigation, transparency, and accountability mechanisms were applied.
- ✅ **Data-Driven Decision Making:** Interactive charts helped communicate findings effectively.
- ✅ **Future Scope:** Expanding to include other demographic factors such as age, ethnicity, and location.

---

## **Conclusion**
This project successfully applied **ethical AI principles** to analyze and address the **gender pay gap**. By incorporating fairness-aware AI techniques, explainability, privacy safeguards, and accountability measures, we ensured responsible and unbiased data analysis.

Moving forward, these methods can be expanded to influence **real-world policy decisions** in employment and salary structures, helping organizations move toward **fairer and more transparent pay practices**.

🚀 **By combining data science with ethics, we take a step toward a more just and equitable workplace!**

---

## **How to Use This Project**
### ** Clone the Repository**
```bash
git clone https://github.com/NaitikJw/gender-pay-gap-analysis.git
cd gender-pay-gap-analysis
```


