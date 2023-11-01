#!/usr/bin/env python
# coding: utf-8

# # Airlines Customer Satisfaction

# **Author**: Monsoon Dutta 
# **Date** : 30th Oct, 2023

# ### Project Overview
This project aims to predict whether a future customer would be satisfied with their service given the details of the other parameters values. Also the airlines need to know on which aspect of the services offered by them have to be emphasized more to generate more satisfied customers.
# ### Problem Statement

# The main problem we are addressing is to identify the key drivers of customer satisfaction in the airline industry. Our goals include:
# 
# 1. Analyzing customer feedback data to understand the factors influencing satisfaction.
# 2. Proposing actionable recommendations for improving customer satisfaction.
# 3. Evaluating the impact of these recommendations on overall customer satisfaction.
# 

# ### Data Source

# The data used in this analysis was collected from surveys and feedback forms provided by an airline company and the dataset is from Kaggle. 
# The actual name of the company is not given due to various purposes that is why the name Invistico Airlines.
# It includes responses from passengers regarding their travel experiences, preferences, and levels of satisfaction.

# In[109]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[110]:


#reading the dataset
df=pd.read_csv("Invistico_Airline.csv")


# In[111]:


df.head()


# In[112]:


df.shape


# There are 129880 rows and 22 columns.

# In[113]:


#showing the counts and datatypes of variables
df.info()


# In[114]:


#finding the missing values
df.isnull().sum()


# There are 393 missing values in Arrival Delay in Minutes column.

# # Data Cleaning

# In[115]:


#As there is substantial volume of data and just 393 missing values, so removing some records with null values 
#is unlikely to significantly impact overall performance

df.dropna(inplace=True)


# In[116]:


df.isnull().sum()


# In[117]:


#Maximum Age
df["Age"].max()


# In[118]:


#Apply binning to the age column
bins = [0, 20, 30, 40, 50, 60, 70, 80, float('inf')] 
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']

df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)


# In[119]:


#dropping the Age column
df.drop(columns=["Age"],axis=1, inplace=True)


# In[120]:


df.head(3)


# # EDA

# In[125]:


# rcParams to customize the plot
sns.set_style("darkgrid")
plt.rcParams["font.size"]=14
plt.rcParams["figure.figsize"]=(12,7)
plt.rcParams["figure.facecolor"]="#FFE5B4"   


# ### 1) Satisfaction vs. Dissatisfaction

# In[122]:


types=df["satisfaction"].value_counts()


# In[123]:


custom_palette = ["#87CEEB", "#F08080"]
#creating subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#countplot
ax1 = sns.countplot(x="satisfaction", data=df, palette=custom_palette, ax=axes[0])
sns.set(rc={"figure.figsize": (4, 4)})
for bars in ax1.containers:
    ax1.bar_label(bars)
ax1.set_title("Count of satisfaction", fontweight="bold", fontsize=16)

labels = df["satisfaction"].value_counts().index
types = df["satisfaction"].value_counts()
custom_colors = ["skyblue", "lightcoral"]
#Pie Chart
ax2 = axes[1]
ax2.pie(types, labels=labels, autopct='%1.1f%%', colors=custom_colors)
ax2.set_title("Satisfaction vs Dissatisfaction", fontweight="bold", fontsize=16)
plt.tight_layout()

plt.show()


# From the above fig., we can see that there are 70882 customers who are satisfied and 58605 who are not satisfied.
# Most of the customers are satisfied as we can see their percentage distribution also which can clearly say thst 54.7% customers
# are satisfied as compared to 45.3% of the customers who are dissatisfied.

# ### 2) Satisfaction / Dissatisfaction based on Age

# In[126]:


plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Age Group', hue='satisfaction', palette={"satisfied": "steelblue", "dissatisfied": "lightcoral"})
plt.title('Satisfaction by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title="Satisfaction", loc="upper right", labels=["Satisfied", "Unsatisfied"])
plt.show()


# From the above fig., we can see that 41-50 age of customers are mostly satisfied whereas 21-30 age of customers are not satisfied.

# ### 3) Satisfaction vs Dissatisfaction based on Class of Travel

# In[127]:


custom_palette = ["#87CEEB", "#F08080", "#4682B4"]
custom_colors = ["skyblue", "lightcoral", "steelblue"]
#creating subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#countplot
ax1 = sns.countplot(x="Class", data=df, palette=custom_palette, ax=axes[0, 0])
for bars in ax1.containers:
    ax1.bar_label(bars)
ax1.set_title("Count of Class", fontweight="bold", fontsize=16)
labels = df["Class"].value_counts().index
types = df["Class"].value_counts()
#Piechart
ax2 = axes[0, 1]
ax2.pie(types, labels=labels, autopct='%1.1f%%', colors=custom_colors)
ax2.set_title("Distribution of Class", fontweight="bold", fontsize=16)
#countplot
ax3 = sns.countplot(x="Class", hue="satisfaction", data=df, palette={"satisfied": "steelblue", "dissatisfied": "lightcoral"}, ax=axes[1, 0])
for bars in ax3.containers:
    ax3.bar_label(bars)
ax3.set_title("Class vs. Satisfaction", fontweight="bold", fontsize=16)
ax3.set_xlabel("Class of Travel")
ax3.set_ylabel("Count")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.legend(title="Satisfaction", loc="upper right", labels=["Satisfied", "Unsatisfied"])
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()


# From the above fig., we can see that there are 47.9% customers i.e. 61990 customers who are from business class.
# Economy (Eco) customers typically have standard seating with limited legroom and pay a lower fare, while Economy Plus customers enjoy extra legroom, greater comfort, and pay a premium for their seats. Economy Plus(eco plus) offers an upgraded travel experience with amenities such as priority boarding, better meals, and more space.
# From the fig., we can see that 7.2% customers i.e. 9380 are from eco plus and 44.9% customers i.e. 58117 are from eco.
# Mostly Business class customers are satisfied i.e. 43977 customers. Eco customers are significantly dissatisfied i.e. 35219.
# The satisfaction and dissatisfaction numbers of Eco Plus customers are almost similar and less.

# ### 4) Satisfaction vs Dissatisfaction based on Type of Travel

# In[128]:


custom_palette = ["#87CEEB", "#F08080"]
custom_colors = ["skyblue", "lightcoral"]
#creating subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#countplot
ax1 = sns.countplot(x="Type of Travel", data=df, palette=custom_palette, ax=axes[0, 0])
for bars in ax1.containers:
    ax1.bar_label(bars)
ax1.set_title("Type of Travel counts", fontweight="bold", fontsize=16)
labels = df["Type of Travel"].value_counts().index
types = df["Type of Travel"].value_counts()
#piechart
ax2 = axes[0, 1]
ax2.pie(types, labels=labels, autopct='%1.1f%%', colors=custom_colors)
ax2.set_title("Distribution of Type of Travel", fontweight="bold", fontsize=16)
#countplot
ax3 = sns.countplot(x="Type of Travel", hue="satisfaction", data=df, palette={"satisfied": "steelblue", "dissatisfied": "lightcoral"}, ax=axes[1, 0])
for bars in ax3.containers:
    ax3.bar_label(bars)
ax3.set_title("Type of Travel vs. Satisfaction", fontweight="bold", fontsize=16)
ax3.set_xlabel("Type of Travel")
ax3.set_ylabel("Count")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.legend(title="Satisfaction", loc="upper right", labels=["Satisfied", "Unsatisfied"])
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()


# From the fig., we can see that the 69.1% customers i.e. 89445 are travelling for business purposes. And 30.9% customers i.e 40042 are personal travelers. Business travelers are mostly satisfied i.e. 52207 customers. But similarly we can see that Business travelers are highly dissatisfied i.e. 37238.

# ### 5) Satisfaction vs Dissatisfaction based on Type of Customer

# In[129]:


custom_palette = ["#87CEEB", "#F08080"]
custom_colors = ["skyblue", "lightcoral"]
#creating the subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#countplot
ax1 = sns.countplot(x="Customer Type", data=df, palette=custom_palette, ax=axes[0, 0])
for bars in ax1.containers:
    ax1.bar_label(bars)
ax1.set_title("Count of Customer Types", fontweight="bold", fontsize=16)
labels = df["Customer Type"].value_counts().index
types = df["Customer Type"].value_counts()
#piechart
ax2 = axes[0, 1]
ax2.pie(types, labels=labels, autopct='%1.1f%%', colors=custom_colors)
ax2.set_title("Distribution of Customer Type", fontweight="bold", fontsize=16)
#countplot
ax3 = sns.countplot(x="Customer Type", hue="satisfaction", data=df, palette={"satisfied": "steelblue", "dissatisfied": "lightcoral"}, ax=axes[1, 0])
for bars in ax3.containers:
    ax3.bar_label(bars)
ax3.set_title("Customer Type vs. Satisfaction", fontweight="bold", fontsize=16)
ax3.set_xlabel("Customer Type")
ax3.set_ylabel("Count")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.legend(title="Satisfaction", loc="upper right", labels=["Satisfied", "Unsatisfied"])
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()


# From the above fig., we can see that 81.7% customers i.e. 105773 are loyal as compared to dissloyal customers which is 18.3% i.e. 23714. Satisfaction and Dissatisfaction responses are mostly from Loyal Customers. The number of loyal type customers i.e. 65194 are highly satisfied.

# ### 6) Satisfaction vs Dissatisfaction based on Ratings

# Airline ratings typically use a scale of 0 to 5, with each number having a specific meaning or interpretation:
# 
# 0: Poor - This rating suggests a very unsatisfactory experience with the airline, often indicating severe issues such as delays, cancellations, poor customer service, or uncomfortable conditions.
# 
# 1: Below Average - A rating of 1 means that the airline's performance falls below the average standard. It might involve issues like limited amenities, average service quality, or occasional delays.
# 
# 2: Average - This rating indicates that the airline provides a standard, middle-of-the-road experience. Nothing outstanding but no significant problems either. It's an okay choice for most travelers.
# 
# 3: Good - A rating of 3 signifies that the airline offers a positive experience overall. It may have good customer service, on-time performance, and reasonable comfort.
# 
# 4: Very Good - A rating of 4 suggests that the airline is an excellent choice, offering high-quality service, exceptional amenities, and a very comfortable travel experience.
# 
# 5: Excellent - This is the highest rating, indicating an exceptional airline experience. Such airlines often go above and beyond in terms of customer service, punctuality, and in-flight services.

# ### Seat Comfort

# In[130]:


counts = df["Seat comfort"].value_counts()


# In[131]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
#bar plot
plt.bar(counts.index, counts)
plt.xlabel('Seat Comfort Rating')
plt.ylabel('Count')
plt.title('Seat Comfort Ratings')

# Create the second subplot i.e. countplot
plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Seat comfort', hue='satisfaction', palette={"satisfied": "steelblue", "dissatisfied": "lightcoral"})
plt.title('Satisfaction by Seat Comfort')
plt.xlabel('Seat Comfort')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title="Satisfaction", loc="upper left", labels=["Satisfied", "Unsatisfied"])
plt.tight_layout()
plt.show()


# From the above fig., we can clearly see that most of the customers gave 3 star rating to the airlines for Seat comfort. 3 star ratings given by customers are highly not satisfied. 

# ### Ratings Analysis of the following:-
# 'Departure/Arrival time convenient',
# 'Food and drink',
#  'Gate location',
# 'Inflight wifi service'
# 'Inflight entertai,nment'
#  'Online support',
# 'Ease of Online bo,oking'
#  'On-board service',
# 'Leg room service',
# 'Baggage handling',
# 'Checkin service',
# 'Cleanliness',
# 'Online boarding'.

# In[132]:


def create_subplot(variable, df):
    counts = df[variable].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(counts.index, counts)
    axes[0].set_xlabel(f'{variable} Rating')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{variable} Ratings')
    
    sns.countplot(data=df, x=variable, hue='satisfaction', palette={"satisfied": "steelblue", "dissatisfied": "lightcoral"}, ax=axes[1])
    axes[1].set_title(f'Satisfaction by {variable}')
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel('Count')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title="Satisfaction", loc="upper left", labels=["Satisfied", "Unsatisfied"])
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)  # You can increase this value for more spacing
    
    # Add titles to each plot
    plt.suptitle(f'Analysis of {variable}', fontsize=16, y=1.05)  # Adjust the 'y' value for title position
    plt.show()

variables_to_visualize = [
    'Departure/Arrival time convenient',
    'Food and drink',
    'Gate location',
    'Inflight wifi service',
    'Inflight entertainment',
    'Online support',
    'Ease of Online booking',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Cleanliness',
    'Online boarding'
]
for variable in variables_to_visualize:
    create_subplot(variable, df)


# From the above figures, we can see that maximum customers gave 4 star ratings for Departure/Arrival time convenient, inflight wifi service, inflight entertainment, online support, ease of online booking, On-board service, leg room service, baggage handling, checkin-service, cleanliness and online boarding. But they give 3 star rating for Food and Drink.
# It means they are mostly satisfied for all the factors but unsatisfied in Food and Drink and Gate Location.

# # Outlier Detection

# In[133]:


#outlier detection only for these three columns
df[["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]]


# In[134]:


#visualizing outliers through boxplot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(x=df["Flight Distance"], ax=axes[0])
axes[0].set_xlabel("Flight Distance")
axes[0].set_title("Box Plot for Flight Distance")

sns.boxplot(x=df["Departure Delay in Minutes"], ax=axes[1])
axes[1].set_xlabel("Departure Delay (minutes)")
axes[1].set_title("Box Plot for Departure Delay")

sns.boxplot(x=df["Arrival Delay in Minutes"], ax=axes[2])
axes[2].set_xlabel("Arrival Delay (minutes)")
axes[2].set_title("Box Plot for Arrival Delay")
plt.tight_layout()

plt.show()


# From the above fig., we can clearly see that in the above three variables, there are some outliers.

# In[135]:


q1 = np.percentile(df["Flight Distance"], 25)
q2 = np.percentile(df["Flight Distance"], 75)
iqr = q2 - q1                                 #interquartile range
lower_bound1 = q1 - 1.5 * iqr
upper_bound1 = q2 + 1.5 * iqr

q3 = np.percentile(df["Departure Delay in Minutes"], 25)
q4 = np.percentile(df["Departure Delay in Minutes"], 75)
iqr = q4 - q3
lower_bound2 = q3-1.5 * iqr
upper_bound2 = q4+ 1.5 * iqr

q5 = np.percentile(df["Arrival Delay in Minutes"], 25)
q6 = np.percentile(df["Arrival Delay in Minutes"], 75)
iqr = q6 - q5
lower_bound3 = q5 - 1.5 * iqr
upper_bound3 = q6 + 1.5 * iqr

print("The lower bound of Flight Distance is", lower_bound1)
print("The upper bound of Flight Distance is", upper_bound1)

print("The lower bound of Departure Delay in Minutes is", lower_bound2)
print("The upper bound of Departure Delay in Minutes is", upper_bound2)

print("The lower bound of Arrival Delay in Minutes is", lower_bound3)
print("The upper bound of Arrival Delay in Minutes is", upper_bound3)


# In[136]:


#outlier treatment

df["Flight Distance"] = np.where(df["Flight Distance"] > upper_bound1, upper_bound1, df["Flight Distance"])
df["Flight Distance"] = np.where(df["Flight Distance"] < lower_bound1, lower_bound1, df["Flight Distance"])

df["Departure Delay in Minutes"] = np.where(df["Departure Delay in Minutes"] > upper_bound2, upper_bound2, df["Departure Delay in Minutes"])
df["Departure Delay in Minutes"] = np.where(df["Departure Delay in Minutes"] < lower_bound2, lower_bound2, df["Departure Delay in Minutes"])

df["Arrival Delay in Minutes"] = np.where(df["Arrival Delay in Minutes"] > upper_bound3, upper_bound3, df["Arrival Delay in Minutes"])
df["Arrival Delay in Minutes"] = np.where(df["Arrival Delay in Minutes"] < lower_bound3, lower_bound3, df["Arrival Delay in Minutes"])



# In[137]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(x=df["Flight Distance"], ax=axes[0])
axes[0].set_xlabel("Flight Distance")
axes[0].set_title("Box Plot for Flight Distance")

sns.boxplot(x=df["Departure Delay in Minutes"], ax=axes[1])
axes[1].set_xlabel("Departure Delay (minutes)")
axes[1].set_title("Box Plot for Departure Delay")

sns.boxplot(x=df["Arrival Delay in Minutes"], ax=axes[2])
axes[2].set_xlabel("Arrival Delay (minutes)")
axes[2].set_title("Box Plot for Arrival Delay")
plt.tight_layout()

plt.show()


# From the above fig., we can see that all the outliers are treated.

# # Conversion of categorical to numerical

# In[138]:


df["satisfaction"] = np.where(df.satisfaction == "satisfied", 1,0)


# In[139]:


df.head(3)


# In[140]:


#decoding categorical to numerical 
le = LabelEncoder()

df["Customer Type"] = le.fit_transform(df["Customer Type"])
df["Type of Travel"] = le.fit_transform(df["Type of Travel"])
df["Class"] = le.fit_transform(df["Class"])
df.head(3)


# In[141]:


#one hot encoding
df2=pd.get_dummies(df, columns=["Age Group"])
df2=df2.astype(int)


# In[142]:


df2.head()


# # Correlation  Analysis

# In[143]:


plt.figure(figsize=(12, 8))  # Adjust the figure size to something more appropriate
sns.heatmap(df2.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# # Model Building

# In[144]:


x=df2.drop("satisfaction", axis=1)
x.head(3)


# In[145]:


y=df2["satisfaction"]
y.head(3)


# In[146]:


#splitting the data into test and train

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)


# In[147]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# ### Decision Tree Classifier

# In[87]:


model_dt=DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6)


# In[88]:


model_dt.fit(x_train, y_train)


# In[89]:


y_pred=model_dt.predict(x_test)


# In[90]:


y_pred


# In[91]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# The current model shows good precision, recall, and an overall F1-score of 0.89, with an accuracy of 90%. It performs well in distinguishing between the two classes.

# In[92]:


print(confusion_matrix(y_test, y_pred)) 


# The overall accuracy is 89.5%.
# But, we can explore alternative models to further enhance the F1-score and improve performance, aiming for higher accuracy.

# ### Random Forest Classifier

# In[96]:


model_rf=RandomForestClassifier(n_estimators=100, criterion="gini", random_state=100, max_depth=6)
model_rf.fit(x_train, y_train)
y_pred_rf=model_dt.predict(x_test)


# In[94]:


print(classification_report(y_test, y_pred_rf, labels=[0,1]))


# We got the similar precision, recall anf F1 score as the above model.

# In[95]:


print(confusion_matrix(y_test, y_pred_rf))


# The overall accuracy is same i.e. 89.5%. 

# ### Gradient Boosting Classifier

# In[100]:


model_grad= GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, 
                                       min_samples_split=2, min_samples_leaf=1, loss="log_loss", random_state=42)
model_grad.fit(x_train, y_train)
y_pred_grad=model_grad.predict(x_test)


# In[98]:


print(classification_report(y_test, y_pred_grad, labels=[0,1]))


# The gradient boosting model exhibits significant performance enhancements, with precision, recall, and F1-score all increasing to 0.92, resulting in an impressive accuracy of 92%.

# In[99]:


print(confusion_matrix(y_test, y_pred_grad))


# The overall accuracy got improved to 91.8%.

# In[101]:


#saving the model

import pickle


# In[102]:


filename="model.sav"


# In[103]:


pickle.dump(model_grad, open(filename, "wb"))


# In[104]:


load_model=pickle.load(open(filename, "rb"))


# In[105]:


load_model.score(x_test, y_test)


# In[106]:


classes = ["Dissatisfied", "Satisfied"]

cm = confusion_matrix(y_test, y_pred_grad)

# Create a figure and axis
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# Add labels to the plot
tick_marks = np.arange(len(classes))  # Corrected this line
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Add text annotations for each cell
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()
plt.show()


# # CONCLUSION:

# In this project we found the following insights:-
# 
# 1) 54.7% are satisfied customers as compared to dissatisfied customers.
# 2) Age group between 41-50 age of customers are highly satisfied.
# 3) Business class customers are mostly satisfied followed by Economy customers and Economy Plus customers.
# 4) Business travellers are more and they are more satisfied as compared to personal travellers.
# 5) 81.7% customers are loyal and highly satisfied as compared to dissloyal which is 18.3%.
# 6) In terms of ratings, customers are highly satisfied for  Departure/Arrival time convenient, inflight wifi service, inflight entertainment, online support, ease of online booking, On-board service, leg room service, baggage handling, checkin-service, cleanliness and online boarding as they gave 4 star ratings in these factors which  indicates a relatively high level of customer satisfaction.
# 
# ### Recommendations for improvements:- 
# In conclusion, for better gate location, airline should use better signs and show real-time information for passengers to find their way easily. To make seats more comfy, use seats that can be adjusted and are softer to sit on. For food and drinks, have more kinds of food, including healthy options, and consider different diets to make in-flight eating better for passengers.
# 
# And lastly we can see that the Gradient Boosting Classifier machine learning model delivered outstanding performance, achieving a remarkable accuracy rate of 92% which reflects a high level of satisfaction with the model's performance.
