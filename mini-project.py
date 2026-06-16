import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
# pyrefly: ignore [missing-import]
from scipy.stats import chi2_contingency
import os


dt_AI_Student = pd.read_csv('Statistics-of-data-science-students-dataset/CS_Students_AI_Perception_Data.csv')

df=dt_AI_Student.drop(columns=['Q4_Other'])
# inspect Data 
print(df.info())
print(df.describe())

os.makedirs("images", exist_ok=True)

# 
sns.pairplot(df,vars=['Q1', 'Q2','Q3','Q4'],hue='Q5')
plt.savefig("images/pairplot.png", bbox_inches='tight')
plt.show()

# Handle missing 
df.fillna(df.mean(),inplace=True)
print(df.isnull().sum())

# Plot a bar chart for Q1 mapping 1 to 'M' and 2 to 'F'
df['Q1'].map({1: 'M', 2: 'F'}).value_counts().plot(kind='bar')
plt.title("Distribution of Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig("images/q1_gender.png", bbox_inches='tight')
plt.show()

df['Q2'].map({1: '<15', 2: '15-18', 3: '19-22', 4: '>22'}).value_counts().plot(kind='bar')
plt.title("Distribution of Age")
plt.xlabel("Age Range")
plt.ylabel("Count")
plt.savefig("images/q2_age.png", bbox_inches='tight')
plt.show()  

df['Q3'].map({1: 'Year 1', 2: 'Year 2', 3: 'Year 3', 4: 'Year 4'}).value_counts().plot(kind='bar')
plt.title("Distribution of Year of Study")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig("images/q3_year.png", bbox_inches='tight')
plt.show()      

df['Q4'].map({1: 'Software/Web/App', 2: 'Network/Cybersecurity', 3: 'Data Science/AI', 4: 'Other'}).value_counts().plot(kind='bar')
plt.title("Distribution of Area of Interest")
plt.xlabel("Area of Interest")
plt.ylabel("Count")
plt.savefig("images/q4_interest.png", bbox_inches='tight')
plt.show()

df['Q5'].map({1: 'ChatGPT/Gemini/Claude', 2: 'GitHub Copilot/Cursor', 3: 'Stack Overflow', 4: 'None'}).value_counts().plot(kind='bar')
plt.title("Distribution of AI Tools Used Most")
plt.xlabel("AI Tools")
plt.ylabel("Count")
plt.savefig("images/q5_ai_tools.png", bbox_inches='tight')
plt.show()

count = df.groupby(['Q5','Q1'])['Q1'].count().reset_index(name='Count')
sns.barplot(data=count,x='Q5',y='Count',hue='Q1')
plt.title("Distribution of AI Tools Used Most by Gender")
plt.xlabel("AI Tools")
plt.ylabel("Count")
plt.savefig("images/q5_q1_tools.png", bbox_inches='tight')
plt.show()



count = df.groupby(['Q5','Q2'])['Q2'].count().reset_index(name='Count')
sns.barplot(data=count,x='Q5',y='Count',hue='Q2')
plt.title("Distribution of AI Tools Used Most by Age")
plt.xlabel("AI Tools")
plt.ylabel("Count")
plt.savefig("images/q5_q2_tools.png", bbox_inches='tight')
plt.show()

df['Q6'].map({1: 'Every day', 2: 'Every week', 3: 'Only when stuck', 4: 'Never'}).value_counts().plot(kind='bar')
plt.title("Distribution of Frequency of AI Usage")
plt.xlabel("Frequency")
plt.ylabel("Count")
plt.savefig("images/q6_frequency.png", bbox_inches='tight')
plt.show()

df['Q7'].map({1: 'Generate Boilerplate Code', 2: 'Debugging & Error Fixing', 3: 'Explaining Logic', 4: 'Writing Documentation'}).value_counts().plot(kind='bar')
plt.title("Distribution of Primary Purpose of AI Usage")
plt.xlabel("Purpose")
plt.ylabel("Count")
plt.savefig("images/q7_purpose.png", bbox_inches='tight')
plt.show()

df['Q8'].map({1: '>50%', 2: '20-50%', 3: '<20%', 4: 'None'}).value_counts().plot(kind='bar')
plt.title("Distribution of AI Time Saved")
plt.xlabel("Time Saved")
plt.ylabel("Count")
plt.savefig("images/q8_time_saved.png", bbox_inches='tight')
plt.show()

df['Q9'].map({1: 'Copy and Paste', 2: 'Read & Understand Logic', 3: 'Verify with Best Practices'}).value_counts().plot(kind='bar')
plt.title("Action After Getting Code from AI")
plt.xlabel("Action")
plt.ylabel("Count")
plt.savefig("images/q9_action.png", bbox_inches='tight')
plt.show()

df['Q10'].map({1: 'High Trust', 2: 'Moderate Trust', 3: 'Low Trust'}).value_counts().plot(kind='bar')
plt.title("Trust Level in AI Generated Code")
plt.xlabel("Trust Level")
plt.ylabel("Count")
plt.savefig("images/q10_trust.png", bbox_inches='tight')
plt.show()

df['Q11'].map({1: 'Stopped using entirely', 2: 'Use both', 3: 'Prefer Stack Overflow', 4: 'Never used Stack Overflow'}).value_counts().plot(kind='bar')
plt.title("Stack Overflow Usage Change")
plt.xlabel("Usage Change")
plt.ylabel("Count")
plt.savefig("images/q11_stackoverflow.png", bbox_inches='tight')
plt.show()

df['Q12'].map({1: 'Essential', 2: 'Useful', 3: 'Not important'}).value_counts().plot(kind='bar')
plt.title("Importance of Prompt Engineering")
plt.xlabel("Importance Level")
plt.ylabel("Count")
plt.savefig("images/q12_prompt_engineering.png", bbox_inches='tight')
plt.show()

df['Q13'].map({1: 'High anxiety', 2: 'Moderate anxiety', 3: 'No anxiety'}).value_counts().plot(kind='bar')
plt.title("Anxiety About AI Replacing Junior Devs")
plt.xlabel("Anxiety Level")
plt.ylabel("Count")
plt.savefig("images/q13_anxiety.png", bbox_inches='tight')
plt.show()

df['Q14'].map({1: 'Deep Coding Syntax', 2: 'Logic & Problem-Solving', 3: 'AI Orchestration Skills'}).value_counts().plot(kind='bar')
plt.title("Crucial Skill for the Future")
plt.xlabel("Skill")
plt.ylabel("Count")
plt.savefig("images/q14_crucial_skill.png", bbox_inches='tight')
plt.show()

df['Q15'].map({1: 'Yes', 2: 'Yes with restrictions', 3: 'No'}).value_counts().plot(kind='bar')
plt.title("Allowing AI in University Exams")
plt.xlabel("Opinion")
plt.ylabel("Count")
plt.savefig("images/q15_exams.png", bbox_inches='tight')
plt.show()

df['Q16'].map({1: 'Code Security Vulnerabilities', 2: 'Loss of critical thinking', 3: 'AI Hallucinations'}).value_counts().plot(kind='bar')
plt.title("Biggest Risk of AI in Study")
plt.xlabel("Risk")
plt.ylabel("Count")
plt.savefig("images/q16_risk.png", bbox_inches='tight')
plt.show()

df['Q17'].map({1: 'Pass if can explain', 2: 'Fail', 3: 'Pass if works'}).value_counts().plot(kind='bar')
plt.title("Grading for 100% AI Generated Assignment")
plt.xlabel("Grading Option")
plt.ylabel("Count")
plt.savefig("images/q17_grading.png", bbox_inches='tight')
plt.show()

df['Q18'].map({1: 'Fully prepared', 2: 'Still learning', 3: 'Not ready'}).value_counts().plot(kind='bar')
plt.title("Preparedness to be an AI Orchestrator")
plt.xlabel("Preparedness Level")
plt.ylabel("Count")
plt.savefig("images/q18_preparedness.png", bbox_inches='tight')
plt.show()

df['Q19'].map({1: 'Excellent', 2: 'Good', 3: 'Average', 4: 'Poor'}).value_counts().plot(kind='bar')
plt.title("Quality of AI Generated Code")
plt.xlabel("Quality Level")
plt.ylabel("Count")
plt.savefig("images/q19_quality.png", bbox_inches='tight')
plt.show()

df['Q20'].map({1: 'Excellent', 2: 'Good but be careful', 3: 'Not good', 4: 'Poor'}).value_counts().plot(kind='bar')
plt.title("AI Appropriateness for Beginners")
plt.xlabel("Appropriateness")
plt.ylabel("Count")
plt.savefig("images/q20_beginners.png", bbox_inches='tight')
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os


dt_AI_Student = pd.read_csv('Statistics-of-data-science-students-dataset/CS_Students_AI_Perception_Data.csv')

df=dt_AI_Student.drop(columns=['Q4_Other'])
# inspect Data 
print(df.info())
print(df.describe())

os.makedirs("images", exist_ok=True)

# 
sns.pairplot(df,vars=['Q1', 'Q2','Q3','Q4'],hue='Q5')
plt.savefig("images/pairplot.png", bbox_inches='tight')
plt.show()

# Handle missing 
df.fillna(df.mean(),inplace=True)
print(df.isnull().sum())

# Plot a bar chart for Q1 mapping 1 to 'M' and 2 to 'F'
df['Q1'].map({1: 'M', 2: 'F'}).value_counts().plot(kind='bar')
plt.title("Distribution of Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig("images/q1_gender.png", bbox_inches='tight')
plt.show()

df['Q2'].map({1: '<15', 2: '15-18', 3: '19-22', 4: '>22'}).value_counts().plot(kind='bar')
plt.title("Distribution of Age")
plt.xlabel("Age Range")
plt.ylabel("Count")
plt.savefig("images/q2_age.png", bbox_inches='tight')
plt.show()  

df['Q3'].map({1: 'Year 1', 2: 'Year 2', 3: 'Year 3', 4: 'Year 4'}).value_counts().plot(kind='bar')
plt.title("Distribution of Year of Study")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig("images/q3_year.png", bbox_inches='tight')
plt.show()      

df['Q4'].map({1: 'Software/Web/App', 2: 'Network/Cybersecurity', 3: 'Data Science/AI', 4: 'Other'}).value_counts().plot(kind='bar')
plt.title("Distribution of Area of Interest")
plt.xlabel("Area of Interest")
plt.ylabel("Count")
plt.savefig("images/q4_interest.png", bbox_inches='tight')
plt.show()

df['Q5'].map({1: 'ChatGPT/Gemini/Claude', 2: 'GitHub Copilot/Cursor', 3: 'Stack Overflow', 4: 'None'}).value_counts().plot(kind='bar')
plt.title("Distribution of AI Tools Used Most")
plt.xlabel("AI Tools")
plt.ylabel("Count")
plt.savefig("images/q5_ai_tools.png", bbox_inches='tight')
plt.show()

count = df.groupby(['Q5','Q1'])['Q1'].count().reset_index(name='Count')
sns.barplot(data=count,x='Q5',y='Count',hue='Q1')
plt.title("Distribution of AI Tools Used Most by Gender")
plt.xlabel("AI Tools")
plt.ylabel("Count")
plt.savefig("images/q5_q1_tools.png", bbox_inches='tight')
plt.show()



count = df.groupby(['Q5','Q2'])['Q2'].count().reset_index(name='Count')
sns.barplot(data=count,x='Q5',y='Count',hue='Q2')
plt.title("Distribution of AI Tools Used Most by Age")
plt.xlabel("AI Tools")
plt.ylabel("Count")
plt.savefig("images/q5_q2_tools.png", bbox_inches='tight')
plt.show()

df['Q6'].map({1: 'Every day', 2: 'Every week', 3: 'Only when stuck', 4: 'Never'}).value_counts().plot(kind='bar')
plt.title("Distribution of Frequency of AI Usage")
plt.xlabel("Frequency")
plt.ylabel("Count")
plt.savefig("images/q6_frequency.png", bbox_inches='tight')
plt.show()

df['Q7'].map({1: 'Generate Boilerplate Code', 2: 'Debugging & Error Fixing', 3: 'Explaining Logic', 4: 'Writing Documentation'}).value_counts().plot(kind='bar')
plt.title("Distribution of Primary Purpose of AI Usage")
plt.xlabel("Purpose")
plt.ylabel("Count")
plt.savefig("images/q7_purpose.png", bbox_inches='tight')
plt.show()

df['Q8'].map({1: '>50%', 2: '20-50%', 3: '<20%', 4: 'None'}).value_counts().plot(kind='bar')
plt.title("Distribution of AI Time Saved")
plt.xlabel("Time Saved")
plt.ylabel("Count")
plt.savefig("images/q8_time_saved.png", bbox_inches='tight')
plt.show()

df['Q9'].map({1: 'Copy and Paste', 2: 'Read & Understand Logic', 3: 'Verify with Best Practices'}).value_counts().plot(kind='bar')
plt.title("Action After Getting Code from AI")
plt.xlabel("Action")
plt.ylabel("Count")
plt.savefig("images/q9_action.png", bbox_inches='tight')
plt.show()

df['Q10'].map({1: 'High Trust', 2: 'Moderate Trust', 3: 'Low Trust'}).value_counts().plot(kind='bar')
plt.title("Trust Level in AI Generated Code")
plt.xlabel("Trust Level")
plt.ylabel("Count")
plt.savefig("images/q10_trust.png", bbox_inches='tight')
plt.show()

df['Q11'].map({1: 'Stopped using entirely', 2: 'Use both', 3: 'Prefer Stack Overflow', 4: 'Never used Stack Overflow'}).value_counts().plot(kind='bar')
plt.title("Stack Overflow Usage Change")
plt.xlabel("Usage Change")
plt.ylabel("Count")
plt.savefig("images/q11_stackoverflow.png", bbox_inches='tight')
plt.show()

df['Q12'].map({1: 'Essential', 2: 'Useful', 3: 'Not important'}).value_counts().plot(kind='bar')
plt.title("Importance of Prompt Engineering")
plt.xlabel("Importance Level")
plt.ylabel("Count")
plt.savefig("images/q12_prompt_engineering.png", bbox_inches='tight')
plt.show()

df['Q13'].map({1: 'High anxiety', 2: 'Moderate anxiety', 3: 'No anxiety'}).value_counts().plot(kind='bar')
plt.title("Anxiety About AI Replacing Junior Devs")
plt.xlabel("Anxiety Level")
plt.ylabel("Count")
plt.savefig("images/q13_anxiety.png", bbox_inches='tight')
plt.show()

df['Q14'].map({1: 'Deep Coding Syntax', 2: 'Logic & Problem-Solving', 3: 'AI Orchestration Skills'}).value_counts().plot(kind='bar')
plt.title("Crucial Skill for the Future")
plt.xlabel("Skill")
plt.ylabel("Count")
plt.savefig("images/q14_crucial_skill.png", bbox_inches='tight')
plt.show()

df['Q15'].map({1: 'Yes', 2: 'Yes with restrictions', 3: 'No'}).value_counts().plot(kind='bar')
plt.title("Allowing AI in University Exams")
plt.xlabel("Opinion")
plt.ylabel("Count")
plt.savefig("images/q15_exams.png", bbox_inches='tight')
plt.show()

df['Q16'].map({1: 'Code Security Vulnerabilities', 2: 'Loss of critical thinking', 3: 'AI Hallucinations'}).value_counts().plot(kind='bar')
plt.title("Biggest Risk of AI in Study")
plt.xlabel("Risk")
plt.ylabel("Count")
plt.savefig("images/q16_risk.png", bbox_inches='tight')
plt.show()

df['Q17'].map({1: 'Pass if can explain', 2: 'Fail', 3: 'Pass if works'}).value_counts().plot(kind='bar')
plt.title("Grading for 100% AI Generated Assignment")
plt.xlabel("Grading Option")
plt.ylabel("Count")
plt.savefig("images/q17_grading.png", bbox_inches='tight')
plt.show()

df['Q18'].map({1: 'Fully prepared', 2: 'Still learning', 3: 'Not ready'}).value_counts().plot(kind='bar')
plt.title("Preparedness to be an AI Orchestrator")
plt.xlabel("Preparedness Level")
plt.ylabel("Count")
plt.savefig("images/q18_preparedness.png", bbox_inches='tight')
plt.show()

df['Q19'].map({1: 'Excellent', 2: 'Good', 3: 'Average', 4: 'Poor'}).value_counts().plot(kind='bar')
plt.title("Quality of AI Generated Code")
plt.xlabel("Quality Level")
plt.ylabel("Count")
plt.savefig("images/q19_quality.png", bbox_inches='tight')
plt.show()

df['Q20'].map({1: 'Excellent', 2: 'Good but be careful', 3: 'Not good', 4: 'Poor'}).value_counts().plot(kind='bar')
plt.title("AI Appropriateness for Beginners")
plt.xlabel("Appropriateness")
plt.ylabel("Count")
plt.savefig("images/q20_beginners.png", bbox_inches='tight')
plt.show()

df['Q21'].map({1: 'Yes', 2: 'No', 3: 'Maybe', 4: 'Not sure'}).value_counts().plot(kind='bar')
plt.title("Recommending AI to Friends")
plt.xlabel("Recommendation")
plt.ylabel("Count")
plt.savefig("images/q21_recommend.png", bbox_inches='tight')
plt.show()



print("\n--- Statistical Tests ---")
# Chi-Square Test between Q5 and Q1
contingency_table_q5_q1 = pd.crosstab(df['Q5'], df['Q1'])
chi2, p, dof, expected = chi2_contingency(contingency_table_q5_q1)
print(f"Chi-Square Test for Q5 (AI Tools) vs Q1 (Gender):")
print(f"Chi2 Statistic: {chi2:.4f}, p-value: {p:.4f}")

# Chi-Square Test between Q5 and Q2
contingency_table_q5_q2 = pd.crosstab(df['Q5'], df['Q2'])
chi2, p, dof, expected = chi2_contingency(contingency_table_q5_q2)
print(f"Chi-Square Test for Q5 (AI Tools) vs Q2 (Age):")
print(f"Chi2 Statistic: {chi2:.4f}, p-value: {p:.4f}")
