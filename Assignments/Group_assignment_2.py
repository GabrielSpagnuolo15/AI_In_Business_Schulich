# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD

# %%
# Load data
df = pd.read_csv(r"C:\Users\gabri\Downloads\farm-ads (1).csv",header=None)

# %%
# Preview data
df.head()

# %%
print(df.columns)

# %%
df.info()

# %%
# Rename columns
df.columns = ['label', 'ad_text']

# %%
df.info()

# %%
# Sample some relevant and non-relevant ads
relevant_ads = df[df['label'] == 1].sample(5)['ad_text'].tolist()
non_relevant_ads = df[df['label'] == -1].sample(5)['ad_text'].tolist()

print("\nSome relevant ads:")
for ad in relevant_ads:
    print(ad)

print("\nSome non-relevant ads:")
for ad in non_relevant_ads:
    print(ad)

# %% [markdown]
# # Create Term-Document Matrix: We'll create a Term-Document matrix to represent the frequency of terms (words) in each document (ad text).

# %% [markdown]
# # 1. TDM (W/ no TF-idf) and CDM (LSA & LDA)

# %%
# Step 1: Create Term-Document matrix
vectorizer = CountVectorizer()
term_document_matrix = vectorizer.fit_transform(df['ad_text'])

# Step 2: Create Concept-Document matrix using LSA (TruncatedSVD)
lsa_model = TruncatedSVD(n_components=20)  # Limiting to 20 concepts
concept_document_matrix = lsa_model.fit_transform(term_document_matrix)

# %%
# Print shapes of matrices for verification
print("Shape of Term-Document matrix:", term_document_matrix.shape)
print("Shape of Concept-Document matrix:", concept_document_matrix.shape)

# %%
term_document_df = pd.DataFrame(term_document_matrix.toarray(), columns=vectorizer.get_feature_names_out())
term_document_df.head()

# %%
# Convert Concept-Document matrix to DataFrame for easier printing
concept_document_df = pd.DataFrame(concept_document_matrix, columns=[f'Concept {i+1}' for i in range(20)])

# Print the Concept-Document matrix
concept_document_df.head()

# %%
n_components = 20
lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
concept_matrix = lda.fit_transform(term_document_matrix)

cdm = pd.DataFrame(concept_matrix, columns=[f'Concept {i+1}' for i in range(n_components)])


# %%
cdm.head()

# %% [markdown]
# # 1.1 TDM( with Tf-idf)

# %%
tfidf_vectorizer = TfidfVectorizer()

tdm_tfidf = tfidf_vectorizer.fit_transform(df['ad_text'])

tdm_tfidf_df = pd.DataFrame(tdm_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())


# %%
print("Shape of Term-Document matrix with TF-idf:", tdm_tfidf_df.shape)

# %%
tdm_tfidf_df.head(50)

# %%
tdm_tfidf_df.to_csv(r"C:\Users\gabri\Downloads\tdm_tfidf.csv")

# %% [markdown]
# # 2. Logistic Regression

# %%
# Split the data into training and validation sets (75% training, 25% validation)
X_train, X_val, y_train, y_val = train_test_split(term_document_matrix, df['label'], test_size=0.25, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict labels for validation set
y_pred = model.predict(X_val)


# %%
# Create a DataFrame to display predicted and actual labels
results_df = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})

# Display the DataFrame
print(results_df)

# %%
# Evaluate model performance
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# %%



