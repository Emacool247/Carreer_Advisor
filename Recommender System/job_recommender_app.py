import streamlit as st
import pandas as p
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download WordNet if not already present
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_skills(text):
    """Normalize skill text by removing suffixes and lemmatizing."""
    text = text.lower()
    text = re.sub(r'\b(skill|skills)\b', '', text)
    tokens = [lemmatizer.lemmatize(t.strip()) for t in text.split(',')]
    return ', '.join(tokens)

def recommend_jobs(df2, user_skills_input, top_n=5):
    """
    Recommends jobs based on semantic similarity using Sentence-BERT.

    Parameters:
    - df2: DataFrame with 'Skills_Clean' and 'Job_Title_Clean'
    - user_skills_input: Raw user input string
    - top_n: Number of top jobs to recommend
    """

    # Clean dataset skills
    df2['Skills_Clean'] = df2['Skills_Clean'].apply(clean_skills)

    # Clean user input
    user_skills_input_cleaned = clean_skills(user_skills_input)

    # Generate embeddings
    job_embeddings = model.encode(df2['Skills_Clean'].tolist())
    user_embedding = model.encode([user_skills_input_cleaned])[0]

    # Compute semantic similarity
    similarities = cosine_similarity([user_embedding], job_embeddings)[0]
    df2['similarity'] = similarities

    # Aggregate by job title
    grouped = df2.groupby('Job_Title_Clean')['similarity'].mean().reset_index()
    top_jobs = grouped.sort_values(by='similarity', ascending=False).head(top_n)

    # Display recommendations
    st.subheader(f"Top {top_n} Job Recommendations Based on Your Skills")
    for i, row in top_jobs.iterrows():
        job_title = row['Job_Title_Clean']
        best_match = df2[df2['Job_Title_Clean'] == job_title].sort_values(by='similarity', ascending=False).iloc[0]

        required_skills = best_match['Skills_Clean']
        similarity_score = round(best_match['similarity'], 3)

        user_skills_set = set(user_skills_input_cleaned.split(', '))
        required_skills_set = set(required_skills.split(', '))

        overlapping = user_skills_set & required_skills_set
        missing = required_skills_set - user_skills_set

        st.markdown(f"### {job_title}")
        st.write(f"**Similarity Score:** {similarity_score}")
        st.write(f"**Required Skills:** {required_skills}")
        st.write(f"**Overlapping Skills:** {', '.join(overlapping) if overlapping else 'None'}")
        st.write(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

        if missing:
            st.info(f"To qualify for this role, consider learning: {', '.join(missing)}")
            st.markdown("**Suggested Learning Resources:**")
            st.markdown("- [Coursera](https://www.coursera.org)")
            st.markdown("- [Udemy](https://www.udemy.com)")
            st.markdown("- [LinkedIn Learning](https://www.linkedin.com/learning)")
        else:
            st.success("You already match the key skills for this role!")

# Streamlit UI
st.title("AI-Powered Job Recommender")
st.write("Enter your skills below to discover job roles that match your skills.")

uploaded_file = st.file_uploader("Joblisting_2.csv", type="csv")
if uploaded_file:
    df2 = pd.read_csv(uploaded_file)
    user_input = st.text_input("Enter your various skills:")
    if user_input:
        recommend_jobs(df2, user_input)