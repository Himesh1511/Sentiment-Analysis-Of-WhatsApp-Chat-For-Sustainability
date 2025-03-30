import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import emoji

nltk.download('stopwords')

# Load chat data from a file
def load_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chat_data = file.readlines()

    pattern = r'(\d+/\d+/\d+, \d+:\d+ [APM]+) - (.*?): (.*)'

    messages = []
    for line in chat_data:
        match = re.match(pattern, line)
        if match:
            date, sender, message = match.groups()
            messages.append([date, sender, message])

    df = pd.DataFrame(messages, columns=['Date', 'Sender', 'Message'])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y, %I:%M %p', dayfirst=True)
    return df

# Analyze sentiments (positive, negative, neutral)
def analyze_sentiments(df):
    def get_sentiment(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    df['Sentiment'] = df['Message'].apply(get_sentiment)
    return df

# Classify formality (Formal or Informal)
def classify_formality(df):
    def is_formal(text):
        stop_words = set(stopwords.words('english'))
        words = text.lower().split()
        formal_word_count = sum(1 for word in words if word not in stop_words)
        return 'Formal' if formal_word_count > len(words) * 0.5 else 'Informal'

    df['Formality'] = df['Message'].apply(is_formal)
    return df

# Generate a word cloud image
def generate_wordcloud(df):
    text = " ".join(df['Message'])  # Combine all messages into a single text
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    
    wordcloud_image_path = 'static/wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)  # Save the image
    
    return wordcloud_image_path

# Generate emoji-specific word cloud image
def generate_emoji_wordcloud(df):
    # Extract only the emojis from the messages using the emoji library
    emojis = ''.join([char for char in ' '.join(df['Message']) if emoji.is_emoji(char)])
    
    # Check if there are any emojis extracted
    if not emojis:
        print("No emojis found in the messages.")
        # If no emojis, return None
        return None  # Or return a default image path if preferred
    
    # Create a wordcloud for emojis
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(emojis)
    
    # Save the emoji wordcloud image
    emoji_wordcloud_image_path = 'static/emoji_wordcloud.png'
    wordcloud.to_file(emoji_wordcloud_image_path)
    
    return emoji_wordcloud_image_path

# Analyze response time (time between messages)
def analyze_response_time(df):
    df['Time_Difference'] = df['Date'].diff().fillna(pd.Timedelta(seconds=0))
    df['Response_Time'] = df['Time_Difference'].dt.total_seconds() / 60  # Convert to minutes
    
    average_time = df['Response_Time'].mean()
    fastest_time = df['Response_Time'].min()
    slowest_time = df['Response_Time'].max()
    
    return {
        'average_time': round(average_time, 2),
        'fastest_time': round(fastest_time, 2),
        'slowest_time': round(slowest_time, 2)
    }

# Perform topic modeling using Latent Dirichlet Allocation (LDA)
def perform_topic_modeling(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    tfidf = vectorizer.fit_transform(df['Message'])
    
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(tfidf)
    
    # Get the topics
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
        topics.append({
            'topic': topic_idx + 1,
            'words': ", ".join(topic_words)
        })
    
    return topics

# Save summary of the analysis
def save_summary(summary_text):
    if not os.path.exists('static'):
        os.makedirs('static')

    with open('static/summary.txt', 'w') as f:
        f.write(summary_text)

# Create visualizations (message frequency, sentiment, formality, etc.)
def create_visualizations(df):
    if not os.path.exists('static'):
        os.makedirs('static')

    plt.figure(figsize=(12, 6))
    df['Day'] = df['Date'].dt.date
    daily_counts = df.groupby('Day').size()
    daily_counts.plot(kind='line', marker='o')
    plt.title('Number of Texts per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Texts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/texts_per_day.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    df['Week'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_counts = df.groupby('Week').size()
    weekly_counts.plot(kind='line', marker='o')
    plt.title('Number of Texts per Week')
    plt.xlabel('Week')
    plt.ylabel('Number of Texts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/texts_per_week.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    df['Month'] = df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
    monthly_counts = df.groupby('Month').size()
    monthly_counts.plot(kind='line', marker='o')
    plt.title('Number of Texts per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Texts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/texts_per_month.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    df['Year'] = df['Date'].dt.year
    yearly_counts = df.groupby('Year').size()
    yearly_counts.plot(kind='bar')
    plt.title('Number of Texts per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Texts')
    plt.tight_layout()
    plt.savefig('static/texts_per_year.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Sender', data=df)
    plt.title('Message Frequency by Sender')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/message_frequency.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.tight_layout()
    plt.savefig('static/sentiment_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='Formality', data=df)
    plt.title('Formality Distribution')
    plt.tight_layout()
    plt.savefig('static/formality_distribution.png')
    plt.close()

# Main function to analyze chat
def analyze_chat(file_path):
    df = load_chat(file_path)
    df = analyze_sentiments(df)
    df = classify_formality(df)
    response_time = analyze_response_time(df)
    topics = perform_topic_modeling(df)
    
    # Generate word cloud
    wordcloud_image = generate_wordcloud(df)
    
    # Generate emoji word cloud
    emoji_wordcloud_image = generate_emoji_wordcloud(df)
    
    # Handle case where no emoji wordcloud is generated
    if emoji_wordcloud_image is None:
        print("No emoji word cloud generated.")
        # You can set a default image or handle it here
    
    create_visualizations(df)

    # Create summary content (example)
    summary_text = "Analysis Summary:\n"
    summary_text += f"Total messages: {len(df)}\n"
    summary_text += f"Positive messages: {df[df['Sentiment'] == 'Positive'].shape[0]}\n"
    summary_text += f"Negative messages: {df[df['Sentiment'] == 'Negative'].shape[0]}\n"
    summary_text += f"Neutral messages: {df[df['Sentiment'] == 'Neutral'].shape[0]}\n"
    summary_text += f"Average Response Time: {response_time['average_time']} minutes\n"
    summary_text += f"Fastest Response: {response_time['fastest_time']} minutes\n"
    summary_text += f"Slowest Response: {response_time['slowest_time']} minutes\n"
    summary_text += "\nTopics Identified:\n"
    for topic in topics:
        summary_text += f"Topic {topic['topic']}: {topic['words']}\n"
    
    # Save summary to a file
    save_summary(summary_text)

    return summary_text, wordcloud_image, emoji_wordcloud_image, topics
