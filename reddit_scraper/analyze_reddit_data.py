from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

def show_word_cloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(str(data))


    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    plt.show()

def sentiment_analysis(engineered_data: pd.DataFrame):

    sentimentclass_list = []
    for i in range(0, len(engineered_data)):
        curr_compound = engineered_data['compound'][i]
        if (curr_compound <= 1.0 and curr_compound >= 0.55):
            sentimentclass_list.append(5)
        elif (curr_compound < 0.55 and curr_compound >= 0.10):
            sentimentclass_list.append(4)
        elif (curr_compound < 0.10 and curr_compound > -0.10):
            sentimentclass_list.append(3)
        elif (curr_compound <= -0.10 and curr_compound > -0.55):
            sentimentclass_list.append(2)
        elif (curr_compound <= -0.55 and curr_compound >= -1.00):
            sentimentclass_list.append(1)
    engineered_data['sentiment_class'] = sentimentclass_list
    engineered_data.tail()['sentiment_class']
    # Verify if the classification assignment is correct:
    engineered_data.iloc[0:5, :][['compound', 'sentiment_class']]

    plt.figure(figsize=(10, 5))
    sns.set_palette('PuBuGn_d')
    sns.countplot(engineered_data['sentiment_class'])
    plt.title('Countplot of sentiment_class')
    plt.xlabel('sentiment_class')
    plt.ylabel('No. of classes')
    plt.show()

    # Display full text:
    pd.set_option('display.max_colwidth', -1)

    # Look at some examples of negative, neutral and positive tweets

    # Filter 10 negative original tweets:
    print("10 random negative original tweets and their sentiment classes:")
    print(engineered_data[(engineered_data['sentiment_class'] == 1) | (engineered_data['sentiment_class'] == 2)].sample(n=10)[['body', 'sentiment_class']])

    # Filter 10 neutral original tweets:
    print("10 random neutral original tweets and their sentiment classes:")
    print(engineered_data[(engineered_data['sentiment_class'] == 3)].sample(n=10)[['body', 'sentiment_class']])

    # Filter 20 positive original tweets:
    print("20 random positive original tweets and their sentiment classes:")
    print(engineered_data[(engineered_data['sentiment_class'] == 4) | (engineered_data['sentiment_class'] == 5)].sample(n=20)[['body', 'sentiment_class']])


def feature_engineering(processed_data: pd.DataFrame):
    # Create a sid object called SentimentIntensityAnalyzer()
    sid = SentimentIntensityAnalyzer()

    # Apply polarity_score method of SentimentIntensityAnalyzer()
    processed_data['sentiment'] = processed_data['cleaned_body'].apply(lambda x: sid.polarity_scores(x))

    # Keep only the compound scores under the column 'Sentiment'
    data = pd.concat([processed_data.drop(['sentiment'], axis=1), processed_data['sentiment'].apply(pd.Series)], axis=1)

    # New column: number of characters in 'review'
    data['numchars'] = data['cleaned_body'].apply(lambda x: len(x))

    # New column: number of words in 'review'
    data['numwords'] = data['cleaned_body'].apply(lambda x: len(x.split(" ")))

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data["cleaned_body"].apply(lambda x: x.split(" ")))]

    model = Doc2Vec(documents, vector_size=4, window=2, min_count=1, workers=4)

    doc2vec_df = data["cleaned_body"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    data = pd.concat([data, doc2vec_df], axis=1)

    # Check the new columns:
    data.tail(2)

    tfidf = TfidfVectorizer(
        max_features=100,
        min_df=10,
        stop_words='english'
    )

    # Fit_transform our 'revi`ew' (the corpus) using the tfidf object from above
    tfidf_result = tfidf.fit_transform(data['cleaned_body']).toarray()

    # Extract the frequencies and store them in a temporary dataframe
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())

    # Rename the column names and index
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data.index

    # Concatenate the two dataframes - 'dataset' and 'tfidf_df'
    # Note: Axis = 1 -> add the 'tfidf_df' dataframe along the columns  or add these columns as columns in 'dataset'.
    data = pd.concat([data, tfidf_df], axis = 1)

    # Check out the new 'dataset' dataframe
    data.tail(2)

    return data

def preprocess_comments(data: pd.DataFrame):
    processed_data = data.copy(deep=True)
    processed_data.dropna(subset=['body'], inplace=True)
    processed_data['body'] = processed_data['body'].apply(lambda x: strip_chinese_words(x))
    processed_data['cleaned_body'] = processed_data['body'].apply(lambda x: clean_text(x))
    # Check out the shape again and reset_index
    print(processed_data.shape)
    processed_data.reset_index(inplace=True, drop=True)

    # Check out data.tail() to validate index has been reset
    processed_data.tail()
    return processed_data


def clean_text(text):
    # Define Emoji_patterns
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # Apply regex expressions first before converting string to list of tokens/words:
    # 1. remove @usernames
    text = re.sub('@[^\s]+', '', text)

    # 2. remove URLs and email addresses
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub('\S*@\S*\s?', '', text)

    # 3. remove hashtags entirely i.e. #hashtags
    text = re.sub(r'#([^\s]+)', '', text)

    # 4. remove emojis
    text = emoji_pattern.sub(r'', text)

    # 5. Convert text to lowercase
    text = text.lower()

    # 6. tokenize text and remove punctuation
    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # 7. remove numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]

    # 8. remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]

    # 9. remove empty tokens
    text = [t for t in text if len(t) > 0]

    # 10. pos tag text and lemmatize text
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # 11. remove words with only one letter
    text = [t for t in text if len(t) > 1]

    # join all
    text = " ".join(text)

    return (text)

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def strip_chinese_words(string):
    # list of english words
    en_list = re.findall(u'[^\u4E00-\u9FA5\u3000-\u303F]', str(string))

    # Remove word from the list, if not english
    for c in string:
        if c not in en_list:
            string = string.replace(c, '')
    return string

if __name__ == "__main__":
    path = os.getcwd()
    path = path.split('\\\\')
    file_path = r'\''.join(path)
    data = r'/reddit_data'
    df = pd.read_csv(file_path + data + "/20200210_163943_hkprotest_comments.csv")
    # show_word_cloud(df.body.tolist())
    processed_data = preprocess_comments(df)
    engineered_data = feature_engineering(processed_data)
    sentiment_analysis(engineered_data)
