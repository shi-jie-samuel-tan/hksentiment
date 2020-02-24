import praw
import pandas as pd
from datetime import datetime
import os

reddit = praw.Reddit(client_id='L3dY4J5tSS6Srw',
                     client_secret='JDH8ov5V7SPDFiaZNToTa0iD1w0',
                     user_agent='SaltPlusPepper ',
                     username='SaltPlusPepper',
                     password='ABCabc123,./')
subreddit = reddit.subreddit('HongKong')

def scrape_reddit(num_of_top_threads):

    db_threads = pd.DataFrame(columns=['id', 'name', 'author', 'comments', 'num_comments', 'created', 'score',
                                       'upvote_ratio'])
    db_comments = pd.DataFrame(columns=['thread_id', 'comment_id', 'body', 'author', 'parent_comment_id', 'score', 'num_replies'])

    for submission in subreddit.top('year', limit=num_of_top_threads):
        author = submission.author
        comments = submission.comments
        created = submission.created
        id = submission.id
        name = submission.name
        num_comments = submission.num_comments
        score = submission.score
        upvote_ratio = submission.upvote_ratio

        thread = [id, name, author, comments, num_comments, created, score, upvote_ratio]
        db_threads.loc[len(db_threads)] = thread
        print("DONE with one thread")
    print("DONE with thread extraction. Moving onto individual comments")

    for index, thread in db_threads.iterrows():
        thread.comments.replace_more(limit=None)
        counter = 0
        for comment in thread['comments'].list():
            thread_id = thread.id
            comment_id = comment.id
            body = comment.body
            author = comment.author
            parent_comment_id = comment.parent_id
            score = comment.score
            num_replies = len(comment.replies.list())

            single_comment = [thread_id, comment_id, body, author, parent_comment_id, score, num_replies]
            db_comments.loc[len(db_comments)] = single_comment
            print(index, "DONE with one comment", len(db_comments))
            counter += 1
        print(index, "DONE with comment extraction for one thread")
    print("DONE with comment extraction for everything")

    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    path = os.getcwd()
    path = path.split('\\\\')
    file_path = r'\''.join(path)
    data = r'\reddit_data\\'
    filename = file_path + data + to_csv_timestamp + '_hkprotest_threads.csv'
    db_threads.to_csv(filename, index=False)
    filename = file_path + data + to_csv_timestamp + '_hkprotest_comments.csv'
    db_comments.to_csv(filename, index=False)
    print('Scraping has completed!')

if __name__ == "__main__":
    # path = os.getcwd()
    # path = path.split('\\\\')
    # file_path = r'\''.join(path)
    # data = r'\reddit_data\\'
    # pd.DataFrame(columns=["happy", "sad"]).to_csv(file_path + data + 'happy.csv', index=False)
    scrape_reddit(20)

