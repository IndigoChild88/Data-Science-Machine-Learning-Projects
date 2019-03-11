# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:30:48 2019
http://www.storybench.org/how-to-scrape-reddit-with-python/
@author: Albert Nunez
"""

#! usr/bin/env python3
import praw
import pandas as pd
import datetime as dt

#Sign in with your credentials to use api
reddit = praw.Reddit(client_id='PERSONAL_USE_SCRIPT_14_CHARS', \
                     client_secret='SECRET_KEY_27_CHARS ', \
                     user_agent='YOUR_APP_NAME', \
                     username='YOUR_REDDIT_USER_NAME', \
                     password='YOUR_REDDIT_LOGIN_PASSWORD')

subreddit = reddit.subreddit('Tinder')
top_subreddit = subreddit.top(limit=500)
print(top_subreddit)

for submission in subreddit.top(limit=2):
    #print(submission.title, submission.id)
    print(submission.title)
    
topics_dict = { "title":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "created": [], \
                "comments":[], \
                }

for submission in top_subreddit:
    topics_dict["title"].append(submission.title)
    topics_dict["score"].append(submission.score)
    topics_dict["id"].append(submission.id)
    topics_dict["url"].append(submission.url)
    topics_dict["comms_num"].append(submission.num_comments)
    topics_dict["created"].append(submission.created)
    topics_dict["comments"].append(submission.comments[0].body)
    #topics_dict["comments"].append(submission.comments)


topics_data = pd.DataFrame(topics_dict)   
print(topics_dict["comments"][:5])
print("Submission:  ")

#a comma-delimited list of comment ID36s
#Turn to CSV
topics_data.to_csv('Tinder.csv', index=False) 

#https://praw.readthedocs.io/en/v2.1.21/pages/comment_parsing.html?highlight=comments

post = reddit.submission(url='https://www.reddit.com/r/gonewild/comments/az8mzr/i_need_your_seed_deep_inside_me_now_f_24/')  # if you have the URL
#post = reddit.submission(id='8ck9mb')  # if you have the ID

# Iterate over all of the top-level comments on the post:
"""
for comment in post.comments:
    # do something, for example:
    # access the replies of a comment
    for reply in comment.replies:
        print(reply.body)
"""