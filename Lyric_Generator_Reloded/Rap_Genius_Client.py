# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:38:59 2019

@author: acn00
"""
#Scrapping from from rap Genius totorial link;
# https://www.johnwmillr.com/scraping-genius-lyrics/

import requests
import urllib.request as urllib2
import json
import lyricsgenius as genius

# Format a request URI for the Genius API
search_term = 'Andy Shauf'
_URL_API = "https://api.genius.com/"
_URL_SEARCH = "search?q="
querystring = _URL_API + _URL_SEARCH + urllib2.quote(search_term)
request = urllib2.Request(querystring)
client_access_token = "XVpHXZg7LcV7_-BNtbYTni47TifOVIQum_mmo9TIC9a0wEh5iySjz2tluzGnzPid"
request.add_header("Authorization", "Bearer " + client_access_token)
request.add_header("User-Agent", "")


#Now that we’ve formatted the URI, we can make a request to the database.
response = urllib2.urlopen(request, timeout=3).read().decode('UTF-8')


response = json.loads(response)
json_obj = response

#We use the built-in .json() command to convert the raw response into a JSON object. 
#We can access fields in the JSON object just like we would a normal Python dictionary:
# The JSON object operates just like a normal Python dictionary

#json_obj.viewkeys()

#dict_keys([u'meta', u'response'])

# List each key contained within a single search hit
print(json_obj['response']['hits'][0]['result'].keys())


api = genius.Genius(client_access_token)
artist = api.search_artist('Andy Shauf', max_songs=3)
#Searching for Andy Shauf...

#Song 1: "Alexander All Alone"
#Song 2: "Begin Again"
#Song 3: "Comfortable With Silence"

#Reached user-specified song limit (3).
#Found 3 songs.

#Done.
print(artist)

#printing artist lyrics
print(artist.songs[0])
