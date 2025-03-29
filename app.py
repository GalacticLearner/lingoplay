from helpers import *

title = input("Enter book title: ")
id = book_meta(title)[0]
scenes = split_scenes(fetch_book(id))
elements = fetch_story_elements(scenes)