from gutenbergpy import textget




def fetch_book(title):
    gutindex = open("static/gutindex.txt", encoding = "utf-8", errors = "ignore")
    index = gutindex.read().find(title)
    gutindex.seek(0)
    
    if title == -1:
        return None
    
    name = ''
    consec_space = 0
    in_id = False
    
    for char in gutindex.read()[index:]:
        if char.isnumeric() and consec_space > 1:
             book_id = char
             in_id = True
             
        if char.isnumeric() and in_id and consec_space <= 1:
             book_id += char
        
        if in_id and not char.isnumeric():
             break
        
        name += char
        if char == ' ':
            consec_space += 1
        else:
             consec_space = 0

    return book_id, name.rstrip(" 0123456789")