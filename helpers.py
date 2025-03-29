import en_core_web_sm
from gutenbergpy import textget
from numpy import array, hstack
import os
from re import findall
from tensorflow import keras

nlp = en_core_web_sm.load()

def book_meta(title):
    gutindex = open("static/gutindex.txt", encoding="utf-8", errors="ignore")
    index = gutindex.read().find(title)
    gutindex.seek(0)
    
    if title == -1:
        return None, None
    
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

def fetch_book(book_id):
    if not book_id:
        return -1
    
    raw_book = textget.get_text_by_id(book_id)
    clean_book = textget.strip_headers(raw_book).decode("utf-8")
    
    return clean_book

def split_scenes(book):
    book = nlp(book[:])

    scenes = []
    current_scene = ""
    setting = None

    for sentence in book.sents:
        next_setting = setting
        for entity in sentence.ents:
            if entity.label_ in ["GPE", "LOC"]:
                next_setting = entity.text
            
        if setting != next_setting:
            scenes.append((setting, current_scene))
            setting = next_setting
            current_scene = sentence.text
        else:
            current_scene += ' ' + sentence.text
    scenes = scenes[1:]

    return scenes

def fetch_story_elements(scenes):
    elements = []
    dialogues = []
    articles = []
    speaker = None

    for setting, scene in scenes:
        scene = nlp(scene)
        for sentence in scene.sents:
            dialogue = findall(r'“(.*?)”|"(.*?)"', sentence.text)
            dialogue = [q for pair in dialogue for q in pair if q]

            for entity in sentence.ents and entity not in dialogue:
                if entity.label_ == "PERSON":
                    speaker = entity.text
                elif entity.label_ in ["PRODUCT", "ORG", "WORK_OF_ART", "MONEY"]:
                    articles.append(entity.text)

            if speaker:
                dialogues.append((speaker, dialogue))

        elements.append([setting, set(articles), dialogues])
        articles = []
        dialogues = []
    
    return elements

def preprocess_dataset(book, scenes):
    text = [book]
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    max_length = max(len(seq) for seq in sequences)
    sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post")

    settings = [scene[0] if scene[0] else "unknown" for scene in scenes]  # Handle None values
    objects = [" ".join(scene[1]) for scene in scenes]
    dialogues = [" ".join([f"{d[0]}: {d[1]}" for d in scene[2]]) for scene in scenes]

    settings = tokenizer.texts_to_sequences(settings)
    objects = tokenizer.texts_to_sequences(objects)
    dialogues = tokenizer.texts_to_sequences(dialogues)

    settings = keras.preprocessing.sequence.pad_sequences(settings, maxlen=max_length, padding="post")
    objects = keras.preprocessing.sequence.pad_sequences(objects, maxlen=max_length, padding="post")
    dialogues = keras.preprocessing.sequence.pad_sequences(dialogues, maxlen=max_length, padding="post")

    labels = hstack([settings, objects, dialogues])

    return sequences, labels, max_length

def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=input_shape),
        keras.layers.Conv1D(64, 5, activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(book, filename):
    scenes = split_scenes(book)
    scenes = fetch_story_elements(scenes)

    X, Y, max_len = preprocess_dataset(book, scenes)
    model = create_model(max_len)
    model.fit(X, Y, epochs=10)

    model.save("static/models/" + filename + ".h5")

def scratch_from_story(scenes):
    project = {"targets": [], "monitors": [], "extensions": [], "meta": {"semver": "3.0.0", "vm": "0.2.0", "agent": "Scratch"}}
    stage = {"isStage": True,
             "name": "Stage", 
             "variables": {},
             "lists": {}, 
             "broadcasts": {},
             "blocks": {}, 
             "comments": {},
             "currentCostume": 0,
             "costumes": [{"name": scene[0], "assetId": scene[0] + ".png"} for scene in scenes],
             "sounds": [],
             "volume": 100}
    project["targets"].append(stage)

    sprites = {}
    for setting, objects, dialogues in scenes:
        for obj in objects:
            if obj not in sprites:
                image_filename = f"{obj}.png"
                image_path = f"project/{image_filename}"
                download_image(obj, image_path)

                sprites[obj] = {
                    "isStage": False,
                    "name": obj,
                    "variables": {},
                    "lists": {},
                    "broadcasts": {},
                    "blocks": {},
                    "comments": {},
                    "currentCostume": 0,
                    "costumes": [{"name": obj, "assetId": image_filename}],
                    "sounds": [],
                    "volume": 100
                }

        # Add dialogues as scripts
        for speaker, dialogue in dialogues:
            if speaker not in sprites:
                image_filename = f"{speaker}.png"
                image_path = f"scratch_project/{image_filename}"
                download_image(speaker, image_path)

                sprites[speaker] = {
                    "isStage": False,
                    "name": speaker,
                    "variables": {},
                    "lists": {},
                    "broadcasts": {},
                    "blocks": {},
                    "comments": {},
                    "currentCostume": 0,
                    "costumes": [{"name": speaker, "assetId": image_filename}],
                    "sounds": [],
                    "volume": 100
                }

            # Add "say" block for dialogue
            block_id = f"{speaker}_say_{len(sprites[speaker]['blocks'])}"
            sprites[speaker]["blocks"][block_id] = {
                "opcode": "looks_sayforsecs",
                "next": None,
                "parent": None,
                "inputs": {
                    "MESSAGE": ["text", dialogue],
                    "SECS": ["number", 2]
                }
            }

    project["targets"].extend(sprites.values())

    os.makedirs("scratch_project", exist_ok=True)

    with open("scratch_project/project.json", "w") as f:
        json.dump(project, f, indent=2)

    # Package as `.sb3` (include images)
    with zipfile.ZipFile(output_path, 'w') as sb3:
        sb3.write("scratch_project/project.json", "project.json")
        for file in os.listdir("scratch_project"):
            if file.endswith(".png"):
                sb3.write(f"scratch_project/{file}", file)

    print(f"Scratch project saved as {output_path}")