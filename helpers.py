import en_core_web_sm
from google_images_search import GoogleImagesSearch
from gutenbergpy import textget
import json
from numpy import *
import os
from re import findall
from tensorflow import keras
import zipfile

nlp = en_core_web_sm.load()
gis = GoogleImagesSearch(os.getenv('google_api_key'), os.getenv('search_engine_cx'))

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
    # Tokenizer for input text
    text = [book]
    input_tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
    input_tokenizer.fit_on_texts(text)

    sequences = input_tokenizer.texts_to_sequences(text)
    max_length = max([len(seq) for seq in sequences])
    sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post")

    # Tokenizers for labels
    setting_tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
    object_tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
    dialogue_tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")

    # Extract labels
    settings = [scene[0] if scene[0] else "unknown" for scene in scenes]
    objects = [" ".join(scene[1]) for scene in scenes]
    dialogues = [" ".join([d[1] for d in scene[2]]) for scene in scenes]

    # Fit tokenizers on respective categories
    setting_tokenizer.fit_on_texts(settings)
    object_tokenizer.fit_on_texts(objects)
    dialogue_tokenizer.fit_on_texts(dialogues)

    # Convert text to single-token labels
    labels = []
    for scene in scenes:
        setting_label = setting_tokenizer.texts_to_sequences([scene[0]])[0][0] if scene[0] else 0
        object_label = object_tokenizer.texts_to_sequences([" ".join(scene[1])])[0][0] if scene[1] else 0
        dialogue_label = dialogue_tokenizer.texts_to_sequences([" ".join([d[1] for d in scene[2]])])[0][0] if scene[2] else 0

        labels.append([setting_label, object_label, dialogue_label])

    labels = array(labels)

    return sequences, labels, max_length, setting_tokenizer, object_tokenizer, dialogue_tokenizer, input_tokenizer

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

def train_model(book, filename, scenes):
    scenes = split_scenes(book)
    scenes = fetch_story_elements(scenes)

    X, Y, max_len, setting_tokenizer, object_tokenizer, dialogue_tokenizer, input_tokenizer = preprocess_dataset(book, scenes)
    model = create_model(max_len)
    
    model.fit(X, Y, epochs=10)
    print(model.summary())
    model.save("static/models/" + filename + ".h5")
    return setting_tokenizer, object_tokenizer, dialogue_tokenizer, input_tokenizer, model, max_len

def model_predict(model, scenes):
    return model.predict(scenes)

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
                download_image(obj)

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

        for speaker, dialogue in dialogues:
            if speaker not in sprites:
                image_filename = f"{speaker}.png"
                download_image(speaker)

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

    os.makedirs("project", exist_ok=True)

    with open("project/project.json", "w") as f:
        json.dump(project, f, indent=2)

    with zipfile.ZipFile("output.sb3", "w") as sb3:
        sb3.write("project/project.json", "project.json")
        for file in os.listdir("project"):
            if file.endswith(".png"):
                sb3.write(f"project/{file}", file)

def download_image(query):
    image_name = f"{query}.jpg"
    image_path = 'project/' + image_name
    if os.path.exists(image_path):
        return None
    
    search_params = {'q': query + "clipart", 'num': 1, 'fileType': 'png', 
                      'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'}
    gis.search(search_params=search_params, path_to_dir='static/covers/')

    for image in gis.results():
        downloaded_image_path = image.path    
        os.rename(downloaded_image_path, image_path)

def preprocess_input(new_text, input_tokenizer, max_length):
    sequence = input_tokenizer.texts_to_sequences([new_text])
    sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding="post")
    return sequence

def predict_scene(model, new_text, input_tokenizer, setting_tokenizer, object_tokenizer, dialogue_tokenizer, max_length):
    # Convert input text to tokenized sequence
    sequence = preprocess_input(new_text, input_tokenizer, max_length)

    # Get predictions
    predictions = model.predict(sequence)  # Shape: (1, 3)

    # Convert softmax outputs to integer labels (argmax)
    predicted_labels = argmax(predictions, axis=2)  # Converts probability distribution to class index
    print(predicted_labels.shape)  # Debugging step

    # Decode labels back into text
    predicted_setting = setting_tokenizer.index_word.get(predicted_labels[0], "unknown")
    predicted_object = object_tokenizer.index_word.get(predicted_labels[1], "unknown")
    predicted_dialogue = dialogue_tokenizer.index_word.get(predicted_labels[2], "unknown")

    return [predicted_setting, predicted_object, predicted_dialogue]
