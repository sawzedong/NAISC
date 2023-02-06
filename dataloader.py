from PIL import Image
import random
import json
import pathlib
import os
import warnings

class ImageOnlyDataLoader:
    #used for generator
    def __init__(self, content_folder="", accepted_extensions=('.jpg', '.png', '.jpeg'),data_queue=None, random_generator=None, seed=None):
        if (data_queue is None) != (random_generator is None):
            warnings.warn('In order to resume data loading order, both "data_queue" and "random_generator" arguments must be provided', RuntimeWarning)
        
        self.path=pathlib.PurePath(content_folder)
        self.dir=[f.name for f in os.scandir(self.path) if (f.is_file() and any(f.name.endswith(ext) for ext in accepted_extensions))]
        if not self.dir:
            raise FileNotFoundError('There are no accepted image files in the data directory')
        
        if random_generator is None:
            self.rng=random.Random(seed)
        else:
            self.rng=random_generator

        if data_queue is None:
            self.data_queue=self.dir.copy()
            self.rng.shuffle(self.data_queue)
        else:
            self.data_queue=data_queue
    
    def __next__(self):
        for _ in range(2):
            while self.data_queue:
                try:
                    return Image.open(str(self.path.joinpath(self.data_queue.pop())))
                except FileNotFoundError:
                    warnings.warn('Contents of data directory was modified', RuntimeWarning)
            self.data_queue=self.dir.copy()
            self.rng.shuffle(self.data_queue)
        raise FileNotFoundError('There are no accepted image files in the data directory')
        

class ImageTextDataLoader:
    # used for discerner
    def __init__(self, JSONfilepath, skipUnratedStatements=False, data_queue=None, random_generator=None, seed=None):
        if (data_queue is None) != (random_generator is None):
            warnings.warn('In order to resume data loading order, both "data_queue" and "random_generator" arguments must be provided', RuntimeWarning)
        self.path=pathlib.PurePath(JSONfilepath).parent
        if random_generator is None:
            self.rng=random.Random(seed)
        else:
            self.rng=random_generator
        self.dataset = []
        with open(JSONfilepath, "r") as file:
            data = json.load(file)
            for data_dict in data.values():
                imgpath = pathlib.PurePath(*data_dict['path'].split('\\'))
                for insult in data_dict['insults']:
                    if "~" not in insult and not skipUnratedStatements:
                        raise ValueError("Unrated statment found, use ~ to rate statements")
                    elif "~" not in insult:
                        continue
                    statement, attitude = insult.split("~")
                    attitude = -float(attitude)
                    self.dataset.append((imgpath, statement, attitude))
                for compliment in data_dict['compliments']:
                    if "~" not in compliment and not skipUnratedStatements:
                        raise ValueError("Unrated statment found, use ~ to rate statements")
                    elif "~" not in compliment:
                        continue
                    statement, attitude = compliment.split("~")
                    attitude = float(attitude)
                    self.dataset.append((imgpath, statement, attitude))
        if data_queue is None:
            self.data_queue=self.dataset.copy()
            self.rng.shuffle(self.data_queue)
        else:
            self.data_queue=data_queue
                
    def __next__(self):
        if not self.data_queue:
            self.data_queue=self.dataset.copy()
            self.rng.shuffle(self.data_queue)
        imgpath, statement, attitude=self.data_queue.pop()
        return Image.open(str(imgpath)), statement, attitude