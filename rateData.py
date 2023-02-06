import os
import tkinter as tk
import json 
from PIL import Image, ImageTk
import time

PATH_TO_ANNOTATIONS = r"C:\Users\harry\Desktop\Computing\NAISC\annDataset\Annotations.json"

class DataToRate():

    def __init__(self):
        with open(PATH_TO_ANNOTATIONS) as jsonFile:
            self.ratingDataset = json.load(jsonFile)
            self.ratedDataKeys = []
            self.toRateKeys = []
        
        # Recording the keys of all data that has already been rated
        for data in self.ratingDataset:
            if (self.ratingDataset[data]['compliments'] != [] and self.IsRating(self.ratingDataset[data]['compliments'][0])) or (self.ratingDataset[data]['insults'] != [] and self.IsRating(self.ratingDataset[data]['insults'][0])):
                self.ratedDataKeys.append(data)
            else:
                self.toRateKeys.append(data)
        # print(self.ratedDataKeys)

    def IsRating(self, statement):
        if statement.find("~") != -1:
            return True
        else:
            return False   
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.toRateKeys) > 0: 
            item = self.toRateKeys.pop(0)
            self.ratedDataKeys.append(0)
            return item, self.ratingDataset[item]
        raise StopIteration

class rater():

    def __init__(self):
        self.data = DataToRate()
        self.dataIterator = iter(self.data)

    def closeWindow(self):
        self.window.destroy()
    
    def rate(self):
        if self.compliments:
            print("==Compliments==")
            for (i, compliment) in enumerate(self.compliments): 
                self.data['compliments'][i] += "~" + input(compliment + ": ")
        
        if self.insults:
            print("\n==Insults==")
            for (i, insult) in enumerate(self.insults):
                self.data["insults"][i] += "~" + input(insult + ": ")
        
        self.compliments = self.insults = []
        print("Rating Completed")
        

    def summonGUI(self, src_file, imgno): 
        self.window = tk.Tk()
        self.window.geometry("400x380")

        # Image 
        image = Image.open(src_file)
        imageDisplayed = ImageTk.PhotoImage(image)

        imageLabel = tk.Label(image = imageDisplayed)

        # Image Number 
        imageNoLabel = tk.Label(text = "===== image " + imgno + " =====")

        # Complete Button
        completeBtn = tk.Button(
            self.window,
            text = "Confirm",
            width = 15,
            height = 2,
            bg = "yellow",
            command = self.closeWindow
        )

        promptBtn = tk.Button(
            self.window,
            text = "Rate",
            width = 15,
            height = 2,
            bg = "green",
            command = self.rate
        )

        # Packing everything
        imageNoLabel.pack()
        imageLabel.pack()
        promptBtn.pack()
        completeBtn.pack()

        # Looping over the window 
        self.window.mainloop()
    
    def writeTOJSON(self, key, value):
        with open(PATH_TO_ANNOTATIONS, 'r') as jsonFile:
            jsonData = json.load(jsonFile)
        
        jsonData[key] = value
        with open(PATH_TO_ANNOTATIONS, 'w') as jsonFile:
            jsonObj = json.dumps(jsonData, indent = 4)
            jsonFile.write(jsonObj)

    def iterateData(self):
        while True:
            try:
                os.system('cls' if os.name == 'nt' else 'clear')
                key, self.data = next(self.dataIterator)
                imgNo = os.path.split(os.path.split(self.data["path"])[1])[0]
                imgPath = self.data["path"]
                self.insults = self.data["insults"]
                self.compliments = self.data["compliments"]
                self.summonGUI(imgPath, imgNo)
                print(self.data)
                self.writeTOJSON(key, self.data)
                time.sleep(0.5)

            except:
                print("Congratulations. All the data has been rated")
                return

if __name__ == "__main__":
    rater = rater()
    rater.iterateData()
