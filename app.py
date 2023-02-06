import cv2
import time
from GAN.generate import Generator
from peekingduck_process.preprocess_image import Preprocessor
import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pyttsx3
import re
import threading

narrator = pyttsx3.init()
narratorthread = threading.Thread()

# AI 
if torch.cuda.is_available():  
    device = "cuda:0"
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:  
    device = "cpu" 

G = Generator(10).to(device)

# load pre-trained model
preprocess=Preprocessor()

disclaimer = "This app is powered by an artifical intelligence that generates text, be it compliments \nor insults. The developers have put it utmost effort to ensure insults do not cause excessive \nharm, but we are unable to control what text may be displayed. By continuing to use this app, \nyou acknowledge the possibility that potentially hurtful text may be generated."


# App
font = ImageFont.truetype("Arial.ttf", 15)
vid = cv2.VideoCapture(0)
cv2.namedWindow("frame")

frame = []
keep_running = True
attitude = 0
skip_disclaimer = False


# Functions 
def generateTextFromImage(image, attitudes):
    image, features=preprocess(image)
    if features:
        features=torch.tensor(features[0],dtype=torch.float).unsqueeze(0)
    else:
        return None
    toks = G.forward(features, attitudes, starting_text=["You are"], max_length=20,temperature=0.1,return_probs=True, echo_input_text=True)[0]
    final_text=G.tokens.batch_decode(toks,skip_special_tokens=True)
    return "".join(final_text)


def capture(frame):
    cv2_im_rgb = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)

    return pil_im, draw

def vidtxt(txt, wrap = True):
    if wrap:
        txt = re.sub("(.{85})", "\\1\n", txt, 0, re.DOTALL)

    bbox = draw.textbbox((10, 10), txt, font=font)
    draw.rectangle(bbox, fill="white")
    draw.text((10, 10), txt, font=font, fill="#000000")
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    

while keep_running:

    if not skip_disclaimer:
        ret, frame = vid.read()
        pil_im, draw = capture(frame)

        cv2_im_processed = vidtxt(disclaimer, wrap=False)
        cv2.imshow('frame', cv2_im_processed)

        skip_disclaimer = True
        k = cv2.waitKey(0)

    while keep_running:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)

        if k == ord('q'):
            keep_running = False
            break

        if k == ord('c'):
            attitude = 0.0
            break


    if keep_running:

        # Displaying wait screen
        while True:
            k = cv2.waitKey(1)

            if k == ord("c"):
                break
            elif attitude < 1.0 and k == ord("w"):
                attitude += 0.1
            elif attitude > -1.0 and k == ord("s"):
                attitude -= 0.1
            
            ret, frame = vid.read()
            pil_im, draw = capture(frame)

            display_text = "[Press \"c\" to confirm, \"w\" to decrease attitude and \"s\" to increase attitude]" + "\n" + "Attitude: " + ("0." if (attitude >= 0) else "-0.") + str(abs(int(attitude * 10))) + "\n\n" + ("Generating compliment..." if (attitude >= 0) else "Generating insult...")

            
            cv2_im_processed = vidtxt(display_text, wrap = False)
            cv2.imshow('frame', cv2_im_processed)


        # Displaying insult/compliment
        pil_im, draw = capture(frame)

        txt = generateTextFromImage(pil_im, torch.tensor([[attitude]]))
        print("\n", txt, "\n", sep="\n")

        cv2_im_processed = vidtxt(txt)
        cv2.imshow('frame', cv2_im_processed)
        k = cv2.waitKey(1)

        # Narrator
        narrator.say(txt)
        narrator.runAndWait()
        narrator.stop()
  
vid.release()
cv2.destroyAllWindows()
