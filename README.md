# S.P.I.C.E.Y. (SPecial Initiative to Comedically Entertain You) 

## Objective

An Artifical Intelligence capable of generating entertaining compliments or insults based off of one's facial features.

## Implementation

1. Dataset Extracted from `https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces`

   Data Labelling done with `rateData.py` and `generateData.py`

2. General Adversial Network GAN AI

   Generator (`GAN/generate.py`): Relies on `peekingduck_process/preprocess_image.py` which preprocesses the image using `peekingduck`, before OPT is used to generate a statement
   Discerner (`GAN/discern.py`): Uses `SGNLP` and `CLIP` to determine how 'real' a statement is compared to the image

3. Training

    The Generator and Discerner are ran in a loop to attempt to continuously optimise both the Generator and Discerner

4. App

    The app uses the webcam to capture an image which is then fed to the generator to generate the statement.

## App setup
Disclaimer: This app is powered by an artifical intelligence that generates text, be it compliments or insults. The developers have put it utmost effort to ensure insults do not cause excessive harm, but we are unable to control what text may be displayed. By continuing to use this app, you acknowledge the possibility that potentially hurtful text may be generated.

1. Ensure that you have python version 3.9 installed
2. Install the following dependencies in the given order:
    ```
    pip3.9 install peekingduck
    pip3.9 install sgnlp
    pip3.9 install git+https://github.com/openai/CLIP.git
    pip3.9 install pyttsx3
    ```
    If the installation order is followed, you may safely ignore warnings about dependency version conflicts. It is best to start with a clean version of python with no pre-installed packages.
3. Run `python3.9 app.py` and wait for the window to launch
Note: The app loads an untrained generator model due to file size limiations on Github and RAM limitations. You may find the model checkpoints uploaded on Google Drive here: [https://drive.google.com/drive/folders/1F5Znp-fasrPhR_1y-CXTME8nyVdNEob5?usp=share_link](https://drive.google.com/drive/folders/1F5Znp-fasrPhR_1y-CXTME8nyVdNEob5?usp=share_link)

## Using the app
1. When the app is first launched, a disclaimer is shown. Press any key to dismiss the disclaimer.
2. Smile, wave, or simply face the camera, and when you're ready, press `c` to generate a compliment, or `i` to generate an insult.
3. After it has finished generating, it will read out the statement and display it on the screen (remember to turn up your volume!)
4. You may continuously use the app, or press `q` to quit.

Warning: It is a known issue that the app instantly closes after generating one statement on macOS. Please run this code on a Windows system instead.
