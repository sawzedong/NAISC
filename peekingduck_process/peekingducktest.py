from peekingduck.pipeline.nodes.model import jde
from PIL import Image
import numpy as np
jde_tracker=jde.Node({}).model.tracker

image=Image.open('people.jpg').convert('RGB')
image_array=np.flip(np.asarray(image),2)
#print(image)
for i in range(2):
    bboxes=np.clip(jde_tracker.track_objects_from_image(image_array)[0],0,1)
    feats={track.track_id: track.features[-1] for track in jde_tracker.tracked_stracks}

x_bboxes=np.rint(image_array.shape[1]*bboxes[:,0::2]).astype(int)
y_bboxes=np.rint(image_array.shape[0]*bboxes[:,1::2]).astype(int)
for num,((left,right),(up,down)) in enumerate(zip(x_bboxes,y_bboxes)):
    image.crop((left,up,right,down)).save(f'person_{num}.jpg')

