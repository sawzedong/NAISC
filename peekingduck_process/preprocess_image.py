from peekingduck.pipeline.nodes.model.mtcnn import Node
import numpy as np

class Preprocessor:
    def __init__(self,**kwargs):
        self.detector=Node(kwargs).model.detector
    def __call__(self,image):
        image=image.convert('RGB')
        image_array=np.flip(np.asarray(image),2)
        predicts=self.detector.predict_object_bbox_from_image(image_array)

        if not predicts[0].any():
            return [],[]
        bboxes=np.clip(predicts[0],0,1)
        unscaled_landmarks=np.concatenate([(yx if ind == 1 else np.flip(yx,1)) for ind,yx in enumerate(np.split(predicts[2],predicts[2].shape[1]//2,axis=1))],axis=1)
        landmark_x=unscaled_landmarks[:,0::2]
        landmark_y=unscaled_landmarks[:,1::2]
        x_bboxes=image_array.shape[1]*bboxes[:,0::2]
        y_bboxes=image_array.shape[0]*bboxes[:,1::2]
        images=[]
        landmarks=[]
        for (left,right),(up,down),land in zip(x_bboxes,y_bboxes,unscaled_landmarks):
            scale=max(right-left,down-up)
            landmark_x=(land[0::2]-left)/scale
            landmark_y=(land[1::2]-up)/scale
            landmark=np.empty(landmark_x.shape[0]+landmark_y.shape[0])
            landmark[0::2]=landmark_x
            landmark[1::2]=landmark_y
            landmarks.append(landmark)
            images.append(image.crop((left,up,right,down)))
        return images,landmarks