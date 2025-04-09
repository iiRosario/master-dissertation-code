import numpy as np
import torch
import torch.nn as nn
import os
from env import *

class Learner:
    def __init__(self, id_, query_strat, model):
        """
        Class representing a Learner.
        
        :param id_: Unique identifier of the learner.
        :param model_path: Path to the trained YOLO model.
        """
        self.model = model
        self.x_trained_data = []              # IMGS
        self.y_trained_data = []              # ANNOTATIONS
        self.query_strat = query_strat
        self.num_fits = 0


        
    def fit(self, x, y, data, epochs, imgsz, optimizer='Adam'):
        """
        Fit the YOLO model using the current batch of data.
        Adds new data to the training set before training the model.
        
        :param x: New batch of images.
        :param y: New batch of annotations.
        """
        self.model.train(data=data, epochs=epochs, imgsz=imgsz, optimizer=optimizer)  

        # Adicionar novas imagens ao conjunto de dados já vistos
        self.x_trained_data.extend(x)  
        self.y_trained_data.extend(y) 

        filename = f"checkpoint_{self.num_fits}.pt"
        save_path = os.path.join(self.path_saved_models, filename)

        self.model.save(save_path)
        self.model = YOLO(save_path)

    def predict(self, x):
        predictions = self.model.predict(x)
        
        # Convert predictions (bounding boxes, classes, and confidences) into a simple class or probability output.
        # Here we're simplifying it to returning the object detection confidence as a "classification."
        
        # For simplicity, assume each image is either "contains object" (1) or "does not contain object" (0)
        return np.array([1 if len(pred) > 0 else 0 for pred in predictions.pred[0]])  

    def __repr__(self):
        return (
            f"Learner(\n"
            f"  id={self.id},\n"
            f"  query_strategy={self.query_strat},\n"
            f"  trained_images={len(self.x_trained_data)},\n"
            f"  trained_annotations={len(self.y_trained_data)},\n"
            f"  saved_models_path='{self.path_saved_models}',\n"
            f"  num_fits={self.num_fits}\n"
            f")"
        )



