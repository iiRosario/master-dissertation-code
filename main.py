import os
import numpy as np
from entities.ActiveLearning import ActiveLearning
from entities.Annotator import Annotator
from entities.Learner import Learner
from entities.Committee import Committee
from utils.Logger import Logger
from env import *
from utils.DataManager import *

def init_run(logger, learner):
    list_annotators = []
    logger.log("INITIALIZING ANNOTATORS:")
    for i in range(NUM_ANNOTATORS):
        np.random.seed(i)          
        
        # Select unique class indices
        specialized_class = np.random.choice(NUM_CLASSES)
        annotator = Annotator(id=i, 
                              alphas=None, 
                              specialized_class=specialized_class, 
                              seed=i,
                              num_classes=NUM_CLASSES,
                              interval_normal=INTERVAL_NORMAL,
                              interval_specialized=INTERVAL_SPECIALIZED,
                              interval_na=INTERVAL_NA
                              )
        
        list_annotators.append(annotator)

    logger.log(list_annotators)

    committee = Committee(annotators=list_annotators)
    logger.log(committee)

    al = ActiveLearning(committe=committee, learner=learner,
                        limit_classes=NUM_CLASSES, num_cycles=NUM_CYCLES, seed=0)
    logger.log(str(al))
    print(str(al))

def main():
    
    logger = Logger()
    logger.log("INITIALIZING LEARNERS:")
    learner = Learner(id_=1, 
                      query_strat="uncertainty_sampling", 
                      path_init_model=INIT_MODEL_LEARNER, 
                      path_saved_models=PATH_SAVED_MODELS_LEARNER_A)
    

    logger.log(str(learner))
    init_run(logger, learner)


if __name__ == "__main__":
    main()
   