import warnings

from sklearn.model_selection import train_test_split
import cntk as C
import utilities.Constants as Constants
from learning import logger


class DistributedLearner(object):

    counter = 0

    def __init__(self, strategy):
        self.strategy = strategy
        self.strategy_original = self.strategy.copy()
        self.rank = None
        self.counter = DistributedLearner.counter
        DistributedLearner.counter += 1

    def create_distributed_trainer(self):
        # create a CNTK distributed trainer
        self.strategy.init_mode()
        self.strategy.model.model._make_train_function()        
        trainer = self.strategy.model.model.train_function.trainer
        assert (trainer is not None), "Cannot find a trainer in Keras Model!"
        learner_no = len(trainer.parameter_learners)
        assert (learner_no > 0), "No learner in the trainer."
        if (learner_no > 1):
            warnings.warn("Unexpected multiple learners in a trainer.")
        learner = trainer.parameter_learners[0]
        dist_learner = C.train.data_parallel_distributed_learner(learner, num_quantization_bits=1, distributed_after=0)
        self.strategy.model.model.train_function.trainer = C.trainer.Trainer(trainer.model, [trainer.loss_function, trainer.evaluation_function], [dist_learner])

    def train_distributed(self, x_train, y_train):

        self.rank = C.Communicator.rank()
        workers = C.Communicator.num_workers()
        logger.info("Training on %i/%i", self.rank, workers)
        if (workers == 1):
            warnings.warn("Only one worker is found.")
        total_items = x_train.shape[0]
        start = self.rank * total_items // workers
        end = min((self.rank + 1) * total_items // workers, total_items)
        epochs = self.strategy.epochs_number        
        x_train, x_test, y_train, y_test = train_test_split(x_train[start: end], y_train[start: end], test_size=0.25)
        logger.info("Current train length: %i and test: %i with batch: %i and epochs: %i",
                    len(x_train),
                    len(x_test),
                    Constants.TRAINING_BATCH,
                    epochs)
        
        self.strategy.model.fit(x_train, y_train,
                            batch_size=Constants.TRAINING_BATCH,                            
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))

        C.Communicator.finalize()

