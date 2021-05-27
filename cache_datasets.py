import pickle
import numpy as np
from wiser.data.dataset_readers import NCBIDiseaseDatasetReader
from wiser.data.dataset_readers import CDRCombinedDatasetReader
from wiser.data.dataset_readers import LaptopsDatasetReader


reader = NCBIDiseaseDatasetReader()
train_data = reader.read('Path to NCBItrainset_corpus.txt')
dev_data = reader.read('Path to NCBIdevelopset_corpus.txt')
test_data = reader.read('Path to NCBItestset_corpus.txt')

with open('Path to train.pickle', 'wb') as f:
    pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
with open('Path to dev.pickle', 'wb') as f:
    pickle.dump(dev_data, f, pickle.HIGHEST_PROTOCOL)
with open('Path test.pickle', 'wb') as f:
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)



