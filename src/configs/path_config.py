import os.path

from definitions import ROOT_DIR

DATA_PATH = os.path.join(ROOT_DIR, "./data/ASVSpoof2019")
FEATURE_PATH = os.path.join(ROOT_DIR, "./data/ASVSpoof2019/LA/features/")
PROTOCOL_PATH = os.path.join(ROOT_DIR, "./data/ASVSpoof2019/LA/ASVSpoof2019_LA_cm_protocols")
MODEL_PATH = os.path.join(ROOT_DIR, "./models/trained/oc_softmax")
