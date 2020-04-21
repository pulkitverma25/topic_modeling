import matlab.engine
import array
import numpy as np
import os


def read_dictionary(file):
    data = []
    f = open(file, "r")
    for x in f:
        x = x.strip()
        data.append(x)

    return data

def getCorel5kData():
	eng = matlab.engine.start_matlab()
	path = os.path.dirname(os.path.abspath(__file__))
	eng.addpath(path,nargout=0)
	annot_test, annot_train, caption_test, caption_train, sift_test, sift_train = eng.loadcorel5k ("corel5k", nargout=6)

	annot_test = np.asarray(matlab.int32(annot_test))
	annot_train = np.asarray(matlab.int32(annot_train))
	caption_test = np.array(caption_test)
	caption_train = np.array(caption_train)
	sift_test = np.asarray(matlab.int32(sift_test))
	sift_train = np.asarray(matlab.int32(sift_train))

	return annot_test, annot_train, caption_test, caption_train, sift_test, sift_train
