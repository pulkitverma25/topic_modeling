import matlab.engine
import array
import numpy as np

def getCorel5kData():
	eng = matlab.engine.start_matlab()
	annot_test, annot_train, caption_test, caption_train, sift_test, sift_train = eng.loadcorel5k ("corel5k", nargout=6)

	annot_test = np.asarray(matlab.int32(annot_test))
	annot_train = np.asarray(matlab.int32(annot_train))
	caption_test = np.array(caption_test)
	caption_train = np.array(caption_train)
	sift_test = np.asarray(matlab.int32(sift_test))
	sift_train = np.asarray(matlab.int32(sift_train))
	return annot_test, annot_train, caption_test, caption_train, sift_test, sift_train

annot_test, annot_train, caption_test, caption_train, sift_test, sift_train = getCorel5kData()
print (annot_test.shape)
print (annot_train.shape)
print (caption_test.shape)
print (caption_train.shape)
print (sift_test.shape)
print (sift_train.shape)