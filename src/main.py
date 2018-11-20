import numpy as np
import corr_lda as cl
import sys
sys.path.append('../data')
from getDataInPython import getCorel5kData, read_dictionary


num_topics = 2  # num of topics

# Dirichlet priors
alpha = 1
gamma = 1
lambdaa=1
phi=1

annot_test, annot_train, caption_test, caption_train, sift_test, sift_train = getCorel5kData()

'''
To cross check dimensions of data read
print (annot_test.shape)
print (annot_train.shape)
print (caption_test.shape)
print (caption_train.shape)
print (sift_test.shape)
print (sift_train.shape)
'''

text_words = np.asarray(read_dictionary("../data/corel5k_dictionary.txt"))
sift_words=np.arange(1000)

cl.corr_lda(num_topics, sift_words, text_words, sift_train, annot_train,  alpha, phi, gamma, lambdaa)

'''
To test sample data

#words = read_dictionary("../data/corel5k_dictionary.txt")
# Words
V_W = np.array([0, 1, 2, 3, 4])
T_W = np.array([0,1,2])

# D := document words
V_D = np.array([
    [0, 0, 1, 2, 2],
    [0, 0, 1, 1, 1],
    [0, 1, 2, 2, 2],
    [4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4]
])
T_D = np.array([
    [0, 0, 2],
    [0, 0, 1],
    [0, 1, 2],
    [1, 0, 2],
    [3, 3, 4],
    [3, 4, 4]
])

cl.corr_lda(num_topics, V_W,T_W, V_D, T_D,  alpha, phi, gamma, lambdaa)
'''