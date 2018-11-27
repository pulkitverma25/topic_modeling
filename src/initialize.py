import numpy as np
from scipy.stats import multivariate_normal as mn

def initialize_count_matrices(num_topics, num_docs, num_vw, num_tw,\
                              visual_word, text_word, visual_words_list, gamma, alpha):
    #word to topic assignment
    t_word_topic = np.zeros(num_tw)
    v_word_topic = np.random.randint(num_topics,size=num_vw)

    #For each doc, number of sift features assigned per topic
    doc_topic_vw = np.zeros((num_docs, num_topics))
    
    #For each doc, number of words assigned per topic
    doc_topic_tw = np.zeros((num_docs, num_topics))

    #number of words per topic per word_type(beta)
    #represents how many times each word assigned to each topic
    topic_t_word = np.zeros((num_tw, num_topics))

    #number of words per topic across all docs
    topic_tw = np.zeros((1, num_topics))

    #mu corresponding to each topic
    topic_mu = np.zeros((num_topics, 1, num_vw))
    #sigma corresponding to each topic
    topic_sigma = np.zeros((num_topics,num_vw, num_vw))

    np.random.seed(1) 
    #initialize z
    for i in range(num_vw):
        vw = visual_word[i]
        topic = v_word_topic[i]
        doc_topic_vw[vw, topic] += 1

    #initialize y for each word
    for i in range(num_tw):
        word = i #Assign ith word to ith word, not sure why need this, if for word-type, remove
        tw = text_word[i]
        #probability of topic given a document p(t|d), (1/num_vw) since y is uniform distribution
        p_topic_doc = doc_topic_vw[tw] + (1/num_vw)
        probs = np.cumsum(p_topic_doc)
#        print(probs)
        randt = (np.random.rand() * probs[-1])
#        print(randt)
  #      sample_locs = probs < randt #(np.random.rand() * probs[-1]) 
 #       print("sample_locs shape",(sample_locs.shape))
 #       sample_locs = np.where(sample_locs)
 #       print(sample_locs)
        topic = np.random.randint(0,4)
#len(sample_locs[0]) 

        t_word_topic[i] = int(topic)
        #print("i= "+str(i)+", topic = "+str(topic)+", word = "+str(word))
        topic_t_word [word, topic] += 1
        topic_tw [0,topic] += 1
        doc_topic_tw [tw, topic] += 1
    
 #   log_likelihood()

#def update_text_word_assignments():
    #update t_word_topic

    randseed=1
    np.random.seed(randseed)
    print(t_word_topic) 
    for i in range(num_tw):
        word = i
        tw = text_word[i]
        topic = int(t_word_topic[i])
        print("i= "+str(i)+", topic = "+str(topic)+", word = "+str(word))
        topic_t_word[word,topic] -=1
        topic_tw [0,topic] -=1
        doc_topic_tw[tw,topic] -=1

        #p(y_i|y,d,w)
        p_tw_topic = (topic_t_word[word,:] + gamma)/(topic_tw + (gamma * num_tw))
        p_topic_vw = doc_topic_vw + (1/num_vw)
        
        p_y = p_tw_topic * p_tw_topic

        p_y = np.squeeze(p_y) / np.sum(p_y)
        print(p_y)
        vec = np.random.multinomial(1, p_y)
        topic = np.where(vec)[0][0]

        t_word_topic[i] = topic
        topic_t_word[word, topic] +=1
        topic_tw[0, topic] +=1
        doc_topic_tw[tw, topic] +=1

#def update_visual_word_assignment():
    #update v_word_assignment
    np.random.seed(randseed)

#    vw_probability = calculate_visual_word_probabily(num_topics, num_vw, topic_mu, topic_sigma, visual_words_list)
    for i in range(num_vw):
        vw = i#visual_word[i]
        print("vw", vw)
        topic = v_word_topic[i]

        doc_topic_vw[vw, topic] -=1
        p_topic_doc = doc_topic_vw[vw,:] + alpha
        p_topic_doc = np.array([p_topic_doc] * 1)

        vw_z_count = doc_topic_vw[vw,:] + (1/num_vw)
        tw_y_count = doc_topic_tw[vw,:]
        logp = tw_y_count * np.log(((vw_z_count + 1)/(vw_z_count+0.001))+0.0001)
        p_vw_topic = np.exp(logp - np.max(logp))

        p_vw_topic = np.array([p_vw_topic]*1)

        pdf = np.squeeze(p_vw_topic) / np.sum(p_vw_topic)
        print(pdf)
        vec = np.random.multinomial(1, pdf)  
        topic = np.where(vec)[0][0] 

        doc_topic_vw[vw, topic] +=1
        v_word_topic[i] +=1
        
#def update_topics():
#       obs1 =  

def calculate_visual_word_probabily (num_topics, num_vw, topic_mu, topic_sigma, visual_words_list):
    vw_prob = np.zeros((num_vw, num_topics))
    for i in range(num_topics):
        pdf = mn.pdf(visual_words_list, topic_mu[i][0], topic_sigma[i])
        vw_prob[:,i] = pdf

    return vw_prob
'''
def log_likelihood():
    #Probability of a sift feature belonging to a topic, given a document
    vw_t_d = doc_topic_vw + alpha
    p_vw_t_d = np.transpose(np.transpose(vw_t_d)/np.sum(vw_t_d, axis=1))

    #Probability of a word belonging to a topic, given a document 
    tw_t_d = doc_topic_tw + float(1/N) #Not sure if doc_topic_vw should come, in gclda they used that
    p_tw_t_d = np.transpose(np.transpose(tw_t_d)/np.sum(tw_t_d, axis=1))
    
    #p(t_w|t)
    tw_t = topic_t_word + gamma
    p_tw_t = tw_t / np.sum(tw_t, axis=0)
    visual_word_probs = calculate_visual_word_probabily(num_topics, num_vw, topic_mu, topic_sigma)
    
    s_ll = 0

    for i in range 
    
'''

def perplexity():
    def worddist():
        return topic_t_word / topic_tw[:, numpy.newaxis]

    temp = worddist()
    log_prop = 0
    N= 0
    for i in num_docs:
        for j in len(caption[i]):
            theta= doc_topic_tw[j] / (num_docs+alpha)
            

sift_train = np.asarray(np.loadtxt('../data/sift_train.csv', delimiter=',', dtype=int))
annot_train = np.asarray(np.loadtxt('../data/annot_train.csv', delimiter=',', dtype=int))
#initialize_coun
#t_matrices(5, 4500, 1000, 260, sift_train, annot_train)
visual_words=sift_train
text_word = annot_train
num_tw=260
num_vw=1000
num_docs=4500
num_topics=5

visual_words_list = np.arange(1000)
initialize_count_matrices(num_topics, num_docs, num_vw, num_tw, visual_words, text_word, visual_words_list, 1, 1)
