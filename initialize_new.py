import numpy as np
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt

i = 2
filename1 = "../data{}/sift_train.csv".format(i)
filename2 = "../data{}/annot_train.csv".format(i)
filename3 = "../data{}/sift_test.csv".format(i)
filename4 = "../data{}/annot_test.csv".format(i)

sift_train = np.asarray(np.loadtxt(filename1, delimiter=',', dtype=int))
annot_train = np.asarray(np.loadtxt(filename2, delimiter=',', dtype=int))
sift_test = np.asarray(np.loadtxt(filename3, delimiter=',', dtype=int))
annot_test = np.asarray(np.loadtxt(filename4, delimiter=',', dtype=int))
#initialize_coun
#t_matrices(5, 4500, 1000, 260, sift_train, annot_train)

topics = 0

topicCount = []
perplexityTrain = []
perplexityTest = []

topicNumList = [2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,\
                160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
gammaList = [0.2,0.4,0.6,0.8,1]

for gamma in gammaList:
    topicCountAlpha = []
    perplexityTrainAlpha = []
    perplexityTestAlpha = []
    for topics in topicNumList:
        
        visual_words=sift_train
        text_word = annot_train
        num_tw=457
        num_vw=len(sift_train[0])
        num_docs=len(sift_train)
        num_topics = topics
          
        #word to topic assignment
        t_word_topic = np.zeros(num_tw)
        v_word_topic = np.random.randint(num_topics,size=num_vw)
        #print "hello"
        #For each doc, number of sift features assigned per topic
        doc_topic_vw = np.zeros((num_docs, num_topics))
        
        #For each doc, number of words assigned per topic
        doc_topic_tw = np.zeros((num_docs, num_topics))
        
        #number of words per topic per word_type(beta)
        #represents how many times each word assigned to each topic
        topic_t_word = np.zeros((num_tw, num_topics))
        topic_v_word = np.zeros((num_vw, num_topics))
        #number of words per topic across all docs
        topic_tw = np.zeros((1, num_topics))
        topic_vw = np.zeros((1, num_topics))
        #mu corresponding to each topic
        #mu = np.array(0.1)*num_topics
        #sigma = np.array(0.01)*num_topics
        mu = np.full((num_topics),0.1)
        sigma= np.full((num_topics),0.01)
        topic_mu = np.random.dirichlet(mu)
        topic_sigma = np.random.dirichlet(sigma)
        #sigma corresponding to each topic
        # topic_sigma = np.zeros((num_topics,num_vw, num_vw))
        
        unN = np.random.uniform(0,num_vw,num_tw)
        
        def initialize_count_matrices(num_topics, num_docs, num_vw, num_tw,\
                                      visual_word, text_word, visual_words_list, gamma, alpha, iters):
            np.random.seed(1) 
            #initialize z
            for i in range(num_vw):
                word = i
                vw = visual_word[i]
            #    print(i)
            #   print(vw)
        
                topic = v_word_topic[i]
                topic_v_word [word, topic] +=1
                topic_vw[0, topic] +=1
                #if i==0:
                #    print(doc_topic_vw[vw,topic].shape)
                doc_topic_vw[vw, topic] += 1
        
            #initialize y for each word
            for i in range(num_tw):
                word = i #Assign ith word to ith word, not sure why need this, if for word-type, remove
                tw = text_word[i]
                #probability of topic given a document p(t|d), (1/num_vw) since y is uniform distribution
                p_topic_doc = doc_topic_vw[tw] + unN[i]
                probs = np.cumsum(p_topic_doc)
        #        print(probs)
                randt = np.random.rand() 
        #        print(randt)
                sample_locs = probs < randt  * probs[-1] 
         #       print("sample_locs shape",(sample_locs.shape))
                sample_locs = np.where(sample_locs)
         #       print(sample_locs)
                topic = ((len(sample_locs[0])) % num_topics)
        
                t_word_topic[i] = int(topic)
                #print("i= "+str(i)+", topic = "+str(topic)+", word = "+str(word))
                topic_t_word [word, topic] += 1
                topic_tw [0,topic] += 1
                doc_topic_tw [tw, topic] += 1
            
         #   log_likelihood()
        
            #def update_text_word_assignments():
                #update t_word_topic
            for q in range(iters):
                randseed=1
                np.random.seed(randseed)
                #print(t_word_topic) 
                
                wordXtopic = []
                for i in range(num_tw):
                    word = i
                    tw = text_word[i]
                    topic = int(t_word_topic[i])
                    #print("i= "+str(i)+", topic = "+str(topic)+", word = "+str(word))
                    topic_t_word[word,topic] -=1
                    topic_tw [0,topic] -=1
                    doc_topic_tw[tw,topic] -=1
        
                    #p(y_i|y,d,w)
                    p_tw_topic = (topic_t_word[word,:] + gamma)/(topic_tw + (gamma * num_tw))
                    p_topic_vw = doc_topic_vw[tw,:] + unN[i]
                    
                    p_y = p_tw_topic * np.sum(p_topic_vw)
                    #print(p_tw_topic.shape)
                    #print(p_topic_vw.shape)
                    #print(p_y.shape)
                    #p_y = p_y.transpose().ravel()
                    #p_y = (p_y) / np.sum(p_y)
                    p_y = np.squeeze(p_y, axis=0) / np.sum(p_y)
                    #print(p_y.shape)
                    #print(type(p_y))
                    wordXtopic.append(p_y)
                    vec = np.random.multinomial(1, p_y)
                    topic = np.where(vec)[0][0]
                    #print(topic)
                    topic = topic%num_topics
                    t_word_topic[i] = topic
                    topic_t_word[word, topic] +=1
                    topic_tw[0, topic] +=1
                    doc_topic_tw[tw, topic] +=1
        
            #def update_visual_word_assignment():
                #update v_word_assignment
                np.random.seed(randseed)
        
                #vw_probability = calculate_visual_word_probabily(num_topics, num_vw, t#opic_mu, topic_sigma, visual_words_list)
                siftXtopic = []
                for i in range(num_vw):
                    vw = i#visual_word[i]
                    #print("vw", vw)
                    topic = v_word_topic[i]
                    topic_v_word [vw, topic] -=1
                    topic_vw[0, topic] -=1
                    
                    doc_topic_vw[vw, topic] -=1
                    p_topic_doc = doc_topic_vw[vw,:] + alpha
                    p_topic_doc = np.array([p_topic_doc] * 1)
        
                    vw_z_count = doc_topic_vw[vw,:] + (unN[i%num_tw])
                    tw_y_count = doc_topic_tw[vw,:]
                    logp = tw_y_count * np.log((vw_z_count + 1)/(vw_z_count))
                    p_vw_topic = np.exp(logp - np.max(logp))
        
                    p_vw_topic = np.array([p_vw_topic]*1)
        
                    p_z = p_vw_topic * p_topic_doc
                    #print(p_z.shape)
                    pdf = np.squeeze(p_z) / np.sum(p_z)
                    #print(pdf)
                    siftXtopic.append(pdf)
                    
                    vec = np.random.normal(topic_mu, topic_sigma) 
                    topic = np.where(vec)[0][0] 
        
                    doc_topic_vw[vw, topic] +=1
                    topic_v_word [vw, topic] +=1
                    topic_vw[0, topic] +=1
                    v_word_topic[i] = topic
                    
                siftXword = np.matmul(siftXtopic, np.transpose(wordXtopic))
                return siftXword
                
        #def update_topics():
        #       obs1 =  
        
        def calculate_visual_word_probabily (num_topics, num_vw, topic_mu, topic_sigma, visual_words_list):
            vw_prob = np.zeros((num_vw, num_topics))
            for i in range(num_topics):
                pdf = mn.pdf(visual_words_list, topic_mu[i][0], topic_sigma[i])
                vw_prob[:,i] = pdf
        
            return vw_prob

        def perplexity(siftXword, sift, annot, siftCount):
            sigma2=0
            for i in range(len(sift)):
                #260 x 1 matrix
                output = np.matmul(np.transpose(siftXword), sift[i])
            
                #normalize output
                output = output/np.sum(output)
                #sum(log p(wm|sd))
                sigma1 = np.sum(np.log(output))
                sigma2 += sigma1
        
            #sum(sift feature matrix)
            denom = np.sum(siftCount)
            braces = np.divide(sigma2, denom)
            #braces = sigma2
            final = np.exp(-braces)
            
            return final
        
        
        
        
        visual_words = [[] for i in range(num_vw)]
        text_word = [[] for i in range(num_tw)]
        
        
        siftCountTrain = np.zeros(num_docs, dtype = int)
        
        for i in range(num_docs):
            for j in range(num_vw):
                if sift_train[i][j] != 0:
                    visual_words[j].append(i)
                    siftCountTrain[i] += 1
        
            for j in range(num_tw):
                if annot_train[i][j] !=0:
                    text_word[j].append(i)
        
        #print(visual_words)
        #print(text_word)
            
        
        
        visual_words_list = np.arange(len(sift_train[0]))
        iters = 5000
        #1000 x 260 matrix
        siftXword = initialize_count_matrices(num_topics, num_docs, num_vw, num_tw, visual_words, text_word, visual_words_list, gamma, 1, iters)
        p_train = perplexity(siftXword, sift_train, annot_train, siftCountTrain)
        
        
        num_docs = len(sift_test)
        visual_words_test = [[] for i in range(num_vw)]
        text_word_test = [[] for i in range(num_tw)]
        siftCountTest = np.zeros(num_docs, dtype = int)
        
        for i in range(num_docs):
            for j in range(num_vw):
                if sift_test[i][j] != 0:
                    visual_words_test[j].append(i)
                    siftCountTest[i] += 1
        
            for j in range(num_tw):
                if annot_test[i][j] !=0:
                    text_word_test[j].append(i)
                    
        #print "Train complete"
        
        iters=20
        siftXwordT = initialize_count_matrices(num_topics, num_docs, num_vw, num_tw, visual_words_test, text_word_test, visual_words_list, gamma, 1, iters)
        p_test = perplexity(siftXwordT, sift_test, annot_test, siftCountTest)
           
        topicCountAlpha.append(num_topics)
        perplexityTrainAlpha.append(p_train)
        perplexityTestAlpha.append(p_test)
        
    topicCount.append(topicCountAlpha)
    perplexityTrain.append(perplexityTrainAlpha)
    perplexityTest.append(perplexityTestAlpha)

plt.figure()
for i in range(len(gammaList)):
    y = perplexityTest[i]
    x = topicCount[i]
    
    
    plt.xticks(np.arange(len(x)),x)
    plt.plot(np.squeeze(y),'--o',label = "gamma = " + str(gamma))
    
plt.xticks(fontsize=12)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.title("Caption Perplexity vs. Number of Topics", fontsize=24)
plt.show()

#
print topicCount[0]
for i in range(len(gammaList)):
    
    print perplexityTest[i]






