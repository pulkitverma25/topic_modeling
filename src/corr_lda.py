import numpy as np
from gibbs_sampling import gibbs_sampling

def init_rvs(num_topics, num_docs, num_vw, num_tw, \
             alpha, phi, gamma, lambdaa):
    #visual_word-document distribution
    theta = np.zeros(shape=[num_docs,num_topics])
    for i in range(num_docs):
        theta[i] = np.random.dirichlet(alpha*np.ones(num_topics))
    
    #visual_word-topic assignment
    z = np.zeros(shape=[num_docs, num_vw])
    for i in range(num_docs):
        for j in range(num_vw):
            z[i][j] = np.random.randint(num_topics)

    #visual_word-topic distribution
    mu = np.zeros(shape=[num_topics,num_vw])
    sigma = np.zeros(shape=[num_topics,num_vw])
    for i in range(num_topics):
        mu[i] = np.random.dirichlet(phi*np.ones(num_vw))
        sigma[i] = np.random.dirichlet(lambdaa*np.ones(num_vw))

    '''
    #text_word-document distribution
    #It should be coming from N, uniformly distributed
    n = np.zeros(shape=[num_docs,num_topics])
    for i in range(num_docs):
        for j in range(num_tw):
            y[i][j] = np.random.random_integers(low=1,high=num_vw)
    '''

    #text_word-topic assignment
    y = np.zeros(shape=[num_docs,num_tw])
    for i in range(num_docs):
        for j in range(num_tw):
            y[i][j] = np.random.random_integers(low=1,high=num_vw)

    #text_word-topic assignment
    beta = np.zeros(shape=[num_topics,num_tw])
    for i in range(num_topics):
        beta[i] = np.random.dirichlet(lambdaa*np.ones(num_tw))
   
    return theta, z, mu, sigma, y, beta

def corr_lda(num_topics, \
             visual_words, text_words, \
             visual_doc, text_doc, \
             alpha, phi, gamma, lambdaa):
    num_vw = len(visual_words)
    num_tw = len(text_words)
    num_docs = visual_doc.shape[0]

    #Sanity check for dimensions
    assert(text_doc.shape[0] == num_docs)
    assert(visual_doc.shape[1] == num_vw)
    assert(text_doc.shape[1] == num_tw)
    
    theta, z, mu, sigma, y, beta = init_rvs(num_topics, num_docs, num_vw, num_tw, alpha, phi, gamma, lambdaa)
    print("old")
    print(theta)
    print(z)
    print(sigma)
    print(mu)
    print(y)
    print(beta)

    #Call gibbs sampling here
    iter_count = 1000
    theta, mu, sigma, z = gibbs_sampling(num_topics, num_docs, num_vw, num_tw, visual_doc, text_doc, theta, z, mu, sigma, y, beta, alpha, phi, gamma, lambdaa, iter_count)
    print("updated")    
    print(theta)
    print(z)
    print(sigma)
    print(mu)

