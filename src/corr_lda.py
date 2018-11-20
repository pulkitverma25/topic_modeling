import numpy as np

def init_rvs(num_topics, num_docs, num_vw, num_tw, \
             alpha, phi, gamma, lambdaa):
    #document distribution
    theta = np.zeros(shape=[num_docs,num_topics])
    for i in range(num_docs):
        theta[i] = np.random.dirichlet(alpha*np.ones(num_topics))
    print(theta)
    
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

    #text_word-topic assignment
    y = np.zeros(shape=[num_docs,num_tw])
    for i in range(num_docs):
        for j in range(num_tw):
            y[i][j] = np.random.random_integers(low=1,high=num_vw)

    #text_word-topic assignment
    beta = np.zeros(shape=[num_topics,num_tw])
    for i in range(num_topics):
        beta[i] = np.random.dirichlet(lambdaa*np.ones(num_tw))
   
    return theta, z, mu, sigma, y 

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
    
    theta, z, mu, sigma, y = init_rvs(num_topics, num_docs, num_vw, num_tw, alpha, phi, gamma, lambdaa)
    
    print(theta)
    print(z)
    print(sigma)
    print(mu)
    print(y)

    #Call gibbs sampling here
