import numpy as np

def gibbs_sampling(num_topics, num_docs, num_vw, num_tw, \
                   visual_doc, text_doc, \
                   theta, z, mu, sigma, \
                   y, beta, \
                   alpha, phi, gamma, lambdaa,\
                   iter_count):

    for it in range(iter_count):
	    for i in range(num_docs):
	        for j in range(num_vw):
	            p_bar_ij = np.exp(np.log(theta[i]) + np.log(mu[:, visual_doc[i][j]]) + np.log(sigma[:, visual_doc[i][j]]))
	            p_ij = p_bar_ij / np.sum(p_bar_ij)

	            z_ij = np.random.multinomial(1, p_ij)
	            z[i][j] = np.argmax(z_ij)

	    for i in range(num_docs):
	        suff_stat1 = np.zeros(num_topics)

	        for j in range(num_topics):
	            suff_stat1[j] = np.sum(z[i] == j)

	        theta[i, :] = np.random.dirichlet(alpha + suff_stat1)

	    for i in range(num_topics):
	        suff_stat2 = np.zeros(num_vw)

	        for j in range(num_vw):
	            for k in range(num_docs):
	                for l in range(num_vw):
	                    suff_stat2[j] += (visual_doc[k][l] == j) and (z[k][l] == i)

	        mu[i, :] = np.random.dirichlet(phi + suff_stat2)
            sigma[i,:] = np.random.dirichlet(lambdaa + suff_stat2)

    return theta, mu, sigma, z

'''
    #Now we need to sample text words
    for it in range(iter_count):
        for i in range(num_docs):
            for j in range(num_tw):
                p_bar_ij = np.exp(np.log())

'''
