from data import get_data, get_data_source
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from pathlib import Path
from recommendation import *

def run_bandit_arms(dt):
    n_rounds = 1000
    candidate_ix = [2, 3, 5, 10]

    df, X, anchor_ids, noof_anchors = get_data(dt)
    src = get_data_source(dt)
    regret = {}
    for cand_sz in candidate_ix:
        regret[cand_sz] = {}
        log_file = Path('../Data/', src, src+'_exp3_%d.log' %(cand_sz))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm with %s selection scheme for epsilon %f with candidate size %d" %(bandit,scheme,epsilon,cand_sz))
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            logging.info("evaluating policy and calculating regret")
            regret[cand_sz][anchor] = regret_calculation(policy_evaluation(bandit, setting, X, true_ids, n_rounds,cand_sz))
            logging.info("finished with regret calculation")

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {2:'b', 3:'r', 5:'k', 10:'c'}
        regret_file = 'cand_cum_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for cand_sz in candidate_ix:
                cum_regret = [sum(x)/noof_anchors for x in zip(*regret[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'$k = {}$'.format(cand_sz))
                ax.set_xlabel(r'k')
                ax.set_ylabel(r'cumulative regret')
                ax.legend()
            fig.savefig('arm_regret.pdf',format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)


def run_bandit_round(dt):
    n_rounds = 10
    cand_set_sz = 3
    setting = 'scratch'
    experiment_bandit = list() 
    df, X, anchor_ids, noof_anchors = get_data(dt)
    if setting == 'pretrained':
        experiment_bandit = ['EXP3', 'GPT', 'XL', 'CTRL', 'BERT', 'BART']
    else:
        experiment_bandit = ['EXP3', 'GPT', 'CTRL']
    
    regret = {}
    src = get_data_source(dt)
    for bandit in experiment_bandit:
        log_file = Path('../Data/', src, 'logs',src+'_%s.log' %(bandit))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm trained from %s" %(bandit,setting))
        regret[bandit] = {}

        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            true_ids.sort() #just in case if
            regret[bandit][anchor] = regret_calculation(policy_evaluation(bandit, setting, X, true_ids, n_rounds, cand_set_sz))

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    with plt.style.context(("seaborn-darkgrid",)):
        f = plt.figure()
        f.clear()
        plt.clf()
        plt.close(f)
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col_list = ['b', 'r', 'k', 'c']
        col = {experiment_bandit[i]:col_list[i] for i in range(len(experiment_bandit))}
        sty = {'EXP3':'-', 'GPT':':', 'CTRL':'--', 'XL':'-.', 'BERT':'-', 'BART':'--'}
        labels = {'EXP3':'SS-EXP3', 'GPT':'GPT', 'CTRL':'CTRL', 'XL':'XL', 'BERT':'BERT', 'BART':'BART'}
        regret_file = 'cum_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for bandit in experiment_bandit:
                cum_regret = [sum(x)/noof_anchors for x in zip(*regret[bandit].values())]
                val = bandit+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[bandit], ls=sty[bandit], label=labels[bandit])
                ax.set_xlabel('rounds')
                ax.set_ylabel('cumulative regret')
                ax.legend()
        fig.savefig('round_regret.pdf',format='pdf')
        f = plt.figure()
        f.clear()
        plt.close(f)


def run_ctrl(setting, X, true_ids, n_rounds, cand_set_sz):
    import random
    random.seed(42)
    seq_error = np.zeros(shape=(n_rounds,1))
    for t in range(n_rounds):
        curr_id = random.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        next_query = get_next_query('CTRL', setting, curr_query)
        score = get_recommendation_score(ground_queries, next_query)
        if score >= 0.5:
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

    return seq_error


def run_gpt(setting, X, true_ids, n_rounds, cand_set_sz):
    import random
    random.seed(42)
    seq_error = np.zeros(shape=(n_rounds,1))
    for t in range(n_rounds):
        curr_id = random.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        next_query = get_next_query('GPT', setting, curr_query)
        score = get_recommendation_score(ground_queries, next_query)
        if score >= 0.5:
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

    return seq_error


def run_exp3(setting, X, true_ids, n_rounds, cand_set_sz):
    import random
    from random import choices
    random.seed(42)
    eta = 1e-3
    seq_error = np.zeros(shape=(n_rounds, 1))
    r_t = 1
    w_t = dict()
    cand = set()
    for t in range(n_rounds):
        curr_id = random.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        cand_t = get_recommendations(curr_query, cand_set_sz, setting)
        tsz = len(cand)
        cand_sz = 1 if tsz == 0 else tsz
        cand_t = cand_t.difference(cand)
        tsz = len(cand_t)
        cand_t_sz = 1 if tsz == 0 else tsz
        for q in cand_t:
            w_t[q] = eta/((1-eta)*cand_t_sz*cand_sz)
        w_k = list(w_t.keys())
        p_t = [ (1-eta)*w + eta/cand_sz for w in w_t.values() ]
        cand.update(cand_t)
        logger.info("candidate set are: {}".format(' '.join(map(str, cand))))
        ind = choices(range(len(p_t)), weights=p_t)[0]
        logger.info("getting recommendation scores")
        score = get_recommendation_score(ground_queries,w_k[ind])
        logger.info("recommendation score is: %f" %(score))
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            r_t = 0
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        r_hat = r_t/p_t[ind]
        w_t[w_k[ind]] = w_t[w_k[ind]]*np.exp(eta*r_hat)

    return seq_error


def policy_evaluation(bandit, setting, X, true_ids, n_rounds, cand_set_sz):
    if bandit == 'EXP3':
        return run_exp3(setting, X, true_ids, n_rounds, cand_set_sz)
    if bandit == 'GPT':
        return run_gpt(setting, X, true_ids, n_rounds, cand_set_sz)
    if bandit == 'CTRL':
        return run_ctrl(setting, X, true_ids, n_rounds, cand_set_sz)


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret 
