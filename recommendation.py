import random
import logging
logger = logging.getLogger(__name__)


def get_recommendation_score(ground_truth,prediction):
    pred_set = set(prediction.split())
    rewards = [len(list(set(g.split()) & pred_set))/len(set(g.split())) for g in ground_truth]
    return max(rewards)


def get_recommendations(curr_query, cand_set_sz, setting):
    import torch
    import transformers
    transformers.logging.set_verbosity_error()
    pretrained_models = ['GPT', 'XL', 'CTRL', 'BERT', 'BART']
    scratch_models = ['GPT', 'CTRL']
    context_q_no = len(curr_query.split())
    cand = set()
    mlen = 2*context_q_no + 8

    if setting == 'scratch':
        from transformers import BertTokenizer
        vocab = '../Data/semanticscholar/tokenizer/wordpiece/vocab.txt'
        tokenizer = BertTokenizer(vocab_file=vocab, unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
        input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
        for method in scratch_models:
            if method == 'GPT':
                sep = ' [sep] '
                special_tokens = ['[sep]', '[bos]']
                model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
                logging.info("getting recommendations for %s trained from %s" %(method,setting))
                from transformers import GPT2LMHeadModel
                model = GPT2LMHeadModel.from_pretrained(model_dest)
                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                #outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, do_sample=False, temperature=0.4)
                #for i in range(cand_set_sz):
                #    print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=False)))
                rmds = [ tokenizer.decode(outputs[i], skip_special_tokens=False).split(sep)[1] for i in range(cand_set_sz) ]
                for i in range(len(rmds)):
                    for j in special_tokens:
                        rmds[i] = rmds[i].replace(j,'')
                cand.update(rmds)
                #cand.update([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[1] for i in range(cand_set_sz)])
            if method == 'CTRL':
                sep = ' [sep] '
                special_tokens = ['[sep]', '[bos]']
                #cand['ctrl'] = set()
                model_dest = '../Data/semanticscholar/model/ctrl'
                logging.info("getting recommendations for %s trained from %s" %(method,setting))
                from transformers import CTRLLMHeadModel
                model = CTRLLMHeadModel.from_pretrained(model_dest)
                #outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, do_sample=False, temperature=0.4)
                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                rmds = [ tokenizer.decode(outputs[i], skip_special_tokens=False).split(sep)[1] for i in range(cand_set_sz) ]
                for i in range(len(rmds)):
                    for j in special_tokens:
                        rmds[i] = rmds[i].replace(j,'')

                cand.update(rmds)

    if setting == 'pretrained':
        for method in pretrained_models:
            if method == 'XL':
                from transformers import (TransfoXLLMHeadModel,TransfoXLTokenizer)
                sep = ' [sep] '
                special_tokens = ['[sep]']
                tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
                model_dest = '../Data/semanticscholar/model/xl'
                model = TransfoXLLMHeadModel.from_pretrained(model_dest)

                input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                rmds = [ tokenizer.decode(outputs[i], skip_special_tokens=False).split(sep)[1] for i in range(cand_set_sz) ]
                for i in range(len(rmds)):
                    for j in special_tokens:
                        rmds[i] = rmds[i].replace(j,'')

                cand.update(rmds)
                
            if method == 'GPT':
                from transformers import (GPT2LMHeadModel,GPT2Tokenizer)
                sep = ' [sep] '
                special_tokens = ['[sep]']
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                model_dest = '../Data/semanticscholar/model/gpt2/pretrained'
                model = GPT2LMHeadModel.from_pretrained(model_dest)

                input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                for i in range(len(rmds)):
                    for j in special_tokens:
                        rmds[i] = rmds[i].replace(j,'')
                cand.update(rmds)


 
    return cand

def get_next_query(algo, setting, curr_query):
    import torch
    import transformers
    transformers.logging.set_verbosity_error()

    pretrained_models = ['GPT','XL','CTRL','BERT','BART']
    scratch_models = ['GPT', 'CTRL']
    #logger.info("running baseline query recommendation algorithms")
    context_q_no = len(curr_query.split())
    mlen = 2*context_q_no + 8
    #mlen = len(curr_query.split()) + max([len(q.split()) for q in curr_query]) + context_q_no + 4

    if setting == 'pretrained':
        if algo not in pretrained_models:
            logging.info("Algorithm: %s does not exist in the pretrained model card" %(algo))
            exit(0)
        if algo == 'GPT':
            gpt_sep = ' [sep] '
            special_tokens = [ '[bos]' ]
    else:
        if algo not in scratch_models:
            logging.info("Algorithm: %s does not exist in the scratch model card" %(algo))
            exit(0)
        from transformers import BertTokenizer
        tokenizer = BertTokenizer(vocab_file=f'../Data/semanticscholar/tokenizer/wordpiece/vocab.txt', unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
        input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
        if algo == 'GPT':
            #logging.info("Getting recommendations from baseline %s method trained from %s" %(algo,setting))
            sep = ' [sep] '
            special_tokens = ['[bos]']
            model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(model_dest)
            #outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, do_sample=False, temperature=0.4)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, max_length=mlen, do_sample=False, temperature=0.4)
            next_query = tokenizer.decode(outputs[0], skip_special_tokens=False).split(sep)[1]
            for tok in special_tokens:
                next_query = next_query.replace(tok, '')
            #print("recommended query is: %s" %(next_query))
            return next_query.strip()
        if algo == 'CTRL':
            #logging.info("Getting recommendations from baseline %s method trained from %s" %(algo,setting))
            sep = ' [sep] '
            special_tokens = ['[bos]']
            model_dest = '../Data/semanticscholar/model/ctrl'
            from transformers import CTRLLMHeadModel
            model = CTRLLMHeadModel.from_pretrained(model_dest)
            #outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, do_sample=False, temperature=0.4)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, max_length=mlen, do_sample=False, temperature=0.4)
            next_query = tokenizer.decode(outputs[0], skip_special_tokens=False).split(sep)[1]
            for tok in special_tokens:
                next_query = next_query.replace(tok, '')
            #print("recommended query is: %s" %(next_query))
            return next_query.strip()
