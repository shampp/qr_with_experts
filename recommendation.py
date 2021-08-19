import random
import logging
logger = logging.getLogger(__name__)


def get_recommendation_score(ground_truth,prediction):
    pred_set = set(prediction.split())
    rewards = [len(list(set(g.split()) & pred_set))/len(set(g.split())) for g in ground_truth]
    return max(rewards)


def get_recommendations(curr_query, cand_set_sz, setting):
    import torch
    from transformers import BertTokenizer
    pretrained_models = ['GPT','XL','CTRL','BERT','BART']
    scratch_models = ['GPT', 'CTRL']
    context_q_no = len(curr_query)
    #print(context_q_no)
    cand = set()
    vocab = '../Data/semanticscholar/tokenizer/wordpiece/vocab.txt'
    tokenizer = BertTokenizer(vocab_file=vocab, unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
    #input_context = ' [sep] '.join(curr_query)
    #print(input_context)
    #mlen = len(input_context.split()) + max([len(q.split()) for q in curr_query]) + context_q_no + 4
    #input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)
    mlen = len(curr_query.split()) + max([len(q.split()) for q in curr_query]) + context_q_no + 4
    input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)

    if setting == 'scratch':
        for method in scratch_models:
            if method == 'GPT':
                gpt_sep = ' [sep] '
                special_tokens = ['[bos]']
                #cand['gpt'] = set()
                model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
                from transformers import GPT2LMHeadModel
                model = GPT2LMHeadModel.from_pretrained(model_dest)

                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                #for i in range(cand_set_sz):
                #    print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=False)))
                #print([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[1] for i in range(cand_set_sz)])
                #rmds = [tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[1] for i in range(cand_set_sz)]
                #print(cand)
                #print(rmds)
                #cand.update(rmds)
                #print(cand)
                #exit(0)
                cand.update([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[1] for i in range(cand_set_sz)])
                #print([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[1] for i in range(cand_set_sz)])

            if method == 'ctrl':
                cand['ctrl'] = set()
                model_dest ='../Data/semanticscholar/model/ctrl'
                from transformers import CTRLLMHeadModel
                model = CTRLLMHeadModel.from_pretrained(model_dest)

                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                #cand.update([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[context_q_no] for i in range(cand_set_sz)])
                cand.update([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[1] for i in range(cand_set_sz)])
                #print([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[1] for i in range(cand_set_sz)])
                
    return cand

def get_next_query(algo, setting, curr_query):
    pretrained_models = ['GPT','XL','CTRL','BERT','BART']
    scratch_models = ['GPT', 'CTRL']

    if setting == 'pretrained':
        if algo not in pretrained_models:
            logging.info("Algorithm: %s does not exist in the pretrained model card" %(algo))
            exit(0)
        if method == 'GPT':
            gpt_sep = ' [sep] '
            special_tokens = [ '[bos]' ]
    else:
        if algo not in scratch_models:
            logging.info("Algorithm: %s does not exist in the scratch model card" %(algo))
            exit(0)
        from transformers import BertTokenizer
        tokenizer = BertTokenizer(vocab_file=f'../Data/semanticscholar/tokenizer/wordpiece/vocab.txt', unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
        if method == 'GPT':
            sep = ' [sep] '
            special_tokens = ['[bos]']
            model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(model_dest)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, max_length=mlen, do_sample=False, temperature=0.4)
            next_query = tokenizer.decode(outputs[0], skip_special_tokens=False).split(sep)[1]
            for tok in special_tokens:
                next_query = next_query.replace(tok, '')
            return next_query.strip()
        if method == 'CTRL':
            sep = ' [sep] '
            special_tokens = ['[bos]']
            model_dest = '../Data/semanticscholar/model/ctrl'
            from transformers import CTRLLMHeadModel
            model = CTRLLMHeadModel.from_pretrained(model_dest)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, max_length=mlen, do_sample=False, temperature=0.4)
            next_query = tokenizer.decode(outputs[0], skip_special_tokens=False).split(sep)[1]
            for tok in special_tokens:
                next_query = next_query.replace(tok, '')
            return next_query.strip()

            
         
