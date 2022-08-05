from doctest import set_unittest_reportflags
import nltk

#Tokenizer function. You can add here different preprocesses.
def preprocess(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization, stem, etc.

    Input: Sentence in string format WRONG -> Input is the whole pandas series
    Output: Preprocessed sentence either as a list or a string WRONG -> It cannot be a list, otherwise feature extraction from CountVectorizer does not work because it .lower everything.
    '''
    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.

    # [x] Tokenize only words
    # [x] Stem the words
    # [x] Sentence splitting
    import pandas as pd
    import nltk

    # Split a sentence into multiple sentences 
    df = pd.concat([sentence, labels], axis=1).reset_index()
    df['Text'] = df['Text'].apply(lambda x : nltk.tokenize.sent_tokenize(x)) # Split the sentence into a list of subsentences
    df = df.explode('Text') # Expand the list into multiple rows while keeping the language
    df = df.dropna()
    sentence = df.loc[:, 'Text']
    labels = df.loc[:, 'language']

    # Convert al the characters to lowercase to avoid issues when counting
    sentence = sentence.str.lower() # This is not necessary since CountVectorizer already converts everything to lowercase
    sentence = sentence.apply(tokenize_sentence) # Apply the auxiliary function to actually tokenize the sentences
    
    return sentence,labels


def tokenize_sentence(sentence):
    '''
    Input:
    Output:
    '''
    import nltk
    
    if contains_cjk(sentence):
        alph_is_cjk, idxs = alphabet_is_cjk(sentence)
        if alph_is_cjk:
            final_sentence = ''
            for idx in idxs:
                final_sentence += sentence[idx] + ' '
            return final_sentence
        else:
            pass

    elif contains_latin(sentence):
        alph_is_latin, idxs = alphabet_is_latin(sentence)
        if alph_is_latin:
            final_sentence = ''
            for idx in idxs:
                final_sentence += sentence[idx]
            sentence = final_sentence
        else:
            pass
            
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\b[^\d\W]+\b') # Splits text into the multiple non-numerical words contained in it
    sentence = tokenizer.tokenize(sentence)

    porter_stem = nltk.stem.PorterStemmer() # Stems all the words using the Porter Stemmer algorithm
    sentence = ' '.join(porter_stem.stem(word) for word in sentence) # The generated list is joined again as a string as CountVectorizer cannot properly read it otherwise

    return sentence

def contains_cjk(unistr):
    '''
    Checks if a sentence contains chinese, japanese, or korean characters.
    '''
    import unicodedata as ud
    
    for uchr in unistr:
        try: ud_name = ud.name(uchr)
        except: ud_name = ''
        if any(x in ud_name for x in ['CJK', 'HANGUL']):
            return True
        else:
            continue
    return False

def contains_latin(unistr):
    '''
    Checks if a sentence contains latin characters.
    '''
    import unicodedata as ud
    
    for uchr in unistr:
        try: ud_name = ud.name(uchr)
        except: ud_name = ''
        if any(x in ud_name for x in ['LATIN']):
            return True
        else:
            continue
    return False

def alphabet_is_cjk(unistr):
    '''
    Checks if a sentence is mostly written in chinese, japanese, or korean characters.
    '''
    import unicodedata as ud
    
    cjk_chr = 0
    other_chr = 0
    idx_list = []
    for idx, uchr in enumerate(unistr):
        try: ud_name = ud.name(uchr)
        except: ud_name = ''
        if any(x in ud_name for x in ['CJK', 'HANGUL']):
            cjk_chr += 1
            idx_list.append(idx)
        else:
            other_chr += 1
            
    cjk_prop = cjk_chr/(other_chr+cjk_chr)
    if cjk_prop >= 0.5:
        return True, idx_list
    
    return False, idx_list

def alphabet_is_latin(unistr):
    '''
    Checks if a sentence is mostly written in latin characters.
    '''
    import unicodedata as ud
    
    latin_chr = 0
    other_chr = 0
    idx_list = []
    for idx, uchr in enumerate(unistr):
        try: ud_name = ud.name(uchr)
        except: ud_name = ''
        if any(x in ud_name for x in ['LATIN']):
            latin_chr += 1
            idx_list.append(idx)
        elif uchr == ' ':
            idx_list.append(idx)
        else:
            other_chr += 1
            
    latin_prop = latin_chr/(other_chr+latin_chr)
    if latin_prop >= 0.5:
        return True, idx_list
    
    return False, idx_list