#https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
# Import Packages
import sys, re, nltk, string, inflect, unicodedata, contractions as con
import emoji, preprocessor as pre, common_acronym_list
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize variables
IS_PYTHON3 = sys.version_info > (3, 0, 0)
ps = PorterStemmer()
lemm = WordNetLemmatizer()
stopwords = stopwords.words('english')

########################
##Tweet Preprocessing##
#######################
negators = ['not','no','never']
RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)')

try:
    # UCS-4
    EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
except re.error:
    # UCS-2
    EMOJIS_PATTERN = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

SMILEYS_PATTERN = re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE)
NUMBERS_PATTERN = re.compile(r"(^|\s)(\-?\d+(?:\.\d)*|\d+)")
#check word existance
MIN_YEAR = 1900
MAX_YEAR = 2100
def is_year(text):
    if (len(text) == 3 or len(text) == 4) and (MIN_YEAR < len(text) < MAX_YEAR):
        return True
    else:
        return False
def lowercase(tweet):
    tweet = tweet.lower()
    return tweet
def check_word_existance(word):
    flag = False
    synonym = wn.synsets(word)
    # for abstract term hypernyms()
    if len(synonym) != 0 and len(synonym[0].hypernyms()) != 0:
        # Just the word: 
        #synonym = synonym[0].lemmas()[0].name()
        #synonym = synonym[0].name().split('.')[0]
        #print(synonym)
        flag = True
    return flag
def get_single_letter_words_pattern():
    return re.compile(r'(?<![\w\-])\w(?![\w\-])')

def get_blank_spaces_pattern():
    return re.compile(r'\s{2,}|\t')
def get_twitter_reserved_words_pattern():
    return re.compile(r'(RT|rt|FAV|fav|VIA|via)')

'''
preproccessor options
URL 	p.OPT.URL
Mention 	p.OPT.MENTION
Hashtag 	p.OPT.HASHTAG
Reserved Words 	p.OPT.RESERVED
Emoji 	p.OPT.EMOJI
Smiley 	p.OPT.SMILEY
Number 	p.OPT.NUMBER
'''
def remove_numbers(tweet, preserve_years=False):
    word_list = tweet.split(' ')
    for word in word_list:
        if word.isnumeric():
            if preserve_years:
                if not is_year(word):
                    word_list.remove(word)
            else:
                word_list.remove(word)

    tweet = ' '.join(word_list)
    return tweet
def remove_twitter_reserved_words(tweet):
    tweet = re.sub(pattern=get_twitter_reserved_words_pattern(), repl='', string=tweet)
    return tweet

def remove_single_letter_words(tweet):
    tweet = re.sub(pattern=get_single_letter_words_pattern(), repl='', string=tweet)
    return tweet
#remove url
def remove_url(tweet):
    #from tweet-preprocessor api
    return re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))', '', tweet)
    #return re.sub(r'http\S+', '', re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', tweet))
    # r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
#replace non asscii characters(Ì,Ê) into normal character
def remove_non_ascii(tweet):
    """Remove non-ASCII characters(Ì,Ê) from list of tokenized words"""
    new_tokens = []
    for token in nltk.word_tokenize(tweet):
        new_token = []
        for character in token:
            new_char = unicodedata.normalize('NFKD', character).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_token.append(new_char)
        new_tokens.append(''.join(new_token))
    return " ".join(new_tokens)
def remove_newline(tweet):
    return re.sub("\\n",'', tweet)
#update acronyms and slangs
def update_acronyms():
    for (key, value) in common_acronym_list.acronym_list.items():
        #print(key, ":",value)
        con.add(key, value)
    return True
#replace contractions and common slangs
def replace_contractions_slangs(tweet):
    return con.fix(tweet)
#Replaceing repeated character in word
# goooooood to good, looooove to love, Moooove to move
def truncate_elongated_words(tokens):
    modified_tokens = tokens
    #remove repetition morethan two times
    for index, token in enumerate(tokens):
        #print("index", index,"Token", token)
        # starting match regex
        new_token = re.sub(r'(^.)\1{2,}', r'\1\1', token)
        #print(new_token)
        if check_word_existance(new_token):
            modified_tokens[index] = new_token
            continue
        new_token = re.sub(r'(^.)\1{2,}', r'\1', token)
        if check_word_existance(new_token):
            modified_tokens[index] = new_token
            continue
        #ending match regex
        new_token = re.sub(r'(.)\1{2,}$', r'\1\1', token)
        if check_word_existance(new_token):
            modified_tokens[index] = new_token
            continue
        new_token = re.sub(r'(.)\1{2,}$', r'\1', token)
        
        if check_word_existance(new_token):
            #print(new_token)
            modified_tokens[index] = new_token
            continue
        new_token = re.sub(r'(.)\1{2,}', r'\1', token)
        if check_word_existance(new_token):
            modified_tokens[index] = new_token
            continue
        #replace more than two times repetition of a chararcter 
        #by two character such as ooooo into oo
        new_token = re.sub(r'(.)\1{2,}', r'\1\1', token)
        if check_word_existance(new_token):
            modified_tokens[index] = new_token
            continue
        tag = False
        for i in range(0,len(new_token)-1):
            if new_token[i] == new_token[i+1]:
                new_word = new_token[:i] + new_token[(i+1):]
                #print(new_word)
                if check_word_existance(new_word):
                    modified_tokens[index] = new_token
                    tag=True
                    break
        if tag:continue
        matcher = re.compile(r'(.)\1{1,}')
        #find the index of repeated character
        for match in matcher.finditer(new_token):
            '''
            print(match_index)#[(1, 3), (3, 5), (5, 7)]
            print(match_index[0][0])#1
            '''
            i = match.span()[0]
            new_word = new_token[:i] + new_token[i+1:]
            if check_word_existance(new_word):
                modified_tokens[index] = new_token
                continue
    #return ' '.join(modified_tokens)
    return modified_tokens

def remove_emoji_Smiley(tweet):
    pre.set_options(pre.OPT.SMILEY, pre.OPT.EMOJI)
    return pre.clean(tweet)
def remove_blank_spaces(tweet):
    tweet = re.sub(pattern=get_blank_spaces_pattern(), repl=' ', string=tweet)
    return tweet
#remove_punctuation either character by character or word by word
def remove_punctuation(tweet, wordbyword=False, charbychar=False):
    if wordbyword:
        return [word for word in nltk.word_tokenize(tweet) if word not in string.punctuation]
    if charbychar:
        return ''.join([char for char in tweet if char not in string.punctuation])
#remove username
def remove_username(tweet):
    return re.sub('@\w*', '', tweet) #re.sub('@[^\s]+', '', tweet)
# remove the # in #hashtag
def remove_hashtag(tweet):
    return re.sub(r'#\w*', '', tweet) #re.sub(r'#([^\s]+)', r'\1', tweet)
#replace number into string
def replace_numbers(tweet):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = [] 
    for token in nltk.word_tokenize(tweet):
        if token.isdigit():
            new_word = p.number_to_words(token)
            new_words.append(new_word)
        else:
            new_words.append(token)
    return new_words
#remove repeatative word such as articles, prepositions etc
def remove_common_words(tweet):
    # remove stopwords from word list
    return [word for word in nltk.word_tokenize(tweet) if word not in stopwords]
def lemmatize(tweet):
    return [lemm.lemmatize(word) for word in nltk.word_tokenize(tweet)]
#Replaceing repeated character in word
def replaceTwoOrMore(self, s):
        # pattern to look for three or more repetitions of any character, including newlines.
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
        return pattern.sub(r"\1\1", s)
#remove repeated punctuations
def remove_elongated_punctuations(tweet):
        return re.sub(r'([!?.])\1{1,}', r'\1', tweet)
#replace emoji into string 
def replace_emoji_into_string(tweet):
    return emoji.demojize(tweet)

'''
Stemming for studies is studi
Stemming for studying is studi
Lemma for studies is study
Lemma for studying is studying
'''
'''
def clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=False):
    #stopwords = nltk.corpus.stopwords.words('english')
    lemm = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = nltk.word_tokenize(tweet)
    if remove_punctuations:
        #punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        tokens = [word for word in tokens if word not in string.punctuation]
    if remove_stop_words:
        tokens = [word for word in tokens if word.lower() not in stopwords]
    if lemmatize:
        tokens = [lemm.lemmatize(word) for word in tokens]
    return tokens
'''
def tweet_preprocess(tweet):
    #remove numbers
    word_list = tweet.split(' ')
    for word in word_list:
        if word.isnumeric():
            word_list.remove(word)
    tweet = ' '.join(word_list)
    #remove twitter reserved words
    tweet = re.sub(pattern=get_twitter_reserved_words_pattern(), repl='', string=tweet)
    #remove single letter words
    tweet = re.sub(pattern=get_single_letter_words_pattern(), repl='', string=tweet)
    #remove url
    tweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))', '', tweet)
    #Remove non-ASCII characters(Ì,Ê) from list of tokenized words
    new_tokens = []
    for token in nltk.word_tokenize(tweet):
        new_token = []
        for character in token:
            new_char = unicodedata.normalize('NFKD', character).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_token.append(new_char)
        new_tokens.append(''.join(new_token))
    tweet = " ".join(new_tokens)
    #remove newline
    tweet = re.sub("\\n",'', tweet)
    #replace contractions and common slangs
    tweet = con.fix(tweet)
    #removing some unused words
    removing_word = ["AcakFilm","amp","USA", "'ve", "'The"]
    for word in removing_word:
        if word in tweet:
            tweet= tweet.replace(word,'')
    return tweet