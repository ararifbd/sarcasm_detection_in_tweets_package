# Import Packages
import time
start_time = time.time()
import sys, csv, os, re, nltk, string, tweet_preprocess as tp, numpy as np, pandas as pd, constants
import operator
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#after nltk installation run pre_setup.py
#https://github.com/cbaziotis/ekphrasis
#pip install ekphrasis
from ekphrasis.classes.segmenter import Segmenter
# segmenter using the word statistics from english Wikipedia
#seg_eng = Segmenter(corpus="english") 
# segmenter using the word statistics from Twitter
seg_tw = Segmenter(corpus="twitter")

# Initialize variables
IS_PYTHON3 = sys.version_info > (3, 0, 0)
sid = SentimentIntensityAnalyzer()
ps = PorterStemmer()
lemm = WordNetLemmatizer()

FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\..\\features\\features.csv"
DATASET_FILE_PATH = os.curdir + "\\..\\data\\dataset.csv"
stopwords = stopwords.words('english')

# Read Data in Dataframe
def read_data(filename):
    data = pd.read_csv(filename, header=None, encoding="utf-8", names=["Index", "Label", "Tweet"])
    #data = data[data["Index"] < 5]
#    data[(data["Index"]>0) & (data["Index"]<1000)]
#    data.loc[[0,2,4]]
    #data.loc[1:3]
    return data
'''
Stemming for studies is studi
Stemming for studying is studi
Lemma for studies is study
Lemma for studying is studying
'''
def clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=False):
    lemm = nltk.stem.wordnet.WordNetLemmatizer()
    tweet = tp.tweet_preprocess(tweet)
    tokens = nltk.word_tokenize(tweet)
    #goooooood to good, looooove to love, Moooove to move
    tokens = tp.truncate_elongated_words(tokens)
    if remove_punctuations:
        #punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        tokens = [word for word in tokens if word not in string.punctuation]
    if remove_stop_words:
        tokens = [word for word in tokens if word.lower() not in stopwords]
    if lemmatize:
        tokens = [lemm.lemmatize(word) for word in tokens]
    #print(tokens)
    return tokens

# Finds the number of Nouns and Verbs in a tweet
def POS_count(tokens):
    # tokens = clean_data(tweet, lemmatize= False)
    Tagged = nltk.pos_tag(tokens)
    #print(Tagged)
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    advebs =['RB', 'RBR', 'RBS']
    adjectives = ['JJ', 'JJR', 'JJS']
    noun_count, verb_count, adveb_count, adjective_count = 0, 0, 0, 0
    no_words = len(tokens)
    for i in range(0, len(Tagged)):
        if Tagged[i][1] in nouns:
            noun_count += 1
        if Tagged[i][1] in verbs:
            verb_count += 1
        if Tagged[i][1] in advebs:
            adveb_count += 1
        if Tagged[i][1] in adjectives:
            adjective_count += 1
    #to avoid divide by zero
    if no_words == 0:
        no_words = 1
    return round(float(noun_count) / float(no_words),2), round(float(verb_count) / float(no_words),2), round(float(adveb_count) / float(no_words),2), round(float(adjective_count) / float(no_words),2)

# Counts the punctuations in a tweet
def punctuations_counter(tweet, punctuation_list):
    punctuation_count = {}
    for p in punctuation_list:
        punctuation_count.update({p: tweet.count(p)})
    return punctuation_count

# Calculate the emoji sentiment from the emojis present in the tweet
def getEmojiSentiment(tweet, emoji_count_list = constants.popular_emoji):
    # Feature - Emoji [Compared with a list of Unicodes and common emoticons]
    emoji_sentiment = 0
    #dict(zip([3],[0]))
    #output={3: 0}
    emoji_count_dict = dict(zip(emoji_count_list, np.zeros(len(emoji_count_list))))
    for e in constants.emoji_sentiment.keys():
        if e in tweet:
            if e in emoji_count_list:
                emoji_count_dict.update({e: tweet.count(e)})
            emoji_sentiment += constants.emoji_sentiment[e]*tweet.count(e)
    if sum(emoji_count_dict.values()) > 0:
        emoji_sentiment = (float(emoji_sentiment) / float(sum(emoji_count_dict.values())))
    return emoji_sentiment, emoji_count_dict

# Counts the interjections in the tweet
def interjections_counter(tweet):
    interjection_count = 0
    #interjections = ['wow', 'haha', 'lol', 'rofl', 'lmao', 'kidding', 'wtf', 'duh', 'ha ha']
# =============================================================================
#     interjections = ['ah', 'aha', 'aww','nah','wew','yay','uh', 'awesome' , 'bah', 'bingo', 'boo', 'bravo', 'brilliant'
#                  ,'brrr', 'duck', 'excellent', 'fabulous', 'fantastic', 'fore', 'great', 'ha-ha', 'ho-ho', 'ho-ho-ho'
#                  , 'ick', 'marvelous', 'my goodness', 'nuts', 'oh', 'ouch', 'ow', 'shh', 'super', 'yippee', 'yummy', 'big deal']
# =============================================================================
    for interj in constants.interjections:
        interjection_count += tweet.lower().count(interj)
    return interjection_count

# Counts the capital words in the tweet
def captitalWords_counter(tokens):
    upperCase = 0
    for word in tokens:
        if word.isupper() and word not in constants.exclude:
            upperCase += 1
    return upperCase

# Counts the repeated letter in a word (ex. "Whaaat")
def repeatLetterWords_counter(tweet):
    repeat_letter_words = 0
    #matcher = re.compile(r'(.)\1*')
    matcher = re.compile(r'(.)\1{2,}')
    repeat_letters = [match.group() for match in matcher.finditer(tweet)]
    for segments in repeat_letters:
        if len(segments) >= 3 and str(segments).isalpha():
            repeat_letter_words += 1
    return repeat_letter_words

# Sentiment score of the tweet
def getSentimentScore(tweet):
    return round(sid.polarity_scores(tweet)['compound'], 2)

# Counts the intensifiers in a tweet
#tokens =["He", "was", "absolutely", "wrong"]
#output = (0, 1)
def intensifier_counter(tokens):
    # tweet = clean_data(tweet, lemmatize= False)
    posC, negC = 0, 0
    for index in range(len(tokens)):
# =============================================================================
#         intensifier_list = ['absolutely', 'amazingly', 'awfully', 'completely', 'considerably', 'decidedly', 'deeply', 'effing',
#                     'enormously', 'entirely', 'especially', 'exceptionally', 'extremely',
#                     'fabulously', 'flipping', 'flippin', 'fricking', 'frigging', 'friggin', 'fully', 'fucking',
#                     'greatly', 'highly', 'hugely', 'incredibly', 'intensely', 'majorly', 'more', 'most', 'particularly',
#                     'purely', 'quite', 'really', 'remarkably', 'so', 'substantially', 'thoroughly', 'totally',
#                     'tremendously', 'unbelievably', 'unusually', 'utterly', 'very', 'almost', 'barely', 'hardly',
#                     'just enough', 'kind of', 'kinda', 'less', 'little']
# =============================================================================
        if tokens[index] in constants.intensifier_list:
            #to skip last index
            if (index < len(tokens) - 1):
                ss_in = sid.polarity_scores(tokens[index + 1])
                if (ss_in["neg"] == 1.0):
                    negC += 1
                if (ss_in["pos"] == 1.0):
                    posC += 1
    return posC, negC

# Finds the most common bigrams and skipgrams(skip 1/2 grams in a tweet )

def skip_grams(tokens, n, k):
    skip_gram_value = 0
    # tokens = clean_data('if it is well hidden', lemmatize= False)
    #a=[('if', 'it'), ('it', 'is'), ('is', 'well'), ('well', 'hidden')] for k=0
    #a= [('if', 'it'), ('if', 'is'), ('it', 'is'), ('it', 'well'), ('is', 'well'), ('is', 'hidden'), ('well', 'hidden')] for k=1
    #a = [('if', 'it'), ('if', 'is'), ('if', 'well'), ('it', 'is'), ('it', 'well'), ('it', 'hidden'), ('is', 'well'), ('is', 'hidden'), ('well', 'hidden')] for k=2
    a = [x for x in nltk.skipgrams(tokens, n, k)]
    for j in range(len(a)):
        for k in range(n):
            ss = sid.polarity_scores(a[j][k])
            if (ss["pos"] == 1):
                skip_gram_value += 1
            if (ss["neg"] == 1):
                skip_gram_value -= 1
    return skip_gram_value

# Finds the polarity flip in a tweet i.e positive to negative or negative to positive change
def polarityFlip_counter(tokens):
    positive = False
    negative = False
    positive_word_count, negative_word_count, flip_count = 0, 0, 0
    for word in tokens:
        ss = sid.polarity_scores(word)
        if ss["neg"] == 1.0:
            negative = True
            negative_word_count += 1
            if positive:
                flip_count += 1
                positive = False
        elif ss["pos"] == 1.0:
            positive = True
            positive_word_count += 1
            if negative:
                flip_count += 1
                negative = False
    return positive_word_count, negative_word_count, flip_count

# Normalize every feature
def normalize( array):
    max = np.max(array)
    min = np.min(array)
    def normalize(x):
        value = 0
        if max-min:
            value = round(((x-min) / (max-min)),2)
        #all values set between 0 and 1
        return value
    if max != 0:
        #call normalize(x) for every array element
        array = [x for x in map(normalize, array)]
    return array

# =============================================================================
#My code begins here
# =============================================================================
# Calculates the polarity of the hashtag
def hashtag_sentiment(tweet):
    #hash_tag = (re.findall("#([a-zA-Z0-9]{1,25})", tweet))
    hash_tags = re.findall(r'#\w*', tweet)
    #print(hash_tags)
    hashtag_polarity = []
    for hashtag in hash_tags:
        tokens = seg_tw.segment(hashtag)
        ss = sid.polarity_scores(tokens)
        #check whether tag contain not word
        if 'not' not in tokens.split(' '):
            hashtag_polarity.append(ss['compound'])
        else:
            hashtag_polarity.append(- ss['compound'])
    sentiment = 0
    if len(hashtag_polarity) > 0:
        sentiment = round(float(sum(hashtag_polarity) / float(len(hashtag_polarity))), 2)
    return sentiment

#find the repeatition(more than two times) vowel letter in the tweet
# need to do normalize if do not use only 0 or 1 values
#change repeated_vowel_letter_words to get total repeatition for checking result change
def repeated_vowel_letter_word_counter(tweet):
    #repeated_vowel_letter_words = 0
    repeated_vowel_letter_words = 0
    vowels = ['a', 'e', 'i', 'o', 'u']
    #matcher = re.compile(r'(.)\1*')
    matcher = re.compile(r'(.)\1{2,}')
    repeated_vowel_letters = [match.group() for match in matcher.finditer(tweet)]
    #print(repeated_vowel_letters)
    for segments in repeated_vowel_letters:
        #check first letter of match whether it contains vowels or not
        if len(segments) >= 3 and str(segments)[0] in vowels:
            #repeated_vowel_letter_words += 1
            repeated_vowel_letter_words = 1
    return repeated_vowel_letter_words

#find quoted part number in a tweet
def quote_counter(tweet):
    totall_quote_counter = []
    quote_counter = 0
    #tweet may have single quote or double quote
    for quote_symbol in [r'"', r"'"]:
        for match in re.finditer(quote_symbol, tweet):
            #keep track of all qoutes
            totall_quote_counter.append(match.start())
    if totall_quote_counter:
        #for complete quote need two quotations. 
        #that is why to get quotated parts divide totall quotes number by 2.
        quote_counter = int(len(totall_quote_counter)/2)
    return quote_counter
#calculate laughters
    
def laughter_counter(tweet):
    laughter_counter = 0
    #combined word and emoji laughters
    all_laughters = constants.laughter_list + constants.laughter_emoji
    #print(all_laughters)
    for laughter in all_laughters:
        laughter_counter += tweet.lower().count(laughter)
    return laughter_counter

#count all hashtags[need normalization]
def hashtag_counter(tweet):
    hash_tags = re.findall(r'#\w*', tweet)
    return len(hash_tags)

#count slang words in tweet[need normalization]
def sarcastic_slang_counter(tweet):
    sarcastic_slang_count = 0
    for key, value in constants.sarcastic_slangs.items():
        key_count = tweet.count(key)
        value_count = tweet.count(value)
        if key_count or value_count:
            #print("key", key_count,"value",value_count)
            sarcastic_slang_count += key_count + value_count
    #print(normalize([sarcastic_slang_count]))
    return sarcastic_slang_count
#count number of words in the tweet
def tweet_word_counter(tweet):
    return len(nltk.word_tokenize(tweet))

#generate sarcastic and non sarcastic word frequency
#need to run this function if dataset is changed
def word_frequency_generator():
    data_set = read_data(DATASET_FILE_PATH)
    unigram_sarcastic_dict = {}
    unigram_non_sarcastic_dict = {}
    tweets = data_set['Tweet'].values
    labels = data_set['Label'].values
    for i, tweet in enumerate(tweets):
        tokens = clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=True)
        for word in tokens:
            if int(labels[i]) == 1:
                if word in unigram_sarcastic_dict.keys():
                    unigram_sarcastic_dict[word] += 1
                else:
                    unigram_sarcastic_dict.update({word: 1})
            if int(labels[i]) == 0:
                if word in unigram_non_sarcastic_dict.keys():
                    unigram_non_sarcastic_dict[word] += 1
                else:
                    unigram_non_sarcastic_dict.update({word: 1})

    #save word frequency in csv file
    word_frequency_table_path = os.curdir + "\\word_frequency_table.csv"
    word_frequency_table = zip(unigram_sarcastic_dict.keys(), unigram_sarcastic_dict.values(), unigram_non_sarcastic_dict.keys(), unigram_non_sarcastic_dict.values())
    headers = ["SarcasticWord","SarcasticFrequency","NonSarcasticWord","NonSarcasticFrequency"]
    with open(word_frequency_table_path, "w", newline='') as header:
        header = csv.writer(header)
        header.writerow(headers)
    # Append the feature list to the file
    with open(word_frequency_table_path, "a", newline='') as word_frequency_csv:
        writer = csv.writer(word_frequency_csv)
        for line in word_frequency_table:
            writer.writerow(line)
    return 1
    #return unigram_sarcastic_dict, unigram_non_sarcastic_dict
#read word frequency table globaly
sarcastic_word_frequency = pd.read_csv("word_frequency_table.csv", header=[0], encoding="utf-8")
#print(sarcastic_word_frequency)
#find the words which are commonly used in tweets
def find_common_sarcastic_unigrams(frequency_value = 400):
    sarcastic_unigram_list = []
    # Creat list of high frequency unigrams
    # change value > 'x' where x is the frequency threshold
    for index, row in sarcastic_word_frequency.iterrows():
        if row["SarcasticWord"] not in constants.manual_stopwords and int(row["SarcasticFrequency"]) > frequency_value:
            sarcastic_unigram_list.append(row["SarcasticWord"])
    #return sarcastic_unigram_list.remove
    # to remove '' word from list discard first item
    return sarcastic_unigram_list[1:]
#getting common sarcastic unigrams 
common_sarcastic_unigrams = find_common_sarcastic_unigrams()
#common sarcastic word counter in individual tweet
#TODO:decide preprocess steps
def common_sarcastic_word_counter(tokens):
    #find common word depending on occurring at 700 hundreads
    common_sarcastic_unigrams_count = 0
    for word in tokens:
        if word in common_sarcastic_unigrams:
            common_sarcastic_unigrams_count += 1
    return common_sarcastic_unigrams_count
#find the words which are rarely used in tweets
def find_rare_sarcastic_unigrams(frequency_value = 1):
    rare_sarcastic_unigram_list = []
    # Creat list of high frequency unigrams
    # change value > 'x' where x is the frequency threshold
    for index, row in sarcastic_word_frequency.iterrows():
        if row["SarcasticWord"] not in constants.manual_stopwords and int(row["SarcasticFrequency"]) == frequency_value:
            rare_sarcastic_unigram_list.append(row["SarcasticWord"])
    #return sarcastic_unigram_list.remove
    # to remove '' word from list discard first item
    return rare_sarcastic_unigram_list
#getting rare sarcastic unigrams
rare_sarcastic_unigrams = find_rare_sarcastic_unigrams()
def rare_sarcastic_word_counter(tokens):
    sarcastic_rare_unigrams_count = 0
    for word in tokens:
        if word in rare_sarcastic_unigrams:
            sarcastic_rare_unigrams_count += 1
    return sarcastic_rare_unigrams_count
#count quote repeatition more than twice[need normalization]
def repeated_quote_counter(tweet):
    patterns = [r"(')\1{2,}", r'(")\1{2,}']
    quote_count = 0
    for pattern in patterns:
        matcher = re.compile(pattern)
        repeated_quote_symbol =[match.group() for match in matcher.finditer(tweet)]
        quote_count += len(repeated_quote_symbol)
    return quote_count
# =============================================================================
# Negation handling starts
# =============================================================================
#get antonym for provided word
def get_antonym(word):
    antonyms = []
    similarity = []
    for syn in wn.synsets(word): 
        for lemma in syn.lemmas(): 
            if lemma.antonyms(): 
                antonyms.append(lemma.antonyms()[0].name())
    antonyms = list(set(antonyms))
    w1 = wn.synsets(word)
    if(w1):
        w1 = w1[0]
    for word in antonyms:
        w2 = wn.synsets(word)
        if(w2):
            w2 = w2[0]
            similarity.append(w1.wup_similarity(w2))
    # to remove None values in list 
    similarity = list(filter(None, similarity)) 
    if(similarity):
        index, value = max(enumerate(similarity), key=operator.itemgetter(1))
        return antonyms[index]
    return word
#search for negation word existance
def check_negation_existance(token, include_nt = True):
    if token in constants.negate_words:
        return True
    if include_nt:
        if "n't" in token:
                return True
    return False
#call this before removing punctuation. get negation until next punctuation
def get_negation_until_punctuation(tokens, add="not_"):
    punctuations = "?.,!:;"
    tokens_len = len(tokens)
    pop_indexes = []
    for index in range(tokens_len):
        word = tokens[index].lower()
        if check_negation_existance(word):
            if index+1 != tokens_len:
                for i in range(index+1,tokens_len-1):
                    #if tokens[i] in string.punctuation:
                    if tokens[i] in punctuations:
                        break
                    else:
                        #print(tokens[i])
                        if add:
                            tokens[i] = add+tokens[i]
                        else:
                            tokens[i] = get_antonym(tokens[i])
                pop_indexes.append(index)
    tokens = list(np.delete(tokens, pop_indexes))
    return tokens
#add not_ to the following word of negation part
def get_single_word_negation(tokens, add="not_"):
    tokens_len = len(tokens)
    pop_indexes = []
    for index in range(tokens_len):
        word = tokens[index].lower()
        if check_negation_existance(word):
            if index+1 != tokens_len:
                if add:
                    pop_indexes.append(index)
                    tokens[index+1] = add+tokens[index+1]
                else:
                    tokens[index] = get_antonym(tokens[index+1])
    tokens = list(np.delete(tokens, pop_indexes))
    return tokens
# =============================================================================
# Negation handling ends
# =============================================================================
# Counts the repeated upper case letters in a word (ex. "WhAAAt")
def repeat_upper_case_segment_counter(tweet):
    repeat_upper_case_segment = 0
    #matcher = re.compile(r'(.)\1*')
    matcher = re.compile(r'(.)\1{2,}')
    repeat_letters = [match.group() for match in matcher.finditer(tweet)]
    for segment in repeat_letters:
        segment_len = len(segment)
        if segment_len >= 3 and str(segment).isalpha() and segment.isupper():
            repeat_upper_case_segment += segment_len
    return repeat_upper_case_segment
# Counts the user mentions in a tweet
def user_mentions(tweet):
    return len(re.findall("@([a-zA-Z0-9]{1,15})", tweet))

def main():
    # Read data and initialize feature lists
    data_set = read_data(DATASET_FILE_PATH)
    label = list(data_set['Label'].values)
    tweets = list(data_set['Tweet'].values)
# =============================================================================
# Lexical based features
# =============================================================================
    noun_count = []
    verb_count = []
    adveb_count = []
    adjective_count = []
    positive_intensifier_count = []
    negative_intensifier_count = []
    sentimentscore = []
# =============================================================================
# Sarcastic based features
# =============================================================================
    exclamation_count = []
    questionmark_count = []
    ellipsis_count = []
    interjection_count = []
    repeatLetter_counts = []
    vowel_repetition_count =[]
    uppercase_count = []
    repeat_upper_case_segment = []
    emoji_sentiment = []
    laughter_count = []
    common_sarcastic_unigrams_count = []
    rare_sarcastic_unigrams_count = []
    sarcastic_slang_count=[]
    repeated_quote_count=[]
    hashtag_sentiment_score = []
    skip_bigrams_sentiment = []
    skip_trigrams_sentiment = []
# =============================================================================
# Contrast base features
# =============================================================================
    emoji_tweet_flip = []
    PWC_after_removing_negation_upto_next_word =[]
    NWC_after_removing_negation_upto_next_word = []
    polarity_flip_after_removing_negation_upto_next_word = []
# =============================================================================
# Context-based features
# =============================================================================
    user_mention_count = []
    hash_tag_count = []


    # process every tweet and add extracted features to appropriate list
    i=1
    for t in tweets:
        i = i+1
        print("\r Processing Tweet Number is ", i, end='', flush=True)
        tokens = clean_data(t)
        #remove sarcasm hashtag
        t = t.replace("#sarcasm",'').replace("#Sarcasm",'')
        lower_case_tweet = t.lower()
# =============================================================================
#         Lexical based features
# =============================================================================
        x = POS_count(tokens)
        noun_count.append(x[0])
        verb_count.append(x[1])
        adveb_count.append(x[2])
        adjective_count.append(x[3])
        x = intensifier_counter(tokens)
        positive_intensifier_count.append(x[0])
        negative_intensifier_count.append(x[1])
        sentimentscore.append(getSentimentScore(t))
# =============================================================================
#         #sarcastic based features
# =============================================================================
        p = punctuations_counter(t, ['!', '?', '...'])
        exclamation_count.append(p['!'])
        questionmark_count.append(p['?'])
        ellipsis_count.append(p['...'])
        interjection_count.append(interjections_counter(t))
        repeatLetter_counts.append(repeatLetterWords_counter(lower_case_tweet))
        vowel_repetition_count.append(repeated_vowel_letter_word_counter(lower_case_tweet))
        uppercase_count.append(captitalWords_counter(tokens))
        repeat_upper_case_segment.append(repeat_upper_case_segment_counter(t))
        x = getEmojiSentiment(t)
        emoji_sentiment.append(x[0])
        laughter_count.append(laughter_counter(t))
        common_sarcastic_unigrams_count.append(common_sarcastic_word_counter(tokens))
        rare_sarcastic_unigrams_count.append(rare_sarcastic_word_counter(tokens))
        sarcastic_slang_count.append(sarcastic_slang_counter(t))
        repeated_quote_count.append(repeated_quote_counter(t))
        hashtag_sentiment_score.append(hashtag_sentiment(t))
        skip_bigrams_sentiment.append(skip_grams(tokens, 2, 0))
        skip_trigrams_sentiment.append(skip_grams(tokens, 3, 0))
# =============================================================================
#         Contrast base features
# =============================================================================
        if (sentimentscore[-1] < 0 and emoji_sentiment[-1] > 0) or (sentimentscore[-1] > 0 and emoji_sentiment[-1] < 0):
            emoji_tweet_flip.append(1)
        else:
            emoji_tweet_flip.append(0)
        x = polarityFlip_counter(get_single_word_negation(clean_data(t, remove_punctuations=False), False))
        PWC_after_removing_negation_upto_next_word.append(x[0])
        NWC_after_removing_negation_upto_next_word.append(x[1])
        polarity_flip_after_removing_negation_upto_next_word.append(x[-1])
# =============================================================================
#         Context-based features
# =============================================================================
        user_mention_count.append(user_mentions(t))
        hash_tag_count.append(hashtag_counter(t))
    #print("normalize(exclamation_count)", normalize(exclamation_count))
# combine all features together to produce features file
    feature_label = zip(label,
                    #Lexical features
                    noun_count, 
                    verb_count, 
                    adveb_count, 
                    adjective_count, 
                    normalize(positive_intensifier_count), 
                    normalize(negative_intensifier_count), 
                    sentimentscore,
                    
                    #Sarcastic features
                    normalize(exclamation_count), 
                    normalize(questionmark_count), 
                    normalize(ellipsis_count), 
                    normalize(interjection_count), 
                    normalize(repeatLetter_counts), 
                    normalize(vowel_repetition_count), 
                    normalize(uppercase_count), 
                    normalize(repeat_upper_case_segment),
                    emoji_sentiment,
                    normalize(laughter_count), 
                    normalize(common_sarcastic_unigrams_count), 
                    normalize(rare_sarcastic_unigrams_count), 
                    normalize(sarcastic_slang_count),
                    normalize(repeated_quote_count), 
                    hashtag_sentiment_score, 
                    skip_bigrams_sentiment, 
                    skip_trigrams_sentiment,
                    
                    #Contrast base features
                    emoji_tweet_flip,
                    normalize(PWC_after_removing_negation_upto_next_word), 
                    normalize(NWC_after_removing_negation_upto_next_word), 
                    normalize(polarity_flip_after_removing_negation_upto_next_word),
                    
                    #Context-based features
                    normalize(user_mention_count), 
                    normalize(hash_tag_count))
    headers = ["label",
    #Lexical features
    "Noun count",
    "Verb count",
    "Adverb count",
    "Adjective count",
    "Positive intensifier",
    "Negative intensifier",
    "Sentiment score",

    #Sarcastic features
    "Exclamation",
    "Question marks",
    "Ellipsis",
    "Interjections",
    "Repeat letters",
    "Vowel repetition count",
    "Uppercase",
    "Repeat upper case segment",
    "Emoji sentiment",
    "Laughter count",
    "Common sarcastic unigram count",
    "Rare sarcastic unigram count",
    "Sarcastic slang count",
    "Repeated quote count",
    "Hashtag sentiment score",
    "Bigrams",
    "Trigrams",

    #Contrast base features
    "Emoji tweet polarity flip",
    "PWC after removing negation upto next word",
    "NWC after removing negation upto next word",
    "polarity flip after removing negation upto next word",

    #Context-based features
    "User mentions",
    "Hash tag count"
    ]
    
    # Writing headers to the new .csv file
    with open(FEATURE_LIST_CSV_FILE_PATH, "w", newline='') as header:
        header = csv.writer(header)
        header.writerow(headers)

    # Append the feature list to the file
    with open(FEATURE_LIST_CSV_FILE_PATH, "a", newline='') as feature_csv:
        writer = csv.writer(feature_csv)
        for line in feature_label:
            writer.writerow(line)


if __name__ == "__main__":
    main()
#to add new line
print()
print("Features has been created successfully.")
#calculate execution time
end_time = time.time() - start_time
total_minutes = int(end_time)/60
hours = total_minutes/60
minutes = total_minutes%60
seconds = int(end_time)%60
print("--- %d Hours %d Minutes %d Seconds ---" % (hours, minutes, seconds))