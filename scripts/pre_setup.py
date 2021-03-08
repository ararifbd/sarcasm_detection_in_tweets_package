from nltk import download
#download pre-requisite files for nltk
def downloadDependencies():
    download('vader_lexicon')
    download('punkt')
    download('wordnet')
    download('averaged_perceptron_tagger')
    download('sentiwordnet')

if __name__ == '__main__':
    downloadDependencies()
