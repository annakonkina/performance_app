from wordcloud import WordCloud
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
# from nltk.corpus import stopwords

def calc_wordcloud(df, stop_words, translation=None, width=1200, height=600,
                    bg_color = 'black', color_pal = 'hot'): # translation - language
    
    # nltk.download('stopwords') #IMPORTANT TO DOWNLOAD FIRST TIME
    # nltk.download('punkt') #IMPORTANT TO DOWNLOAD FIRST TIME
    # stop_words = set(stopwords.words('english'))
    # stop_words = set(list(stop_words) +['would']) # uppend words here 
    
    df = df.drop_duplicates(['question','answer'])
    words = df['answer'].unique().tolist()
    words = pd.Series([i.lower() for i in words])

    if translation:
        words = words.apply(lambda x: GoogleTranslator(source=translation, target='english').translate(x)) # translation 

    text = ' '.join(list(words)) # joining the answers to one str
    
    for i in ['-', '  ', '’', "\'", '.', ',']: # drop extra symbols
        text = text.replace(i, '') 
        
    text = ' '.join([WordNetLemmatizer().lemmatize(b,'v') for b in text.split(' ')])

    wordcloud = WordCloud(
        width=width, height=height, 
        background_color=bg_color,
        max_words=len(text),
        # max_font_size=210, 
        relative_scaling=.01,
        collocations = False,
        colormap = color_pal,
        stopwords = stop_words).generate(text) 

    return wordcloud
