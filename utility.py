import pandas as pd 
import re
import regex
import pickle
from underthesea import sent_tokenize, word_tokenize, pos_tag
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
import emoji


def process_special_word(text):
    #có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()
def process_postag_thesea(text):
    new_document = []
    for sentence in sent_tokenize(text):
        # Remove periods as per original logic (consider if this is necessary for your use case)
        sentence = sentence.replace('.', '')
        # Tokenize
        tokens = word_tokenize(sentence, format="text")
        
        
        # Append the processed sentence to the new document
        new_document.append(tokens)
    
    # Join sentences and remove excess whitespace
    new_document = ' '.join(new_document)
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    
    return new_document   
def remove_stopword(text):
    # Load StopWords
    file = open('data/vietnamese-stopwords.txt','r', encoding='utf8')
    stopwords = file.read().split('\n')
    file.close()
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document
def remove_words(text):
    text = text.lower()
    remove_word_list = ['danh mục','vui lòng', 'liên hệ','địa chỉ','inbox','sản xuất','đặt hàng','tư vấn','', 'shopee','thời trang nam','thời trang', 'phong cách', 'dáng kiểu', 'kiểu dáng','kho hàng', 'mô tả', 'thông tin', 'sản phẩm', 'xuất xứ', 'chất liệu',  'gửi từ', 'thương hiệu', 'phong cách',  'mô tả ','dành cho','họa tiết','màu sắc','giới tính', 'giao hàng', 'giỏ hàng', 'shop', ]
    # Split the text into lines
    lines = text.split('\n')
    updated_lines = []
    for line in lines:
        # Remove the words from the line
        for word in remove_word_list:
            line = line.replace(word, '')
        updated_lines.append(line)
    # Join the filtered lines back into a single string with newline characters
    cleaned_text = '\n'.join(updated_lines)
    return cleaned_text
def clean_text(text):
    text = str(text).lower()
    text = remove_words(text)
    #Loại bỏ ký tự đặc biệt
    text = re.sub('[\.\:\,\-\-\-\+\d\!\%\...\.\"\*\>\<\^\&\/\[\]\(\)\=\~●•#_—]',' ',text)
    #Loại bỏ emoji
    text = emoji.replace_emoji(text)
    text = remove_words(text)
    #Loại bỏ các từ không cần thiết
    text = re.sub('\ss\s|\sm\s|\sl\s|\sxl|xxl|xxx|xxxxl|2xl|3xl|4xl|size|\smm\s|\scm\s|\sm\s|\sg\s|\skg\s',' ',text)
    #Loại bỏ khoảng trắng thừa
    text = re.sub('\s+',' ',text)
    text = process_special_word(text)
    
    text = process_postag_thesea(text)
    text = remove_stopword(text)
    return text 



def recommend_products_collaborativefiltering(user_id, data_ratings, recommendation_model, number_show_products = 5 ): 
    data = data_ratings
    list_products = [] 
    if user_id not in data['user_id'].values:
        product_ratings = data.groupby('product_id').agg(
            average_rating=('rating', 'mean'),
            number_of_ratings=('rating', 'count')
        ).reset_index()
        product_ratings = product_ratings[(product_ratings['number_of_ratings'] > 100 )& (product_ratings['average_rating'] > 4.5)].sort_values(by='average_rating', ascending=False)
        for i in range(number_show_products):
            if i < len(product_ratings):
                list_products.append(product_ratings.iloc[i]['product_id'])
    else: 
        predictions = [(product_id, recommendation_model.predict(user_id, product_id).est) for product_id in data['product_id'].unique()]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:number_show_products]
        list_products = [product_id for product_id, _ in predictions]

    return list_products

def converID2Products(list_products, data_products):
    #return dataframe 
    return data_products[data_products['product_id'].isin(list_products)]