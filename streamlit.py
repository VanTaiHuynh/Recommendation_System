import streamlit as st 
import pandas as pd
import pickle
import io
import pandas as pd
from utility import *
from streamlit_utility import *
from wordcloud import WordCloud

@st.cache_data
def load_data_products():
    df_products = pd.read_csv('data/Products_ThoiTrangNam_cleaned_part1.csv')
    df_products_2 = pd.read_csv('data/Products_ThoiTrangNam_cleaned_part2.csv')
    df_products = pd.concat([df_products, df_products_2], axis=0)
    return df_products
@st.cache_data
def load_data_ratings():
    df_ratings = pd.read_csv('data/Products_ThoiTrangNam_rating_cleaned.csv')
    return df_ratings
@st.cache_resource
def load_collaborative_model():
    with open('models/CollaborativeFiltering/recommendation_CollaborativeFiltering_model_Baseline.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

df_products = load_data_products()

df_ratings = load_data_ratings()
df_users = df_ratings[['user_id','user']].drop_duplicates().set_index('user_id')
collaborativeModel = load_collaborative_model()
st.image('data/images/topic.png', caption='Shoppe')
st.write("""# Shoppe Recommendation System""")  

menu = ["Overview", "Build Project" ,"Collaborative Filtering", "ContentBase Filtering"]
choice = st.sidebar.selectbox('Danh mục', menu)
if choice == 'Overview': 
    st.write("""### Thành viên nhóm:
- Huỳnh Văn Tài  
- Trần Thế Lâm""") 
    st.write("""

        ### Tổng quan

        **Shopee** là một sàn thương mại điện tử hàng đầu tại Việt Nam, cung cấp đa dạng các sản phẩm, bao gồm:
        - **Thời trang**
        - **Mỹ phẩm**
        - **Đồ gia dụng**
        - **Điện tử**
        - **Thực phẩm**
        - **Văn phòng phẩm**
        - **Dụng cụ thể thao**
        - **Sách vở**
        - **Đồ chơi**
        - **Đồ dùng cho thú cưng**, và nhiều hơn nữa.

        Việc xây dựng hệ thống đề xuất sản phẩm giúp Shopee:
        - Cung cấp sản phẩm phù hợp với người dùng, giúp họ tìm kiếm sản phẩm một cách nhanh chóng và dễ dàng.
        - Tăng trải nghiệm mua sắm của người dùng.
        - Tăng doanh số bán hàng cho các đối tác.
        """)
    st.write("""
        ### Mục tiêu

        **Chúng tôi nhằm xây dựng hệ thống đề xuất sản phẩm cho khách hàng, với các mục tiêu cụ thể như sau:**
        - **Xây dựng hệ thống đề xuất sản phẩm:** Phát triển hệ thống để đề xuất sản phẩm phù hợp với nhu cầu và sở thích của khách hàng.
        - **Dữ liệu đầu vào đa chiều:** Sử dụng lịch sử tìm kiếm, đánh giá (ratings), và các sản phẩm đang xem để đề xuất sản phẩm.
        - **Phương pháp :** Áp dụng các phương pháp như Collaborative Filtering, Content-based Cosine Filtering, và Content-based Gensim để cải thiện độ chính xác và hiệu quả của hệ thống đề xuất.

        """)

    st.write("""
    ### Hệ thống đề xuất (Recomendation System)
    Là hệ thống được thiết kế để dự đoán sở thích của người dùng và đề xuất sản phẩm phù hợp. Điều này được thực hiện bằng cách phân tích lịch sử tìm kiếm, đánh giá (ratings), sản phẩm đang xem, và mua hàng của người dùng.
    """)
    st.image('data/images/RecommendationSystem.png', caption='Hệ thống đề xuất')

    st.write("""
    ### Collaborative Filtering
    Phương pháp này đề xuất sản phẩm dựa trên đánh giá của người dùng. Đối với người dùng mới, hệ thống sẽ dựa vào đánh giá của người dùng khác để đề xuất sản phẩm phù hợp.
    """)
    st.image('data/images/CollaborativeFiltering.png', caption='Collaborative Filtering')

    st.write("""
    ### Content-based Cosine Filtering
    Là phương pháp đề xuất sản phẩm dựa trên nội dung của chính sản phẩm đó. Hệ thống sẽ phân tích nội dung sản phẩm để đề xuất các sản phẩm tương tự cho người dùng.
    """)
    st.image('data/images/ContentBasedCosineFiltering.png', caption='Content-based Cosine Filtering')

    st.write("""
    ### Dữ liệu
    Dữ liệu của hệ thống bao gồm:
    - **Bảng Products:** Chứa thông tin về các sản phẩm.
    - **Bảng Ratings:** Chứa thông tin đánh giá của người dùng về sản phẩm.
    """)

    st.write("""
    ### Kết quả
    Hệ thống đề xuất có khả năng:
    - Đề xuất sản phẩm cho khách hàng mới.
    - Đề xuất sản phẩm dựa trên lịch sử tìm kiếm của khách hàng.
    - Đề xuất sản phẩm dựa trên đánh giá của khách hàng.
    - Đề xuất sản phẩm dựa trên sản phẩm mà khách hàng đang xem.
    """)

elif choice == 'Build Project':
    st.write("""# Xây dựng dự án""")
    st.write("""### 1. Tiền xử lý dữ liệu""")
    st.write("""### 2. Xây dựng mô hình""")
    st.write("""### 3. Đánh giá mô hình""")
    st.write("""### 4. Triển khai mô hình""")
    st.write("""### 5. Hệ thống đề xuất sản phẩm""")
    st.write("""### 6. Tài liệu""")
elif choice == 'Collaborative Filtering':
    st.write("""# Collaborative Filtering""")
    st.write("""- Collaborative Filtering: là phương pháp đề xuất sản phẩm dựa vào rating của người dùng, để đề xuất sản phẩm cho người dùng mới, hệ thống sẽ dựa vào rating của người dùng khác để đề xuất sản phẩm cho người dùng mới""")
    st.write("""### Dữ liệu: Bảng Ratings""")
    st.write("""### Mô hình: Surprise BaselineOnly""")
    st.text("""- Nếu không có chuỗi tìm kiếm user name, hệ thống sẽ chọn ngẫu nhiên 10 user name""")
    st.text("""- Nếu có chuỗi tìm kiếm, hệ thống sẽ chọn 10 user name chứa chuỗi tìm kiếm""")
    

    if 'random_username_list' not in st.session_state:
        st.session_state.random_username_list = df_users['user'].sample(10).tolist()
    user_searchStr = st.text_input('Tìm kiếm user name: ')
    if user_searchStr:
        user_searchStr = user_searchStr.lower()
        username_list = df_users[df_users['user'].str.contains(user_searchStr)]['user'].sort_values().tolist()[:10]
    else:
        username_list = st.session_state.random_username_list
    options = [''] + username_list
    selected_username = st.selectbox('Chọn user name:', options)    
    
    
    if st.button('Tìm kiếm'):
        if selected_username:
            user_id = df_users[df_users['user'] == selected_username].index[0]
            list_products = recommend_products_collaborativefiltering(user_id, df_ratings, collaborativeModel, 5)
            list_products = converID2Products(list_products, df_products)
            st.write("""### Các từ khóa quan tâm của người dùng {0}""".format(selected_username))
            #vẽ wordcloud
            text = ' '.join(text for text in list_products['all_text'])
            wordcloud = WordCloud(max_words=50, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
            st.image(wordcloud.to_image(), caption='Wordcloud')
            st.write(f"""### Đề xuất sản phẩm cho người dùng {selected_username}""")

            st.write("""### Danh sách sản phẩm đề xuất""")
            
            printRecomProductList(list_products, width_image=150)

            

    

    
elif choice == "ContentBase Filtering": 
    st.write("""# ContentBase Filtering""")
    st.write("""### Content-based Cosine Filtering: là phương pháp đề xuất sản phẩm dựa vào nội dung của sản phẩm, hệ thống sẽ phân tích nội dung sản phẩm để đề xuất sản phẩm tương tự cho người dùng""")
    st.write("""### Dữ liệu: Bảng Products""")
    st.write("""### Mô hình: Cosine Similarity kết hợp với Gensim""")
    st.write("""### Kết quả: Đề xuất sản phẩm tương tự cho người dùng""")
    