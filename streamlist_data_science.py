import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utility import *
from joblib import load
import io

@st.cache_data
def load_gensim_model(num_parts, prefix='models/project2/surprise/recommendation_CollaborativeFiltering_model_part_'):
    full_model_bytes = b''

    # Concatenate each part
    for i in range(num_parts):
        with open(f'{prefix}{i}.joblib', 'rb') as f:
            full_model_bytes += f.read()

    # Load the model directly from the bytes in memory
    model = load(io.BytesIO(full_model_bytes))
    return model

@st.cache_data
def load_data_products():
    df_products = pd.read_csv('data/project2/Products_ThoiTrangNam_cleaned_part1.csv')
    df_products_2 = pd.read_csv('data/project2/Products_ThoiTrangNam_cleaned_part2.csv')
    df_products = pd.concat([df_products, df_products_2], axis=0)
    return df_products
@st.cache_data
def load_data_ratings():
    df_ratings = pd.read_csv('data/project2/Products_ThoiTrangNam_rating_cleaned.csv')
    return df_ratings

surprise_model = load_gensim_model(6)
df_products = load_data_products()
df_ratings = load_data_ratings()    
st.title("Đồ Án Tốt Nghiệp Data Science - Machine Learning")
st.write("""### Thành viên nhóm:
- Huỳnh Văn Tài
- Trần Thế Lâm""") 
menu = ["Home", "Build Project" ,"Recommendation System Prediction"]
choice = st.sidebar.selectbox('Danh mục', menu)
if choice == 'Home': 
    st.write("""# Đề tài: Xây dựng hệ thống đề xuất sản phẩm cho khách hàng cho sàn thương mại điện tử Shoppe""")   
    st.write("""### Mục tiêu:
    - Xây dựng hệ thống đề xuất sản phẩm cho khách hàng
    - Dựa vào lịch sử tìm kiếm, rating, sản phẩm đang xem để đề xuất sản phẩm
    - Sử dụng các phương pháp: Collaborative Filtering, Content-based Cosine Filtering, Content-based Gensim""")
    st.write("""### Dữ liệu:
    - Dữ liệu gồm 2 bảng: 
        - Bảng Products: chứa thông tin sản phẩm
        - Bảng Ratings: chứa thông tin rating của người dùng đối với sản phẩm""")
    st.write("""### Công nghệ:
    - Ngôn ngữ lập trình: Python
    - Thư viện: Pandas, Numpy, Matplotlib, Seaborn, Gensim, Scikit-learn, Streamlit""")
    st.write("""### Kết quả:
    - Đề xuất sản phẩm cho khách hàng vãng lai mới
    - Đề xuất sản phẩm cho khách hàng dựa vào lịch sử tìm kiếm
    - Đề xuất sản phẩm cho khách hàng dựa vào rating
    - Đề xuất sản phẩm cho khách hàng dựa vào sản phẩm đang xem""")
elif choice == 'Build Project':
    st.write("""# Build Project""")
    st.write("""### 1. Load dữ liệu""")
    #st.image('data/project2/image/load_data.png', use_column_width=True)
elif choice == 'Recommendation System Prediction':
    st.write("""# Recommendation System""")
    #
    # df_users = df_ratings[['user_id','user']].drop_duplicates().set_index('user_id')
    # st.write('### Dữ liệu Ratings demo')
    # st.write(df_ratings.sample(5))
    # st.write('### Dữ liệu Products demo')
    # st.write(df_products.sample(5))
    # st.write('### Dữ liệu Users demo')
    # st.write(df_users.sample(5))
    #
    # st.write('### 1.Đề xuất cho khách hàng vãng lai mới')
    # st.write('##### Đưa ra 10 đề xuất cho với các sản phẩm có số lượng rating nhiều nhất và trung bình rating trên 4.5')
    # list_products1 = recommend_products(df_ratings, df_products,surprise_model, number_of_recommen=5).sort_values(by="price").set_index('product_id')
    # st.write(list_products1)
    #
    # st.write('### 2. Đề xuất tìm kiếm sản phẩm cho khách hàng')
    # input2 = st.text_input('Từ khóa tìm kiếm: ')
    # button2_timkiem =st.button('Tìm kiếm')
    # if button2_timkiem:
    #     list_products2 = recommend_products(df_ratings, df_products,surprise_model, str_search=input2, number_of_recommen=5).set_index('product_id')
    #     st.write(list_products2)
    #
    # st.write('### 3. Gọi ý cho khách hàng có lịch sử tìm kiếm')
    # st.session_state.user_history = ['Bộ nỉ nam năng động tay dài', 'Bộ nỉ dày nam nữ mặc siêu ấm, set nỉ nam có mũ, quần áo thể thao thu đông cực ấm', ]
    #
    # input3 = st.text_input('Thêm lịch sử tìm kiếm: ')
    # button3 = st.button('Thêm')
    # if button3:
    #     st.session_state.user_history.append(input3)
    #     st.session_state.user_history = st.session_state.user_history[-5:]
    #     search_string =  ', '.join(st.session_state.user_history)
    #     st.write('Lịch sử tìm kiếm của khách hàng:')
    #     st.write(search_string)
    #     list_products3 = recommend_products(df_ratings, df_products,surprise_model, str_search=search_string, number_of_recommen=5).set_index('product_id')
    #     st.write(list_products3)
    #
    #
    # st.write('### 4. Đề xuất cho khách hàng dựa vào rating')
    # input4 = st.text_input('Nhập Mã số khách hàng: ')
    # button4 = st.button('Random user')
    #
    # if input4:
    #     input4 = int(input4)
    #     st.write('Đề xuất sản phẩm cho khách hàng id: ', input4)
    #     list_products4 = recommend_products(df_ratings, df_products,surprise_model, user_id=input4, number_of_recommen=5).set_index('product_id')
    #     st.write(list_products4)
    # if button4:
    #     user_id4 = df_ratings.sample(1)['user_id'].values[0]
    #     st.write('Đề xuất sản phẩm cho khách hàng id: ', user_id4)
    #     list_products4 = recommend_products(df_ratings, df_products,surprise_model, user_id=user_id4, number_of_recommen=5).set_index('product_id')
    #     st.write(list_products4)
    #
    #
    # st.write('### 5. Đề xuất cho khách hàng dựa vào sản phẩm đang xem')
    # product_id5 = st.text_input('Nhập mã sản phẩm: ', key='product_id')
    # button5 = st.button('Random product')
    #
    #
    # if product_id5:
    #     product_id5 = int(product_id5)
    #     product_str = df_products[df_products['product_id'] == product_id5]['product_name'].values[0] + ' ' + df_products[df_products['product_id'] == product_id5]['description'].values[0]
    #     st.write('Sản phẩm đang xem: ', df_products[df_products['product_id'] == product_id5]['product_name'].values[0])
    #
    #     st.write("Sản phẩm tương tự")
    #     list_products5 = recommend_products(df_ratings, df_products,surprise_model, user_id=None, str_search=product_str, number_of_recommen=5)
    #     st.write(list_products5)
    # if button5:
    #     product_id5 = df_products.sample(1)['product_id'].values[0]
    #     st.write('Sản phẩm đang xem: ', df_products[df_products['product_id'] == product_id5]['product_name'].values[0])
    #     product_str = df_products[df_products['product_id'] == product_id5]['product_name'].values[0] + ' ' + df_products[df_products['product_id'] == product_id5]['description'].values[0]
    #     st.write("Sản phẩm tương tự")
    #     list_products5 = recommend_products(df_ratings, df_products,surprise_model, user_id=None, str_search=product_str, number_of_recommen=5)
    #     st.write(list_products5)
    #
    #
    # # st.write("##### Đề xuất sản phẩm cho khách hàng vãng lai - chưa có lịch sử tìm kiếm")
    # # list_products = recommend_products_collaborativefiltering(0, df_ratings, df_products, n=5)
    # # st.write(list_products)
    # # st.write("##### Đề xuất sản phẩm cho khách hàng vãng lai - có lịch sử tìm kiếm")
    # # st.session_state.user_history.append("Áo thun nam ")
    # # st.session_state.user_history.append("Quần jean nam ")
    # # st.session_state.user_history.append("Áo khoác nam ")
    # # st.session_state.user_history.append("Áo len nữ ")
    # # st.session_state.user_history.append("Quần kaki nam ")
    # # st.write("Lịch sử tìm kiếm của khách hàng: ", st.session_state.user_history)
    # # ## tăng trọng số của các sản phẩm tìm kiếm gần đây
    # # search_string = ''
    # # for i in range(len(st.session_state.user_history)):
    # #     search_string += ' ' +  (st.session_state.user_history[i]+' ') * (i+1)
    # # st.write("Lịch sử tìm kiếm của khách hàng: ", search_string)
    # # list_products = recommendation_gensim(search_string, df_products, 10)
    # # st.write(list_products)
    #
    #
    #
    #
    # # st.write("##### Đề xuất sản phẩm cho khách hàng dựa trên Mã số khách hàng")
    # # text = st.text_input("Nhập Mã số khách hàng: ")
    # # if text:
    # #     user_id = int(text)
    # #     if user_id not in df_users.index:
    # #         st.write("Không tìm thấy Mã số khách hàng - Hiển thị sản phẩm dựa trên số lượng rating và trung bình rating")
    # #     list_products = recommend_products_collaborativefiltering(user_id, df_ratings, df_products, n=5)
    # #     st.write(list_products)
    #
    # # st.write("##### Đề xuất sản phẩm cho khách hàng vãn lai")
    #
    # # input = st.text_input("Nhập thông tin tìm kiếm: ")
    # # # gợi ý sản phẩm dựa trên thông tin tìm kiếm
    # # if input:
    # #     # Chỉ lưu 5 thông tin tìm kiếm gần nhất
    # #     st.session_state.user_history.append(input)
    # #     if len(st.session_state.user_history) > 5:
    # #         st.session_state.user_history = st.session_state.user_history[-5:]
    # #     st.write("Lịch sử tìm kiếm của khách hàng: ", input)
    # #     list_products = recommendation_cosin(input, df_products, number_of_similar_product = 5)
    # #     st.write(list_products)
    #
    #
    #
    #
    #
    
    
    
    # data = {
    #     'KH001A': 'Nguyễn Văn A',
    #     'KH002B': 'Trần Thị B',
    #     'KH003C': 'Phạm Văn C',
    #     'KH004D': 'Lê Thị D',
    #     'KH005E': 'Hoàng Văn E',
    #     'KH006F': 'Nguyễn Thị F',
    #     'KH007G': 'Trần Văn G',
    #     'KH008H': 'Phạm Thị H',
    #     'KH009I': 'Lê Văn I',
    #     'KH010K': 'Hoàng Thị K'
    # }
    # # Tạo DataFrame từ dictionary
    # df_KH = pd.DataFrame(list(data.items()), columns=['Mã số', 'Họ tên'])
    #     # In df_KH ra màn hình dạng table
    # st.write("Danh sách khách hàng:")
    # st.write(df_KH)
    # # lấy ngẫu nhiên 3 khách hàng từ df_KH
    # df_sample = df_KH.sample(5)
    # # In 3 khách hàng này ra màn hình
    # st.write("Danh sách khách hàng ngẫu nhiên:")
    # st.write(df_sample)
    # st.write("##### Đề xuất sản phẩm cho khách hàng dựa trên Mã số khách hàng")
    # # Tạo một điều khiển và đưa khách hàng ngẫu nhiên này vào đó    
    # selected = st.selectbox("Chọn khách hàng", df_sample['Họ tên'])
    # st.write("Khách hàng đã chọn:", selected)
    # # Từ khách hàng được chọn này lấy mã số tương ứng và tiếp tục xử lý phần đề xuất sản phẩm cho khách hàng này
    # # load model, predict và hiển thị kết quả
    # st.write("Xử lý và hiển thị thông tin Đề xuất sản phẩm cho khách hàng...")
    # st.write("##### Đề xuất sản phẩm cho khách hàng dựa trên sản phẩm")
    # # Tạo dataframe danh sách hàng hóa gồm 10 sản phẩm gồm mã sản phẩm, tên sản phẩm, mô tả tóm tắt khoảng 50 ký tự và giá bán
    # data = {
    #     'Mã SP': ['SP001', 'SP002', 'SP003', 'SP004', 'SP005', 'SP006', 'SP007', 'SP008', 'SP009', 'SP010'],
    #     'Tên SP': ['Áo thun nam', 'Áo sơ mi nữ', 'Quần jean nam', 'Quần legging nữ', 'Áo khoác nam', 'Áo len nữ', 'Quần kaki nam', 'Quần legging nữ', 'Áo khoác nam', 'Áo len nữ'],
    #     'Mô tả': ['Áo thun nam hàng hiệu', 'Áo sơ mi nữ hàng hiệu', 'Quần jean nam hàng hiệu', 'Quần legging nữ hàng hiệu', 'Áo khoác nam hàng hiệu', 'Áo len nữ hàng hiệu', 'Quần kaki nam hàng hiệu', 'Quần legging nữ hàng hiệu', 'Áo khoác nam hàng hiệu', 'Áo len nữ hàng hiệu'],
    #     'Giá': [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    # }
    # df_SP = pd.DataFrame(data)
    # # In danh sách sản phẩm ra table
    # st.dataframe(df_SP)
    # # Tạo điều khiển để người dùng chọn sản phẩm
    # st.write("##### 1. Chọn sản phẩm")
    # selected_SP = st.selectbox("Chọn sản phẩm", df_SP['Tên SP'])
    # st.write("Sản phẩm đã chọn:", selected_SP)
    # # Tìm sản phẩm liên quan đến sản phẩm đã chọn
    # st.write("##### 2. Sản phẩm liên quan")
    # # Lấy thông tin sản phẩm đã chọn
    # SP = df_SP[df_SP['Tên SP'] == selected_SP]
    # mo_ta_chon = SP['Mô tả'].iloc[0].lower()
    # # Gợi ý sản phẩm liên quan dựa theo mô tả của sản phẩm đã chọn, chuyển thành chữ thường trước khi tìm kiếm 
    # related_SP = df_SP[df_SP['Mô tả'].str.lower().str.contains(mo_ta_chon, na=False)]
    # # In danh sách sản phẩm liên quan ra màn hình
    # st.write("Danh sách sản phẩm liên quan:")
    # st.dataframe(related_SP)
    # # Từ sản phẩm đã chọn này, người dùng có thể xem thông tin chi tiết của sản phẩm, xem hình ảnh sản phẩm
    # # hoặc thực hiện các xử lý khác
    # # tạo điều khiển để người dùng tìm kiếm sản phẩm dựa trên thông tin người dùng nhập
    # st.write("##### 3. Tìm kiếm sản phẩm")
    # search = st.text_input("Nhập thông tin tìm kiếm")
    # # Tìm kiếm sản phẩm dựa trên thông tin người dùng nhập vào search, chuyển thành chữ thường trước khi tìm kiếm
    # result = df_SP[df_SP['Mô tả'].str.lower().str.contains(search.lower())]    
    # # In danh sách sản phẩm tìm được ra màn hình
    # st.write("Danh sách sản phẩm tìm được:")
    # st.dataframe(result)
    # # Từ danh sách sản phẩm tìm được này, người dùng có thể xem thông tin chi tiết của sản phẩm, xem hình ảnh sản phẩm...