import streamlit as st
def printRecomProductList(products, width_image = 200, col=2):
    products['image'] = products['image'].fillna('https://via.placeholder.com/{0}'.format(width_image))
    if products.empty:
        st.write("No products to display.")
        return
    header_cols = st.columns([2, 3, 1])
    with header_cols[0]:
        st.markdown("**Image**")
    with header_cols[1]:
        st.markdown("**Product Name**")
    with header_cols[2]:
        st.markdown("**Price**")

    for _, row in products.iterrows():
        cols = st.columns([1, 3, 1])  # Adjust the column ratios as needed
        with cols[0]:
            st.image(row['image'], width=100, output_format='PNG')
        with cols[1]:
            # Use Markdown to control the font size and add border
            st.markdown(f"<div style='font-size: 14px; border:1px solid #ccc; padding: 5px;'>{row['product_name']}</div>", unsafe_allow_html=True)
        with cols[2]:
            # Add border for price column
            st.markdown(f"<div style='border:1px solid #ccc; padding: 5px;'>{row['price']}</div>", unsafe_allow_html=True)
def printRecomProductListwithButton(products, width_image=200):
    if products.empty:
        st.write("No products to display.")
        return
    products['image'] = products['image'].fillna('https://via.placeholder.com/{0}'.format(width_image))
    header_cols = st.columns([2, 3, 1, 1])
    with header_cols[0]:
        st.markdown("**Image**")
    with header_cols[1]:
        st.markdown("**Product Name**")
    with header_cols[2]:
        st.markdown("**Price**")
    with header_cols[3]:
        st.markdown("**Action**")

    for _, row in products.iterrows():
        cols = st.columns([2, 3, 1, 1])
        with cols[0]:
            st.image(row['image'], width=width_image)
        with cols[1]:
            st.markdown(f"**{row['product_name']}**")
        with cols[2]:
            st.markdown(f"**{row['price']}**")
        with cols[3]:
            if st.button("Tìm sản phẩm tương tự", key=f"details_{row['product_id']}"):
                st.session_state['clicked_product'] = row['product_id']
def printProductDetail(product):
    st.write('*'*50)
    st.write("## Product Details")
    st.image(product['image'], width=200)
    st.markdown(f"**Product Name:** {product['product_name']}")
    st.markdown(f"**Category:** {product.get('category', 'N/A')}")
    st.markdown(f"**Price:** {product['price']}")