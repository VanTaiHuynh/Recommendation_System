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
