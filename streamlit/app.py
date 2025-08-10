import json
import requests
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

st.title('Spam or Not Spam?')

txt = st.text_area('Paste the e-mail content bellow and click "Analyse" to verify')

if st.button("Analyse", type="primary"):
    data = {
        "content": txt
    }
    with st.spinner("Analysing content..."):
                response = requests.post("https://model-inference-314766854658.us-east1.run.app/classificate_email/", json=data).json()

                st.markdown(f'''**Is Spam?** :red[{response['is_spam']}]''')
                st.markdown(f'''**Reason:** {response['reason']}''')