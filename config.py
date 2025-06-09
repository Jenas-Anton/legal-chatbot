import streamlit as st

def show_sidebar():
    """Display sidebar with API configuration and settings"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Enter your Gemini API Key:",
            type="password",
            help="Enter your Google Gemini API key to enable AI features",
            placeholder="Your Gemini API key here..."
        )
        
        if api_key and api_key.strip():
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your API key")
        
        st.markdown("---")
        
        # Number of cases setting
        st.subheader("📊 Search Settings")
        num_cases = st.slider(
            "Number of similar cases to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="Select how many similar cases to show in analysis"
        )
        
        st.markdown("---")
        
      

        
        # Information section
        st.subheader("ℹ️ About")
        st.markdown("""
        **Legal AI Assistant**
        
        Features:
        - Legal case analysis
        - Document summarization
        - Case outcome prediction
        - Legal document generation
        
        **Note:** This tool provides general legal information only. Always consult qualified legal professionals for specific legal advice.
        """)
        
        # API Key help
        with st.expander("🔗 How to get Gemini API Key"):
            st.markdown("""
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Create API Key"
            4. Copy and paste the key above
            
            The API key is free to use with generous limits.
            """)
    
    return api_key, num_cases

def get_app_config():
    """Return app configuration settings"""
    return {
        "page_title": "Comprehensive Legal AI Assistant",
        "page_icon": "⚖️",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }