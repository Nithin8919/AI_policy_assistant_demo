#!/usr/bin/env python3
"""
Minimal test dashboard to debug the issue
"""
import streamlit as st
import requests
import json

st.set_page_config(page_title="Debug Dashboard", layout="wide")

st.title("🔍 Debug Dashboard")

# Test API connection
if st.button("Test API Connection"):
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is healthy")
            st.json(response.json())
        else:
            st.error(f"❌ API Error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")

# Test query
query = st.text_input("Enter query:", value="schools in Krishna")
mode = st.selectbox("Mode:", ["citation_first", "exploratory", "balanced", "auto"])

if st.button("Test Query"):
    try:
        payload = {
            "query": query,
            "mode": mode,
            "limit": 3
        }
        
        response = requests.post(
            "http://localhost:8000/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            st.success("✅ Query successful")
            
            # Show raw response
            st.subheader("Raw API Response")
            st.json(results)
            
            # Show parsed data
            st.subheader("Parsed Data")
            st.write(f"**Mode used:** {results.get('metadata', {}).get('mode_used', 'unknown')}")
            st.write(f"**Answer:** {results.get('answer', 'No answer')}")
            st.write(f"**Citations:** {len(results.get('citations', []))}")
            st.write(f"**Total results:** {results.get('total_results', 0)}")
            
            # Test rendering
            st.subheader("Rendered Output")
            answer = results.get('answer', 'No response generated')
            st.markdown("#### 🤖 Response")
            st.write(answer)
            
            citations = results.get('citations', [])
            if citations:
                st.markdown(f"#### 📚 Citations ({len(citations)})")
                for i, citation in enumerate(citations):
                    st.write(f"**Citation {i+1}:** {citation.get('doc_title', 'Unknown')}")
                    st.write(f"**Excerpt:** {citation.get('excerpt', 'No excerpt')[:100]}...")
            
        else:
            st.error(f"❌ Query failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"❌ Query error: {e}")

