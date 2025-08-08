import streamlit as st
import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import io
import pdfplumber
import fitz  # PyMuPDF
import camelot
import tabula
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced Table Extraction RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .table-container {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .progress-container {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = None
    if 'extracted_tables' not in st.session_state:
        st.session_state.extracted_tables = []
    if 'table_extraction_complete' not in st.session_state:
        st.session_state.table_extraction_complete = False
    if 'named_tables' not in st.session_state:
        st.session_state.named_tables = {}

def setup_groq_client():
    """Setup Groq client"""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error("❌ GROQ_API_KEY not found in .env file")
        st.stop()
    
    try:
        st.session_state.groq_client = Groq(api_key=groq_key)
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        return False

def extract_tables_camelot(pdf_file, pages_per_batch=50):
    """Extract tables using Camelot (best for PDFs with clear table structures)"""
    tables = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        # Get total pages
        pdf_document = fitz.open(tmp_file_path)
        total_pages = pdf_document.page_count
        pdf_document.close()
        
        status_text.text(f"📄 Processing {total_pages} pages with Camelot...")
        
        # Process pages in batches to avoid memory issues
        for batch_start in range(1, total_pages + 1, pages_per_batch):
            batch_end = min(batch_start + pages_per_batch - 1, total_pages)
            page_range = f"{batch_start}-{batch_end}"
            
            try:
                # Extract tables from current batch
                batch_tables = camelot.read_pdf(
                    tmp_file_path,
                    pages=page_range,
                    flavor='lattice',  # Try lattice first (for bordered tables)
                    line_scale=40,
                    copy_text=['v']
                )
                
                if len(batch_tables) == 0:
                    # Try stream flavor for borderless tables
                    batch_tables = camelot.read_pdf(
                        tmp_file_path,
                        pages=page_range,
                        flavor='stream',
                        table_areas=None,
                        columns=None,
                        row_tol=2
                    )
                
                for i, table in enumerate(batch_tables):
                    if table.df.shape[0] > 1 and table.df.shape[1] > 1:  # Filter out empty tables
                        tables.append({
                            'page': table.parsing_report['page'],
                            'table_index': i,
                            'dataframe': table.df,
                            'accuracy': table.accuracy if hasattr(table, 'accuracy') else 0.0,
                            'method': 'camelot'
                        })
                
                progress = min((batch_end / total_pages), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"📊 Processed pages {batch_start}-{batch_end} | Found {len(tables)} tables so far")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing pages {page_range}: {str(e)}")
                continue
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Camelot extraction complete! Found {len(tables)} tables")
        return tables
        
    except Exception as e:
        st.error(f"❌ Camelot extraction failed: {str(e)}")
        return []

def extract_tables_tabula(pdf_file, pages_per_batch=50):
    """Extract tables using Tabula (Java-based, good alternative)"""
    tables = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        # Get total pages
        pdf_document = fitz.open(tmp_file_path)
        total_pages = pdf_document.page_count
        pdf_document.close()
        
        status_text.text(f"📄 Processing {total_pages} pages with Tabula...")
        
        # Process pages in batches
        for batch_start in range(1, total_pages + 1, pages_per_batch):
            batch_end = min(batch_start + pages_per_batch - 1, total_pages)
            page_range = list(range(batch_start, batch_end + 1))
            
            try:
                # Extract tables from current batch
                batch_tables = tabula.read_pdf(
                    tmp_file_path,
                    pages=page_range,
                    multiple_tables=True,
                    pandas_options={'header': None}
                )
                
                for i, df in enumerate(batch_tables):
                    if df.shape[0] > 1 and df.shape[1] > 1:  # Filter out empty tables
                        # Estimate which page this table is from
                        estimated_page = batch_start + (i * len(page_range) // len(batch_tables))
                        
                        tables.append({
                            'page': estimated_page,
                            'table_index': i,
                            'dataframe': df,
                            'accuracy': 0.8,  # Default accuracy for tabula
                            'method': 'tabula'
                        })
                
                progress = min((batch_end / total_pages), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"📊 Processed pages {batch_start}-{batch_end} | Found {len(tables)} tables so far")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing pages {batch_start}-{batch_end}: {str(e)}")
                continue
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Tabula extraction complete! Found {len(tables)} tables")
        return tables
        
    except Exception as e:
        st.error(f"❌ Tabula extraction failed: {str(e)}")
        return []

def extract_tables_pdfplumber(pdf_file, pages_per_batch=50):
    """Extract tables using pdfplumber (lightweight, good for simple tables)"""
    tables = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        pdf_file.seek(0)  # Reset file pointer
        
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            status_text.text(f"📄 Processing {total_pages} pages with pdfplumber...")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract tables from current page
                    page_tables = page.extract_tables()
                    
                    for i, table in enumerate(page_tables):
                        if table and len(table) > 1 and len(table[0]) > 1:  # Filter out empty tables
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])  # First row as header
                            
                            tables.append({
                                'page': page_num + 1,
                                'table_index': i,
                                'dataframe': df,
                                'accuracy': 0.7,  # Default accuracy for pdfplumber
                                'method': 'pdfplumber'
                            })
                    
                    # Update progress
                    if page_num % 10 == 0:  # Update every 10 pages
                        progress = (page_num + 1) / total_pages
                        progress_bar.progress(progress)
                        status_text.text(f"📊 Processed page {page_num + 1}/{total_pages} | Found {len(tables)} tables so far")
                    
                except Exception as e:
                    st.warning(f"⚠️ Error processing page {page_num + 1}: {str(e)}")
                    continue
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ pdfplumber extraction complete! Found {len(tables)} tables")
        return tables
        
    except Exception as e:
        st.error(f"❌ pdfplumber extraction failed: {str(e)}")
        return []

def name_table_with_ai(client, table_data, model_name):
    """Use Groq AI to name tables based on their content"""
    try:
        # Convert table to string representation (first few rows)
        df = table_data['dataframe']
        sample_data = df.head(5).to_string() if df.shape[0] > 5 else df.to_string()
        
        # Prepare prompt for financial table naming
        prompt = f"""
Analyze this financial table data and provide a descriptive name in financial terms:

Table Data Sample:
{sample_data}

Table Info:
- Page: {table_data['page']}
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Extraction Method: {table_data['method']}

Instructions:
1. Give this table a specific financial name based on its content
2. Consider common financial statements: Balance Sheet, Income Statement, Cash Flow, P&L, etc.
3. Be specific about the type of data (e.g., "Quarterly Revenue by Segment", "Asset Breakdown", "Liability Schedule")
4. Keep the name under 10 words
5. If you can't determine the financial nature, use a generic descriptive name

Respond with ONLY the table name, no explanation.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a financial analyst expert at identifying and naming financial tables and statements."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=50
        )
        
        table_name = response.choices[0].message.content.strip()
        return table_name
        
    except Exception as e:
        return f"Financial Table {table_data['page']}-{table_data['table_index']}"

def main():
    # Initialize session state
    initialize_session_state()
    
    # Setup Groq client
    if not st.session_state.groq_client:
        if not setup_groq_client():
            return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Advanced Table Extraction RAG</h1>
        <p>Extract tables from large PDFs and name them intelligently!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload & Settings")
        
        # PDF Upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file (up to 600 pages supported)"
        )
        
        # Extraction method selection
        st.subheader("🔧 Extraction Method")
        extraction_method = st.selectbox(
            "Choose extraction method:",
            [
                "Auto (Try all methods)",
                "Camelot (Best for bordered tables)",
                "Tabula (Java-based, robust)",
                "pdfplumber (Lightweight, fast)"
            ],
            help="Different methods work better for different PDF types"
        )
        
        # Model selection for naming
        st.subheader("🤖 AI Model for Naming")
        naming_model = st.selectbox(
            "Choose model for table naming:",
            [
                "llama-3.1-8b-instant",     # Fast for naming
                "llama-3.1-70b-versatile",  # More accurate
                "llama3-8b-8192",           # Alternative
                "gemma2-9b-it"              # Backup
            ],
            help="Lighter models are faster for naming tasks"
        )
        
        # Processing settings
        st.subheader("⚙️ Processing Settings")
        pages_per_batch = st.slider(
            "Pages per batch:",
            min_value=10,
            max_value=100,
            value=50,
            help="Smaller batches use less memory but take longer"
        )
        
        min_table_size = st.slider(
            "Minimum table size (rows x cols):",
            min_value=4,
            max_value=20,
            value=6,
            help="Filter out very small tables"
        )
        
        # Status
        st.markdown("---")
        st.subheader("📊 Status")
        if st.session_state.table_extraction_complete:
            st.success(f"🟢 {len(st.session_state.extracted_tables)} Tables Extracted")
            named_count = len(st.session_state.named_tables)
            st.info(f"🏷️ {named_count} Tables Named")
        else:
            st.warning("🟡 No Tables Extracted Yet")
    
    # Main content area
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Start Table Extraction", type="primary", use_container_width=True):
                st.session_state.extracted_tables = []
                st.session_state.table_extraction_complete = False
                st.session_state.named_tables = {}
                
                with st.spinner("🔍 Extracting tables from PDF..."):
                    # Choose extraction method
                    if extraction_method == "Auto (Try all methods)":
                        # Try Camelot first, then others
                        tables = extract_tables_camelot(uploaded_file, pages_per_batch)
                        if len(tables) < 5:  # If not many tables found, try other methods
                            uploaded_file.seek(0)
                            tables.extend(extract_tables_tabula(uploaded_file, pages_per_batch))
                            uploaded_file.seek(0)
                            tables.extend(extract_tables_pdfplumber(uploaded_file, pages_per_batch))
                    elif extraction_method == "Camelot (Best for bordered tables)":
                        tables = extract_tables_camelot(uploaded_file, pages_per_batch)
                    elif extraction_method == "Tabula (Java-based, robust)":
                        tables = extract_tables_tabula(uploaded_file, pages_per_batch)
                    else:  # pdfplumber
                        tables = extract_tables_pdfplumber(uploaded_file, pages_per_batch)
                    
                    # Filter tables by minimum size
                    filtered_tables = []
                    for table in tables:
                        df = table['dataframe']
                        if df.shape[0] * df.shape[1] >= min_table_size:
                            filtered_tables.append(table)
                    
                    st.session_state.extracted_tables = filtered_tables
                    st.session_state.table_extraction_complete = True
                    
                    if filtered_tables:
                        st.success(f"✅ Successfully extracted {len(filtered_tables)} tables!")
                    else:
                        st.warning("⚠️ No tables found. Try adjusting the extraction method or settings.")
        
        with col2:
            if st.session_state.extracted_tables and st.button("🏷️ Name All Tables with AI", type="secondary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_tables = len(st.session_state.extracted_tables)
                
                for i, table in enumerate(st.session_state.extracted_tables):
                    status_text.text(f"🤖 Naming table {i+1}/{total_tables}...")
                    
                    table_name = name_table_with_ai(
                        st.session_state.groq_client,
                        table,
                        naming_model
                    )
                    
                    table_key = f"page_{table['page']}_table_{table['table_index']}"
                    st.session_state.named_tables[table_key] = table_name
                    
                    progress_bar.progress((i + 1) / total_tables)
                    time.sleep(0.1)  # Small delay to prevent API rate limiting
                
                status_text.text(f"✅ All {total_tables} tables have been named!")
                st.success("🎉 Table naming complete!")
    
    # Display extracted tables
    if st.session_state.extracted_tables:
        st.markdown("---")
        st.subheader("📊 Extracted Tables")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tables", len(st.session_state.extracted_tables))
        with col2:
            total_cells = sum(table['dataframe'].shape[0] * table['dataframe'].shape[1] 
                            for table in st.session_state.extracted_tables)
            st.metric("Total Cells", total_cells)
        with col3:
            pages_with_tables = len(set(table['page'] for table in st.session_state.extracted_tables))
            st.metric("Pages with Tables", pages_with_tables)
        with col4:
            named_count = len(st.session_state.named_tables)
            st.metric("Named Tables", named_count)
        
        # Display each table
        for i, table in enumerate(st.session_state.extracted_tables):
            table_key = f"page_{table['page']}_table_{table['table_index']}"
            table_name = st.session_state.named_tables.get(table_key, f"Table {i+1}")
            
            with st.expander(f"📋 {table_name} (Page {table['page']}, Method: {table['method']})", expanded=(i==0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(table['dataframe'], use_container_width=True, height=300)
                
                with col2:
                    st.write("**Table Info:**")
                    st.write(f"📄 Page: {table['page']}")
                    st.write(f"📊 Size: {table['dataframe'].shape[0]} × {table['dataframe'].shape[1]}")
                    st.write(f"🎯 Accuracy: {table['accuracy']:.1%}")
                    st.write(f"🔧 Method: {table['method']}")
                    
                    # Manual naming option
                    if st.button(f"✏️ Rename", key=f"rename_{i}"):
                        new_name = st.text_input(f"New name for table {i+1}:", value=table_name, key=f"name_input_{i}")
                        if new_name != table_name:
                            st.session_state.named_tables[table_key] = new_name
                            st.rerun()
                    
                    # Export options
                    csv_data = table['dataframe'].to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        file_name=f"{table_name.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"download_{i}"
                    )
        
        # Bulk export option
        if st.session_state.extracted_tables:
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("📦 Export All Tables Summary", type="secondary"):
                    summary_data = []
                    for i, table in enumerate(st.session_state.extracted_tables):
                        table_key = f"page_{table['page']}_table_{table['table_index']}"
                        table_name = st.session_state.named_tables.get(table_key, f"Table {i+1}")
                        
                        summary_data.append({
                            'Table_Name': table_name,
                            'Page': table['page'],
                            'Rows': table['dataframe'].shape[0],
                            'Columns': table['dataframe'].shape[1],
                            'Method': table['method'],
                            'Accuracy': f"{table['accuracy']:.1%}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    csv_summary = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        "📊 Download Tables Summary",
                        csv_summary,
                        file_name="tables_summary.csv",
                        mime="text/csv"
                    )
    
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
            <h2>🚀 Advanced Table Extraction</h2>
            <p style="font-size: 1.1em; margin: 2rem 0;">
                Upload a PDF file to extract tables and automatically name them using AI!
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;">
                    <h3>📊 Smart Extraction</h3>
                    <p>Multiple extraction methods for different PDF types</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;">
                    <h3>🤖 AI Naming</h3>
                    <p>Automatically name tables using financial terminology</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;">
                    <h3>⚡ Large Files</h3>
                    <p>Handle PDFs up to 600 pages efficiently</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Installation instructions
    with st.expander("📚 Installation Requirements", expanded=False):
        st.markdown("""
        **Required Python packages:**
        ```bash
        pip install streamlit python-dotenv groq
        pip install PyPDF2 pdfplumber PyMuPDF
        pip install camelot-py[cv] tabula-py
        pip install pandas opencv-python pillow
        ```
        
        **System requirements:**
        - Java (for Tabula)
        - Ghostscript (for Camelot)
        
        **For Ubuntu/Debian:**
        ```bash
        sudo apt-get install openjdk-8-jdk
        sudo apt-get install ghostscript
        ```
        
        **For macOS:**
        ```bash
        brew install openjdk@8
        brew install ghostscript
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #6c757d;">
        <p>📊 Advanced Table Extraction | 🤖 Powered by Groq | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()