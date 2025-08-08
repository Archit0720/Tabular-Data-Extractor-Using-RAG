# 📊 Advanced Table Extraction RAG

A powerful Streamlit application that extracts tables from large PDF documents (up to 600 pages) and automatically names them using AI. Perfect for financial documents, reports, and data analysis.

## ✨ Features

- **🔍 Smart Table Extraction**: Multiple extraction methods (Camelot, Tabula, pdfplumber)
- **🤖 AI-Powered Naming**: Automatically name tables using Groq's LLM models
- **📊 Large File Support**: Handle PDFs up to 600 pages efficiently
- **⚡ Batch Processing**: Process pages in batches to optimize memory usage
- **📈 Financial Focus**: Specialized in financial table identification and naming
- **📥 Export Options**: Download individual tables or bulk summaries as CSV
- **🎯 Smart Filtering**: Filter tables by size and accuracy
- **📱 Responsive UI**: Modern, user-friendly interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Java 8+ (for Tabula)
- Ghostscript (for Camelot)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install openjdk-8-jdk ghostscript
```

#### macOS
```bash
brew install openjdk@8 ghostscript
```

#### Windows
- Install [Java JDK 8+](https://www.oracle.com/java/technologies/downloads/)
- Install [Ghostscript](https://www.ghostscript.com/download/gsdnld.html)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd advanced-table-extraction-rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Extraction Methods

- **Auto**: Tries all methods for best results
- **Camelot**: Best for tables with clear borders (lattice/stream)
- **Tabula**: Java-based, robust for various PDF types
- **pdfplumber**: Lightweight and fast for simple tables

### AI Models for Naming

- **llama-3.1-8b-instant**: Fast and efficient
- **llama-3.1-70b-versatile**: More accurate, slower
- **llama3-8b-8192**: Alternative option
- **gemma2-9b-it**: Backup model

### Processing Settings

- **Pages per batch**: 10-100 (default: 50)
- **Minimum table size**: Filter small tables (default: 6 cells)

## 📖 Usage

1. **Upload PDF**: Choose a PDF file (up to 600 pages)
2. **Configure Settings**: Select extraction method and AI model
3. **Extract Tables**: Click "Start Table Extraction"
4. **Name Tables**: Use "Name All Tables with AI" for automatic naming
5. **Export Data**: Download individual tables or summary report

## 🏗️ Project Structure

```
advanced-table-extraction-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
├── README.md             # This file
├── docs/                 # Documentation
│   ├── installation.md   # Detailed installation guide
│   └── troubleshooting.md # Common issues and solutions
└── examples/             # Example files and outputs
    ├── sample.pdf        # Sample PDF for testing
    └── output_examples/   # Example extracted tables
```

## 🔑 Environment Variables

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com/keys

## 🛠️ Troubleshooting

### Common Issues

#### Java Not Found (Tabula)
```bash
# Check Java installation
java -version

# Set JAVA_HOME if needed
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
```

#### Ghostscript Not Found (Camelot)
```bash
# Check Ghostscript installation
gs --version

# On Ubuntu/Debian
sudo apt-get install ghostscript

# On macOS
brew install ghostscript
```

#### Memory Issues with Large PDFs
- Reduce "Pages per batch" setting
- Use lighter extraction methods (pdfplumber)
- Close other applications to free memory

#### Poor Table Detection
- Try different extraction methods
- Adjust minimum table size
- Check PDF quality and table structure

### Error Messages

| Error | Solution |
|-------|----------|
| `GROQ_API_KEY not found` | Add API key to `.env` file |
| `Java not found` | Install Java JDK 8+ |
| `Ghostscript not found` | Install Ghostscript |
| `Memory error` | Reduce batch size |

## 📊 Performance Tips

- **Large Files**: Use smaller batch sizes (10-25 pages)
- **Speed**: Use pdfplumber for simple tables
- **Accuracy**: Use Camelot for complex bordered tables
- **Mixed Content**: Use Auto mode to try all methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web app framework
- [Groq](https://groq.com/) - AI inference platform
- [Camelot](https://camelot-py.readthedocs.io/) - PDF table extraction
- [Tabula](https://tabula-py.readthedocs.io/) - PDF table extraction
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF text extraction

## 📞 Support

- 📧 Email: architdogra07@gmail.com


## 🔄 Changelog

### v1.0.0 (Current)
- Initial release
- Multi-method table extraction
- AI-powered table naming
- Batch processing for large files
- Export functionality
- Responsive UI

---

**Made with ❤️ for data extraction and analysis**
