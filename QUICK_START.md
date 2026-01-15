# 🚀 Quick Start Guide

## ✅ Issues Fixed

**✅ Import Conflict Resolved** - Renamed `statistics/` → `stat_analysis/` to avoid conflict with Python's built-in statistics module

**✅ Streamlit Config Fixed** - Moved `st.set_page_config()` to top of file before any other Streamlit commands

**✅ Graceful Error Handling** - App shows helpful preview mode when dependencies aren't installed

## 🏃‍♂️ Quick Setup (2 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the App  
```bash
streamlit run main_app.py
```

**That's it!** 🎉

## 🔧 Alternative Installation

If you prefer automated setup:
```bash
python install_dependencies.py
streamlit run main_app.py
```

## 📱 What You'll See

✅ **With Dependencies:** Full interactive ML algorithms + statistics
⚠️ **Without Dependencies:** Preview mode with installation instructions

## 🧪 Verify Setup

Test the structure is working:
```bash
python test_structure.py
```

## 📂 Current Structure

```
Stat_Visualizer-/
├── algorithms/              # 10+ ML algorithms
├── stat_analysis/           # Statistical analysis (renamed from statistics)
├── utils/                   # Helper functions
├── main_app.py             # 🆕 New modular app
├── app.py                  # Original app (preserved)
├── requirements.txt        # Dependencies
├── install_dependencies.py # Auto-installer
├── test_structure.py      # Structure tester
├── QUICK_START.md         # This guide
└── README.md              # Full documentation
```

## 🎓 Learning Journey

1. **📊 Home** - Overview and learning path
2. **📊 Statistics** - Descriptive statistics with your data  
3. **🤖 Machine Learning** - 10+ interactive algorithms
4. **ℹ️ About** - Complete documentation

## 🆘 Troubleshooting

**"Dependencies Missing" error:**
```bash
pip install streamlit numpy pandas matplotlib seaborn plotly scipy scikit-learn
```

**Port already in use:**
```bash
streamlit run main_app.py --server.port 8502
```

**Permission issues:**
```bash
pip install --user -r requirements.txt
```

---

**Ready to learn statistics and ML interactively! 📊🤖**