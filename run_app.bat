@echo off
echo =====================================================================
echo  Brain Tumor MRI Classification and Explainable AI (XAI) System
echo =====================================================================
echo.
echo Launching Streamlit GUI Application...
echo.

py -3.10 -m streamlit run app.py --server.port 8501 || streamlit run app.py --server.port 8501

pause
