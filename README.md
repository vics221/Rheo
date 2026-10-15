# Rheology Matrix Master (Streamlit)

Upload Excel-filer (.xls/.xlsx) fra reologi-målinger og få plots og statistik.

## Kør lokalt
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy på Streamlit Community Cloud
1. Læg `app.py`, `requirements.txt` og `README.md` i et GitHub-repo.
2. Gå til https://share.streamlit.io og log ind med GitHub.
3. Vælg "Create app" -> dit repo -> branch `main` -> main file `app.py` -> Deploy.
4. Du får et link (`https://dit-navn.streamlit.app`) som kan åbnes i Chrome.
