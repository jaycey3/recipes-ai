# 🍎 Recept Extractor

Een applicatie dat kookvideo's omzet naar gestructureerde recepten. De applicatie
combineert object detection (Faster R-CNN), audio transcriptie (Whisper), Optical Character Recognition (EasyOCR)
en een LLM (Claude) om uit een video een recept met ingredientenlijst
en stappen te genereren.

## 📐 Architectuur

Video → [Frames + Audio] → [Detectie + Transcript + OCR] → [Merge] → [Claude] → Recept

Zie `notebooks/recipes.ipynb` voor het trainen van het object detection model.

## 📂 Structuur

- recepten-ai/
- ├── notebooks/        Trainingsnotebook en datast voor Faster R-CNN
- ├── src/              Pipeline modules
- │   ├── frames.py     Frame extraction (OpenCV)
- │   ├── detection.py  Object detection wrapper
- │   ├── audio.py      Audio transcriptie met Whisper
- │   ├── merge.py      Detecties, transcript en OCR samenvoegen
- │   ├── llm.py        Claude API call
- │   ├── ocr.py        OCR met EasyOCR
- │   └── pipeline.py   Orchestratie van alle stappen
- ├── app/              Streamlit web frontend
- ├── models/           Getrainde object detection models (.pth)
- └── data/             Input videos + output recepten

## 🚀 Setup

1. **Installeer dependencies:**
   pip install -r requirements.txt

2. **Installeer ffmpeg** (systeem, niet via pip):
   - macOS: brew install ffmpeg
   - Windows: download van https://ffmpeg.org

3. **Maak een `.env` bestand** met je Anthropic API key:
   cp .env.example .env

4. **Train het detection model** (eenmalig):
   - Open `notebooks/recipes.ipynb`
   - Run alle cellen
   - Het model wordt opgeslagen in `models/faster_rcnn_model.pth`

## 💻 Gebruik

### Streamlit web app

streamlit run app/streamlit.py

## 🚧 Bekende beperkingen / future work

- Dataset moet uitgebreid worden naar meer ingredienten
