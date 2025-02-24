# Kocka-Sík-Függvényes Szövegértelmező Rendszer

## A projektről

A Kocka-Sík-Függvényes Szövegértelmező Rendszer egy komplex érzelmi elemző alkalmazás, amely háromdimenziós térben modellezi a szövegekben megjelenő érzelmeket. A rendszer újszerű megközelítése lehetővé teszi az érzelmi dinamika részletes elemzését és vizualizációját.

## Funkciók

- **Háromdimenziós érzelmi modellezés**: Valencia (pozitív-negatív), Arousal (aktív-passzív), Dominancia (erős-gyenge)
- **Kontextus-érzékeny elemzés**: Különböző kontextusokban (üzleti, tudományos, személyes, közösségi média) eltérő érzelmi értelmezés
- **Érzelmi folyamatok vizualizációja**: 3D modellek, idősorok, 2D vetületek
- **Automatikus kontextus-felismerés**: Gépi tanulás alapú kontextus-detektálás
- **Érzelmi trajektória elemzés**: Időbeli érzelmi változások és mintázatok feltárása
- **Metszéspont-analízis**: Érzelmi görbék és kontextus-síkok metszéspontjainak értelmezése

## Rendszerkövetelmények

- Python 3.8+
- TensorFlow 2.4+
- SpaCy
- Flask
- Transformers (Hugging Face)
- NumPy, SciPy, Pandas
- Plotly, Matplotlib

## Telepítés

```bash
# Virtuális környezet létrehozása
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Függőségek telepítése
pip install -r requirements.txt

# SpaCy nyelvmodell letöltése (magyar vagy angol)
python -m spacy download hu_core_news_sm
# VAGY
python -m spacy download en_core_web_sm
```

## Használat

### Webalkalmazás indítása

```bash
python run.py
```

Ezután a böngészőben nyissa meg a http://127.0.0.1:5000 címet.

### Szöveg elemzése

Az alkalmazás lehetővé teszi:
1. Szöveg elemzését manuálisan megadott vagy automatikusan felismert kontextusban
2. Az eredmények részletes vizualizációját
3. HTML jelentés generálását az elemzési eredményekről

## Érzelmi tér magyarázata

A rendszer a következő érzelmi térmodelleket használja:

1. **Érzelmi kockák**: 8 alapvető érzelmi kategória a 3D tér oktánsainak megfelelően
   - Ellenséges: Alacsony valencia, magas arousal, alacsony dominancia
   - Stresszes: Alacsony valencia, magas arousal, magas dominancia
   - Izgatott: Magas valencia, magas arousal, alacsony dominancia
   - Lelkes: Magas valencia, magas arousal, magas dominancia
   - Depresszív: Alacsony valencia, alacsony arousal, alacsony dominancia
   - Nyugodt: Alacsony valencia, alacsony arousal, magas dominancia
   - Elégedett: Magas valencia, alacsony arousal, alacsony dominancia
   - Boldog: Magas valencia, alacsony arousal, magas dominancia

2. **Érzelmi síkok**: Különböző kontextusokhoz tartozó síkok, amelyek mentén eltérő érzelmi értelmezések jönnek létre

3. **Érzelmi függvények**: Időben változó érzelmi trajektóriák matematikai modellezése

## Modellek betanítása

A rendszer támogatja saját kontextuális érzelmi modellek betanítását:

```bash
# Adatkészlet létrehozása
python create_emotional_dataset.py --input texts.json --output dataset.csv

# Modell betanítása
python train_context_emotion_model.py --dataset dataset.csv --output models/my_model
```

## Fejlesztőknek

A projekt moduláris felépítésű, fő komponensei:

- `text_preprocessor.py`: Szövegek előfeldolgozása
- `emotion_analyzer.py`: Érzelmi dimenziók elemzése
- `emotion_cube_modeler.py`: Érzelmi tér modellezése
- `emotional_space_model.py`: Érzelmi kockák, síkok és függvények definiálása
- `emotion_visualizer.py`: Vizualizációs eszközök
- `context_detector.py`: Automatikus kontextus-felismerés
- `intersection_analyzer.py`: Érzelmi trajektóriák elemzése

## Licensz

MIT License

## Kapcsolat

[https://lemonscript.info]
