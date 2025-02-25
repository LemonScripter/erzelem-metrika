#!/bin/bash
# Alap függőségek telepítése
pip install -r requirements.txt

# spaCy modellek telepítése
# Először a kisebb (sm) modelleket telepítjük, mert ezek kevesebb memóriát igényelnek
python -m spacy download hu_core_news_sm
# Ha angolra is szükséged van
# python -m spacy download en_core_web_sm

echo "Setup completed successfully!"