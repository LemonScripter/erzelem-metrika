#!/bin/bash
# Alap függőségek telepítése requirements.txt nélkül PyTorch
pip install $(grep -v "torch" requirements.txt)

# PyTorch külön telepítése, csak CPU verzió a memóriahasználat csökkentéséhez
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu

# spaCy modellek telepítése
python -m spacy download hu_core_news_sm

echo "Setup completed successfully!"