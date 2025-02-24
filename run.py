# run.py
from app import app
import os

if __name__ == '__main__':
    print("Kocka-Sík-Függvényes Szövegértelmező alkalmazás indítása...")
    print("A webszerver a következő címen érhető el: http://127.0.0.1:5000")
    # A Render.com által biztosított PORT környezeti változó használata
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)