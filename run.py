# run.py
from app import app

if __name__ == '__main__':
    print("Kocka-Sík-Függvényes Szövegértelmező alkalmazás indítása...")
    print("A webszerver a következő címen érhető el: http://127.0.0.1:5000")
    app.run(debug=True)