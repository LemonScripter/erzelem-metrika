# run.py
import os

# A Render.com által biztosított PORT környezeti változó beolvasása
port = int(os.environ.get('PORT', 10000))

# Később ez a port lesz felhasználva a webszerver indításakor

# Memória optimalizáláshoz beállítások a TensorFlow számára
# Ezeket fontos a modulok betöltése előtt beállítani
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3: csak ERROR logok
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU kikapcsolása

# Opcionális: TensorFlow memória korlátozása
import tensorflow as tf
# A memóriakorlát beállítása - a példában 300MB, 
# ami 512MB alatt tartja a teljes memóriahasználatot
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(
    tf.config.experimental.list_physical_devices('CPU')[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=300)]
)

# Csak ezután importáljuk a Flask alkalmazást
from app import app

if __name__ == '__main__':
    print("Kocka-Sík-Függvényes Szövegértelmező alkalmazás indítása...")
    print(f"A webszerver a következő címen érhető el: http://0.0.0.0:{port}")
    # Itt használjuk a korábban beolvasott port értéket
    app.run(host='0.0.0.0', port=port, debug=False)