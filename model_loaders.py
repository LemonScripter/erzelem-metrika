# model_loaders.py

def get_torch_model(model_name="distilbert-base-uncased"):
    """
    Lusta betöltés a PyTorch modellek számára.
    Csak akkor tölti be a torch és transformers modulokat,
    amikor ténylegesen szükség van rájuk.
    
    Args:
        model_name: A betöltendő modell neve a Hugging Face modell hubról
        
    Returns:
        A betöltött modell
    """
    import gc
    
    # TensorFlow munkamenetének tisztítása, ha aktív
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass
    
    # Most importáljuk a torch-ot
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    # Memória felszabadítása
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Modell betöltése CPU-ra
    device = torch.device("cpu")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.to(device)
    model.eval()  # Értékelési módba kapcsolás
    
    return model, tokenizer