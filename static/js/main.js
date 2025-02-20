// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    // Alapvető felület kezelése
    console.log('Kocka-Sík-Függvényes Szövegértelmező alkalmazás betöltve!');
    
    // Kontextus választó kezelése
    const contextOptionAuto = document.getElementById('context_auto');
    const contextOptionManual = document.getElementById('context_manual');
    const contextSelectorContainer = document.getElementById('contextSelectorContainer');
    const contextSelector = document.getElementById('context');
    
    if (contextOptionAuto && contextOptionManual && contextSelectorContainer) {
        // Rádiógombok változásának kezelése
        contextOptionAuto.addEventListener('change', toggleContextSelector);
        contextOptionManual.addEventListener('change', toggleContextSelector);
        
        // Inicializáláskor ellenőrizzük az állapotot
        toggleContextSelector();
        
        // Kontextusok betöltése
        loadAvailableContexts();
    }
    
    function toggleContextSelector() {
        if (contextOptionManual.checked) {
            contextSelectorContainer.classList.remove('d-none');
        } else {
            contextSelectorContainer.classList.add('d-none');
        }
    }
    
    function loadAvailableContexts() {
        fetch('/contexts')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.contexts) {
                    // Kontextusok feltöltése a választóba
                    contextSelector.innerHTML = '';
                    
                    data.contexts.forEach(context => {
                        const option = document.createElement('option');
                        option.value = context;
                        option.textContent = formatContextName(context);
                        contextSelector.appendChild(option);
                    });
                }
            })
            .catch(error => {
                console.error('Kontextusok betöltési hiba:', error);
            });
    }
    
    function formatContextName(context) {
        // Kontextus név formázása (első betű nagybetű, underscore helyett szóköz)
        return context
            .replace(/_/g, ' ')
            .replace(/\b\w/g, char => char.toUpperCase());
    }
});