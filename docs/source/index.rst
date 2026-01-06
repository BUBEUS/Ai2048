Dokumentacja Projektu 2048 AI
=============================

Wstęp
-----

Projekt polega na stworzeniu inteligentnego agenta (AI), który samodzielnie uczy się grać w popularną grę logiczną **2048**. 
Celem algorytmu jest osiągnięcie jak najwyższego wyniku oraz kafelka o wartości 2048 (lub wyższej) poprzez optymalizację wag i strategii ruchu.

Struktura Projektu
------------------

Poniżej przedstawiono strukturę plików w katalogu głównym projektu:

.. code-block:: text

    Ai2048/
    ├── source/                  # Folder z plikami źródłowymi dokumentacji
    │   ├── conf.py              # Konfiguracja Sphinx
    │   └── index.rst            # Główny plik spisu treści (ten plik)
    ├── ai_player.py             # Logika AI (Algorytm Minimax/Heurystyka)
    ├── benchmark_module.py      # Moduł do testowania skuteczności modelu
    ├── find_bestWagi.py         # Skrypt optymalizujący wagi (uczenie)
    ├── game_2048.py             # Główny silnik gry (logika bez grafiki)
    ├── game_gui.py              # Interfejs graficzny gry 
    ├── plot_charts.py           # Generowanie wykresów wyników
    └── train.py                 # Skrypt uruchamiający trening AI

Wymagania i Biblioteki
----------------------

Projekt napisany jest w języku Python. Wykorzystuje następujące biblioteki:

**Biblioteki zewnętrzne (wymagana instalacja):**
* ``pandas`` - do analizy i przetwarzania danych.
* ``numpy`` - do szybkich obliczeń macierzowych.
* ``matplotlib`` - do wizualizacji wyników (wykresy).

**Biblioteki standardowe (wbudowane w Python):**
* ``os`` - obsługa systemu plików.
* ``time`` - pomiar czasu i opóźnienia.
* ``pickle`` - zapisywanie i odczytywanie stanu nauki.
* ``tkinter`` - obsługa okien systemowych.
* ``threading`` - obsługa wielowątkowości.

Instalacja
----------

Aby zainstalować wymagane pakiety zewnętrzne, uruchom w terminalu:

.. code-block:: bash

   pip install pandas numpy matplotlib 

Jak uruchomić projekt
---------------------

1. **Aby zobaczyć grę w oknie graficznym:**

   .. code-block:: bash

      python game_gui.py

2. **Aby uruchomić trenowanie modelu AI:**

   .. code-block:: bash

      python train.py

3. **Aby wygenerować wykresy skuteczności:**

   .. code-block:: bash

      python plot_charts.py

Dokumentacja Kodu (API)
=======================

.. toctree::
   :maxdepth: 2
   :caption: Spis treści:

Silnik i Interfejs Gry
----------------------

.. automodule:: game_2048
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: game_gui
   :members:
   :undoc-members:
   :show-inheritance:

Sztuczna Inteligencja i Trenowanie
----------------------------------

.. automodule:: ai_player
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: train
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: find_bestWagi
   :members:
   :undoc-members:
   :show-inheritance:

Analiza i Wykresy
-----------------

.. automodule:: benchmark_module
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: plot_charts
   :members:
   :undoc-members:
   :show-inheritance: