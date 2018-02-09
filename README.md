# DH Network Simulations

Simulations of hourly energy demand, capacity, storage, and CO2 for different expansion scenarios of 
District Heating (DH) network using EnergyPlus building energy demand .csv files and
PVT generation input .csv file.

## Run script
It is recommended to use a python environment manager such as venv or conda. Make sure you have python 3.3+, pandas, numpy, and matplotlib. You should be able to import them without errors.
```bash
python3
import pandas
import numpy
import matplotlib
```
Run the script from terminal:
```bash
python Hourly_Balance_Model_v3.py
```

## output sankey diagrams
To generate the sankey diagrams use this online tool: http://sankeymatic.com/build/ <br>
Copy and paste the sankey.txt files directly into the input box and hit "preview".