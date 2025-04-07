install requirements 
pip install -r requirements.txt

conda env create -f environment.yml

guardar os requirements
conda list --export > requirements.txt

Gerar o build/env 
conda env export --no-builds > environment.yml