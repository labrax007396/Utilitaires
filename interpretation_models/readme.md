## Description
La librairie interpretation_models permet de calculer la contribution des facteurs d'un modèle sous forme de shape values
(https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)

## Données d'entrée:
- pkl_model_file : emplacement du modèle au format pkl (shapley ne prend pas en charge les modèles au format onnx)
- config_model_file : fichier json de configuration du modèle qui a été utilisé pour créer le modèle
- date_debut : date de début de calcul des shapes values
- date_fin : date de fin de calcul des shapes values

## Données sortie:

- shape values au format .csv

## Contenu

- data_test: contient un exemple de modèle au format pkl ainsi que le fichier de configuration de celui ci (model_config.json)
- compute_shape_values.py: script python pour calculer les shape values

## Fonctionnement

la librairie charge les données depuis influx db et calcule les valeurs du shapeley


## Synthaxe

- python compute_shape_values.py config_model_file date_debut date_fin pkl_model_file


## Test

- Dans le répertoir "Modeling", activer l'environnement virtuel: python_env/Script/activate
- Dans l'invité de commande, se placer dans le répertoire "Modeling/packages/interpretation_models"
- Executer: python compute_shape_values.py "data_test/model_config.json" "09/04/2023  00:00:00" "14/04/2023  11:00:00" "data_test/model.pkl"