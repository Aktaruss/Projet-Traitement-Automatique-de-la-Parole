#!/bin/bash

VENV_NAME="env_tap_tp_partie1"

echo "Démarrage de l'installation pour le TP TAP"

if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "Le paquet 'python3-venv' est manquant il faut faire : sudo apt update && sudo apt install python3-venv"
    exit 1
fi

if [ -d "$VENV_NAME" ]; then
    echo "Le dossier '$VENV_NAME' existe déjà. On continue l'installation dedans."
else
    echo "Création de l'environnement virtuel '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
fi

echo "⬇️  Mise à jour de pip et installation des librairies..."
./$VENV_NAME/bin/pip install --upgrade pip
./$VENV_NAME/bin/pip install -r requirements.txt

echo ""
echo "Installation terminée faire : source $VENV_NAME/bin/activate"