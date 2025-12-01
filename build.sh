#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Zbiera pliki statyczne (CSS admina) do jednego folderu
python manage.py collectstatic --no-input

# Tworzy tabele w bazie danych (Postgres)
python manage.py migrate