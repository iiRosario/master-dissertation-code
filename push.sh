#!/bin/bash

# Mensagem de commit passada como argumento
if [ -z "$1" ]; then
  echo "Uso: ./git_push.sh \"Mensagem do commit\""
  exit 1
fi

# Adiciona todas as alterações
git add .

# Faz o commit com a mensagem fornecida
git commit -m "$1"

# Faz o push para o repositório
git push
