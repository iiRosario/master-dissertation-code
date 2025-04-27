#!/bin/bash

# Confirma se o diretório atual é um repositório git
if [ ! -d ".git" ]; then
  echo "Este não é um repositório git. Abortando."
  exit 1
fi

# Faz um reset para o último commit remoto e força o pull
echo "Desfazendo as mudanças locais e forçando o pull..."
git fetch --all
git reset --hard origin/$(git rev-parse --abbrev-ref HEAD)

# Confirma que o pull foi bem-sucedido
if [ $? -eq 0 ]; then
  echo "Pull forçado bem-sucedido!"
else
  echo "Falha ao fazer o pull."
fi
