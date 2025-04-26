#!/bin/bash

# Verifica se há alterações no repositório
if ! git diff-index --quiet HEAD --; then
    echo "Há alterações no repositório. Preparando para commit..."

    # Adiciona todas as alterações ao commit
    git add .

    # Solicita a mensagem de commit
    echo "Digite a mensagem de commit:"
    read commit_message

    # Realiza o commit com a mensagem fornecida
    git commit -m "$commit_message"

    # Faz o push para o repositório remoto
    echo "Realizando o push para o repositório remoto..."
    git push

    # Confirma se o push foi bem-sucedido
    if [ $? -eq 0 ]; then
        echo "Commit e push realizados com sucesso!"
    else
        echo "Falha ao realizar o push."
    fi
else
    echo "Nenhuma alteração detectada. Nenhum commit necessário."
fi
