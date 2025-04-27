import os
import json
import shutil
from env import *  # Importa as variáveis do ambiente

# Definir caminhos das pastas originais
original_dirs = [
    os.path.join(DATA_DIR, LABELED, "train2017/train2017/"),
    os.path.join(DATA_DIR, LABELED, "val2017/val2017/"),
]

# Definir a nova pasta "all" para onde as imagens serão copiadas
new_dir = os.path.join(DATA_DIR, LABELED, "all/")

# Criar a nova pasta "all" se não existir
os.makedirs(new_dir, exist_ok=True)

# Listas para armazenar dados de anotações
all_images = []
all_annotations = []
categories = {}

# Função para encontrar a imagem na pasta correta
def encontrar_imagem(file_name):
    for dir_path in original_dirs:
        src_path = os.path.join(dir_path, file_name)
        if os.path.exists(src_path):
            return src_path  # Retorna o caminho correto da imagem
    return None  # Se não encontrar em nenhuma pasta, retorna None

# Função para mover imagens e salvar anotações
def move_images_and_save_annotations(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    for img_info in data["images"]:
        file_name = img_info["file_name"]

        # Procurar a imagem nas pastas originais
        src_path = encontrar_imagem(file_name)

        if src_path:
            # Copiar a imagem para a nova pasta "all"
            dst_path = os.path.join(new_dir, file_name)
            shutil.copy(src_path, dst_path)
            print(f"✅ Copiado: {file_name} → {new_dir}")
            all_images.append(img_info)  # Adiciona a imagem às imagens do novo dataset
        else:
            print(f"⚠️ Arquivo não encontrado: {file_name}")

    # Adicionar as anotações
    all_annotations.extend(data["annotations"])

    # Adicionar as categorias, apenas a primeira vez
    if not categories:
        categories.update({cat['id']: cat['name'] for cat in data["categories"]})

# Mover imagens de train.json e val.json
move_images_and_save_annotations(TRAIN_ANNOTATIONS_FILE)
move_images_and_save_annotations(VAL_ANNOTATIONS_FILE)

# Salvar o novo arquivo JSON com todas as anotações
new_json_data = {
    "images": all_images,
    "annotations": all_annotations,
    "categories": list(categories.values()),  # Lista de nomes de categorias
}

# Salvar o novo arquivo JSON
with open(os.path.join(DATA_DIR, LABELED, "all_annotations.json"), "w") as f:
    json.dump(new_json_data, f)
    print("✅ Novo arquivo JSON criado: all_annotations.json")

print("🚀 Transferência concluída!")
