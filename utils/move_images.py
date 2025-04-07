import os
import json
import shutil
from env import *  # Importa as variáveis do ambiente

# Definir caminhos das pastas originais
original_dirs = [
    os.path.join(DATA_DIR, LABELED, "train2017/"),
    os.path.join(DATA_DIR, LABELED, "val2017/")
]

# Definir novas pastas onde as imagens serão movidas
new_dirs = {
    "train": os.path.join(DATA_DIR, LABELED, "train/"),
    "val": os.path.join(DATA_DIR, LABELED, "val/"),
    "test": os.path.join(DATA_DIR, LABELED, "test/")
}

# Criar as novas pastas se não existirem
for path in new_dirs.values():
    os.makedirs(path, exist_ok=True)

# Criar uma lista para armazenar imagens não encontradas
imagens_nao_encontradas = []

# Função para encontrar a imagem na pasta correta
def encontrar_imagem(file_name):
    for dir_path in original_dirs:
        src_path = os.path.join(dir_path, file_name)
        if os.path.exists(src_path):
            return src_path  # Retorna o caminho correto da imagem
    return None  # Se não encontrar em nenhuma pasta, retorna None

# Função para mover imagens
def move_images(json_file, target_folder):
    with open(json_file, "r") as f:
        data = json.load(f)

    for img_info in data["images"]:
        file_name = img_info["file_name"]
        
        # Procurar a imagem nas pastas originais
        src_path = encontrar_imagem(file_name)

        if src_path:
            dst_path = os.path.join(target_folder, file_name)
            shutil.move(src_path, dst_path)
            print(f"✅ Movido: {file_name} → {target_folder}")
        else:
            print(f"⚠️ Arquivo não encontrado: {file_name}")
            imagens_nao_encontradas.append(file_name)

# Mover imagens para as novas pastas
move_images(TRAIN_ANNOTATIONS_FILE, new_dirs["train"])
move_images(VAL_ANNOTATIONS_FILE, new_dirs["val"])
move_images(TEST_ANNOTATIONS_FILE, new_dirs["test"])

# Salvar lista de imagens não encontradas
if imagens_nao_encontradas:
    with open("imagens_nao_encontradas.txt", "w") as f:
        for img in imagens_nao_encontradas:
            f.write(img + "\n")
    print("⚠️ Algumas imagens não foram encontradas! Lista salva em 'imagens_nao_encontradas.txt'.")

print("🚀 Transferência concluída!")
