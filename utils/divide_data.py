import json
import random
from pycocotools.coco import COCO
from env import * 

# Caminhos para os arquivos COCO originais
train_json_path = TRAIN_ANNOTATIONS_FILE
val_json_path = VAL_ANNOTATIONS_FILE

# Carregar os dados COCO
train_coco = COCO(train_json_path)
val_coco = COCO(val_json_path)

# Combinar imagens e anotações dos dois conjuntos
all_images = train_coco.dataset["images"] + val_coco.dataset["images"]
all_annotations = train_coco.dataset["annotations"] + val_coco.dataset["annotations"]

# Embaralhar os dados para randomizar a distribuição
random.shuffle(all_images)

# Definir os tamanhos dos subconjuntos
total_images = len(all_images)
train_size = int(0.7 * total_images)
test_size = int(0.2 * total_images)

# Criar divisões
train_images = all_images[:train_size]
test_images = all_images[train_size:train_size + test_size]
val_images = all_images[train_size + test_size:]

# Função para filtrar anotações com base nas imagens selecionadas
def filter_annotations(images_subset, annotations):
    image_ids = {img["id"] for img in images_subset}
    return [ann for ann in annotations if ann["image_id"] in image_ids]

# Criar novos datasets COCO
train_data = {
    "info": train_coco.dataset["info"],
    "licenses": train_coco.dataset["licenses"],
    "categories": train_coco.dataset["categories"],
    "images": train_images,
    "annotations": filter_annotations(train_images, all_annotations)
}

test_data = {
    "info": train_coco.dataset["info"],
    "licenses": train_coco.dataset["licenses"],
    "categories": train_coco.dataset["categories"],
    "images": test_images,
    "annotations": filter_annotations(test_images, all_annotations)
}

val_data = {
    "info": train_coco.dataset["info"],
    "licenses": train_coco.dataset["licenses"],
    "categories": train_coco.dataset["categories"],
    "images": val_images,
    "annotations": filter_annotations(val_images, all_annotations)
}

# Salvar os novos arquivos JSON
with open("coco_train.json", "w") as f:
    json.dump(train_data, f)

with open("coco_test.json", "w") as f:
    json.dump(test_data, f)

with open("coco_val.json", "w") as f:
    json.dump(val_data, f)

print("✅ Divisão concluída! Novos arquivos COCO gerados.")
