import json
from PIL import Image, ImageDraw
import os
from env import * 

# Carregar as anotações do arquivo JSON
with open(NEW_ANNOTATIONS_FILE) as f:
    annotations = json.load(f)

# Função para obter o image_id a partir do nome da imagem
def get_image_id_from_filename(file_name):
    for image in annotations['images']:
        if image['file_name'] == file_name:
            return image['id']
    return None  # Caso não encontre

# Função para obter as anotações de uma imagem específica
def get_annotations_for_image(image_id):
    annotations_for_image = []
    for annotation in annotations['annotations']:
        if annotation['image_id'] == image_id:
            annotations_for_image.append(annotation)
    return annotations_for_image

# Função para desenhar as bounding boxes na imagem
def draw_bounding_boxes(file_name, annotations_for_image):
    # Obter o image_id a partir do nome do arquivo
    image_id = get_image_id_from_filename(file_name)
    
    if image_id is None:
        print(f"Imagem com o nome {file_name} não encontrada.")
        return

    # Carregar a imagem
    image_path = os.path.join(NEW_IMAGES_FILE, file_name)  # Ajuste o caminho e o formato da imagem conforme necessário
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Desenhar as bounding boxes
    for annotation in annotations_for_image:
        bbox = annotation['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        print(f"Desenhando bbox: {bbox} na imagem {image_id}")  # Verificação
        # Desenhar a caixa se as coordenadas estiverem dentro do limite da imagem
        if 0 <= x < image.width and 0 <= y < image.height:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

    # Mostrar a imagem com as bounding boxes
    image.show()

# Nome da imagem que você quer visualizar
file_name = '000000014736.jpg'  # Substitua pelo nome do arquivo da imagem que você deseja
image_id = get_image_id_from_filename(file_name)

if image_id:
    annotations_for_image = get_annotations_for_image(image_id)
    if annotations_for_image:
        draw_bounding_boxes(file_name, annotations_for_image)
    else:
        print(f"Nenhuma anotação encontrada para a imagem {file_name}")
else:
    print(f"Imagem {file_name} não encontrada no arquivo de anotações.")
