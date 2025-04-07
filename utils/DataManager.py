import json
import os
from env import *

# Carregar anotações COCO
def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data



# Extrair imagens e anotações do dataset COCO
def extract_coco_data(coco_data, image_dir):
    images = []
    labels = []
    image_id_to_filename = {img['id']: os.path.join(image_dir, img['file_name']) for img in coco_data['images']}
    
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id in image_id_to_filename:
            images.append(image_id_to_filename[image_id])
            labels.append(annotation)  # Guardamos a anotação completa
    
    return images, labels


def count_images(coco_data): return len(coco_data.getImgIds())
def count_annotations(coco_data): return len(coco_data.getAnnIds())

    
def list_classes(coco_data, limit):
    categories = coco_data.loadCats(coco_data.getCatIds())
    classes = [cat['name'] for cat in categories]
    return classes[:limit] if limit > 0 else classes
    


def count_images_annotations(json_path):
    """
    Conta o número de imagens e o número total de anotações em um arquivo JSON do COCO.
    
    :param json_path: Caminho para o arquivo JSON de anotações do COCO
    :return: Tupla (num_imagens, num_anotacoes)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_imagens = len(data.get("images", []))
        num_anotacoes = len(data.get("annotations", []))
        
        return num_imagens, num_anotacoes
    
    except Exception as e:
        print(f"Erro ao processar o arquivo JSON: {e}")
        return None, None
    

def count_images_annotations_by_class(json_path, class_id):
    """
    Conta o número de imagens e o número total de anotações para uma classe específica em um arquivo JSON do COCO.

    :param json_path: Caminho para o arquivo JSON de anotações do COCO
    :param class_id: ID da classe que queremos filtrar
    :return: Tupla (num_imagens, num_anotacoes)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        imagens_com_classe = set()
        num_anotacoes = 0
        
        for annotation in data.get("annotations", []):
            if annotation.get("category_id") == class_id:
                imagens_com_classe.add(annotation.get("image_id"))
                num_anotacoes += 1
        
        num_imagens = len(imagens_com_classe)
        
        return num_imagens, num_anotacoes

    except Exception as e:
        print(f"Erro ao processar o arquivo JSON: {e}")
        return None, None
    

def get_classes(json_path):
    """
    Retorna uma lista com todas as classes presentes no dataset COCO.
    
    :param json_path: Caminho para o arquivo JSON de anotações do COCO
    :return: Lista de classes presentes no dataset
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        classes = {category["id"]: category["name"] for category in data.get("categories", [])}
        
        return list(classes.values())
    
    except Exception as e:
        print(f"Erro ao processar o arquivo JSON: {e}")
        return []



