import json
import matplotlib.pyplot as plt
from collections import Counter
import os
from env import *  # TRAIN_ANNOTATIONS_FILE, VAL_ANNOTATIONS_FILE, TEST_ANNOTATIONS_FILE

# Function to load COCO annotations
def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)

# Function to count images and annotations per class
def count_images_and_annotations_by_class(annotations):
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    image_counts = Counter()
    annotation_counts = Counter()
    image_ids_per_class = {cat_id: set() for cat_id in categories}

    for annotation in annotations['annotations']:
        cat_id = annotation['category_id']
        annotation_counts[cat_id] += 1
        image_ids_per_class[cat_id].add(annotation['image_id'])

    for cat_id, image_ids in image_ids_per_class.items():
        image_counts[cat_id] = len(image_ids)

    return image_counts, annotation_counts, categories

# Function to generate the plots
def plot_top_20_classes(coco_json_paths):
    total_image_counts = Counter()
    total_annotation_counts = Counter()
    categories = {}

    for path in coco_json_paths:
        if os.path.exists(path):
            print(f"Processing {path}...")
            annotations = load_annotations(path)
            img_counts, ann_counts, cats = count_images_and_annotations_by_class(annotations)

            total_image_counts.update(img_counts)
            total_annotation_counts.update(ann_counts)

            if not categories:
                categories = cats
        else:
            print(f"⚠️ File {path} not found.")

    # Top 20 by images
    top_images = total_image_counts.most_common(20)
    top_annots = total_annotation_counts.most_common(20)

    # Prepare data for plotting
    img_class_names = [categories[cid] for cid, _ in top_images]
    img_counts = [count for _, count in top_images]

    annot_class_names = [categories[cid] for cid, _ in top_annots]
    annot_counts = [count for _, count in top_annots]

    # Colors
    img_colors = [plt.cm.get_cmap('tab20c')(i / 20) for i in range(len(img_class_names))]
    annot_colors = [plt.cm.get_cmap('tab20c')(i / 20) for i in range(len(annot_class_names))]

    # Plot: Top 20 Images
    plt.figure(figsize=(10, 6))
    plt.barh(img_class_names, img_counts, color=img_colors)
    plt.xlabel('Nº of Images')
    plt.ylabel('Classes')
    plt.title('Top 20 Classes by Nº of Images')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Plot: Top 20 Annotations
    plt.figure(figsize=(10, 6))
    plt.barh(annot_class_names, annot_counts, color=annot_colors)
    plt.xlabel('Nº of Annotations')
    plt.ylabel('Classes')
    plt.title('Top 20 Classes by Nº of Annotations')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Example usage
coco_json_paths = [TRAIN_ANNOTATIONS_FILE, VAL_ANNOTATIONS_FILE]
plot_top_20_classes(coco_json_paths)