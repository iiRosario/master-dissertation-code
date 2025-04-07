import json
import matplotlib.pyplot as plt
from collections import Counter
import os
from env import *  # Ensure TRAIN_ANNOTATIONS_FILE, VAL_ANNOTATIONS_FILE, TEST_ANNOTATIONS_FILE are defined

# Function to load annotations from a JSON file
def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)

# Function to count the number of images and annotations per class
def count_images_and_annotations_by_class(annotations):
    categories = {category['id']: category['name'] for category in annotations['categories']}
    image_counts = Counter()
    annotation_counts = Counter()
    image_ids_per_class = {category_id: set() for category_id in categories}

    # Iterate over annotations to count images and annotations per category
    for annotation in annotations['annotations']:
        category_id = annotation['category_id']
        annotation_counts[category_id] += 1
        image_ids_per_class[category_id].add(annotation['image_id'])

    # Convert image ID sets into counts
    for category_id, image_ids in image_ids_per_class.items():
        image_counts[category_id] = len(image_ids)

    # Map category IDs to category names
    class_names = {category_id: categories[category_id] for category_id in image_counts}
    
    image_class_count = [(class_names[category_id], image_counts[category_id]) for category_id in image_counts]
    annotation_class_count = [(class_names[category_id], annotation_counts[category_id]) for category_id in annotation_counts]

    return image_class_count, annotation_class_count

# Function to plot and print data for a specific list of classes
def plot_specific_classes(coco_json_paths, selected_classes):
    total_image_counts = Counter()
    total_annotation_counts = Counter()
    category_names = {}

    # Process each annotation file
    for json_path in coco_json_paths:
        if os.path.exists(json_path):
            print(f"Processing {json_path}...")
            annotations = load_annotations(json_path)
            image_counts, annotation_counts = count_images_and_annotations_by_class(annotations)

            # Aggregate counts for selected classes only
            for category, count in image_counts:
                if category in selected_classes:
                    total_image_counts[category] += count

            for category, count in annotation_counts:
                if category in selected_classes:
                    total_annotation_counts[category] += count

            if not category_names:
                category_names = {cat['id']: cat['name'] for cat in annotations['categories']}
        else:
            print(f"⚠️ File {json_path} not found.")

    # Sort by count in descending order
    sorted_images = sorted(total_image_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_annotations = sorted(total_annotation_counts.items(), key=lambda x: x[1], reverse=True)

    # Prepare data for plotting
    image_class_names = [item[0] for item in sorted_images]
    image_counts = [item[1] for item in sorted_images]
    annotation_class_names = [item[0] for item in sorted_annotations]
    annotation_counts = [item[1] for item in sorted_annotations]

    # Print values
    print("\nNº of Images per Class:")
    for cls, count in sorted_images:
        print(f"{cls}: {count}")

    print("\nNº of Annotations per Class:")
    for cls, count in sorted_annotations:
        print(f"{cls}: {count}")

    # Generate a list of colors
    colors = [plt.cm.get_cmap('tab20c')(i / len(image_class_names)) for i in range(len(image_class_names))]

    # Plot: Images per Class
    plt.figure(figsize=(10, 6))
    plt.barh(image_class_names, image_counts, color=colors)
    plt.xlabel('Nº of Images')
    plt.ylabel('Classes')
    plt.title('Nº of Images for Selected Classes')
    plt.gca().invert_yaxis()
    plt.show()

    # Plot: Annotations per Class
    plt.figure(figsize=(10, 6))
    plt.barh(annotation_class_names, annotation_counts, color=colors)
    plt.xlabel('Nº of Annotations')
    plt.ylabel('Classes')
    plt.title('Nº of Annotations for Selected Classes')
    plt.gca().invert_yaxis()
    plt.show()

# Example usage
selected_classes = ["person", "car", "chair", "book", "bottle"]  # Replace with actual class names
coco_json_paths = [TRAIN_ANNOTATIONS_FILE, VAL_ANNOTATIONS_FILE, TEST_ANNOTATIONS_FILE]
plot_specific_classes(coco_json_paths, selected_classes)