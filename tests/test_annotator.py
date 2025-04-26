import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entities.Annotator import Annotator


def main():
    # Inicializa um anotador com ID 1, uma semente específica e sem valores de alphas fornecidos
    annotator1 = Annotator(id_=1, seed=1)
    
    # Exibe o estado inicial do anotador
    print("Annotator 1:")
    print(annotator1)
    print()

    # Inicializa um anotador com ID 1, uma semente específica e sem valores de alphas fornecidos
    annotator11 = Annotator(id_=2, seed=1)
    
    # Exibe o estado inicial do anotador
    print("Annotator 1.1:")
    print(annotator11)
    print()

    # Inicializa um anotador com ID 2, uma semente diferente e alphas específicos
    custom_alphas = [0.5, 1.2, 3.4, 2.8, 5.6, 4.3, 6.7, 1.1, 3.3, 2.1, 0.9]
    annotator2 = Annotator(id_=2, alphas=custom_alphas, seed=1)

    # Exibe o estado do segundo anotador
    print("Annotator 2:")
    print(annotator2)
    print()

    # Inicializa um anotador com especialização em algumas classes
    specialized_classes = [1, 3, 7]
    annotator3 = Annotator(id_=3, seed=7)
    annotator3.alphas = annotator3.init_alphas(specialized_classes=specialized_classes, factor=2.5)

    # Exibe o estado do terceiro anotador
    print("Annotator 3 (with specialized classes):")
    print(annotator3)





if __name__ == "__main__":
    main()
