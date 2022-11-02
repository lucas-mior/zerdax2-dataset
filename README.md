# Modelagem 3D de tabuleiros sintéticos de xadrez

## Objetivo
- Milhares de imagens de tabuleiros com peças e posições anotadas,
  com o objetivo de usar para treinamento de rede neural.

## Detalhes
- Utilizar o **blender**
- Modelar 10 tipos de tabuleiros diferentes, o que significa:
    - Diferentes cores de tabuleiro e peças
    - Diferentes peças (dimensões)
    - Borda e altura do tabuleiro variável
    - Isso pode ser feito de maneira manual se forem feitos até 10 tabuleiros
- Automatizar renderização de imagens diferentes, o que inclui
    - Peças presentes
    - Posição das peças
    - Posição e ângulo da câmera
    - Posição da lâmpada
    - Para isso, é preciso parametrizar todos esses valores,
      e é necessário saber a posição das casas para posicionar
      as peças dentro das casas corretas
    - Para cada imagem renderizada, gerar também um arquivo de anotação, contendo:
        - Bounding box de cada peça
        - ...

## Links úteis
1. [Documentação do blender](https://docs.blender.org://docs.blender.org/)
2. [Tutorial de python no blender](https://www.youtube.com/watch?v=XqX5wh4YeRw)
