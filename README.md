# TCC-RL

TCC - Reinforcement Learning

 - Trabalho em desenvolvimento..

Engenharia de Computaçao - PUC MINAS

Autor: Rodrigo Fonseca Marques

Orientador: Zenilton Kleber

-----

#### Conteúdo do documento:

- [Configuração](#configuração)
    - [Sistema Operacional](#sistema-operacional)
    - [Hardware](#hardware)
    - [Instalação](#instalação)   
- [Treinamento](#treinamento)
- [Play](#play)


-----
## CONFIGURAÇÃO


### Sistema operacional
O sistema operacional usado foi o Ubuntu 16.04. 

### Hardware
O hardware utilizado para a prática da monografia é composto por um processador
Intel Core i7-6700K, com 32 GB de memória RAM e uma placa de vídeo GTX 1070 com
8GB de GPU.

### Instalação

Inicialmente foi feita a instalação do S.O. e configurado o driver da placa de vídeo.
Após, foi atualizado os pacotes:

`Apt-get update` e `Apt-get upgrade`

Logo após, é recomendado fazer um 

`reboot`

Então é necessário fazer as instalações abaixo:

`apt-get install python3-pip`

`apt-get install python-pip`

`apt-get install cmake`

Após a instalação do pip e cmake, para que seja usado a placa de vídeo corretamente para o treinamento, é necessário fazer a instalação do CUDA 9.0 com cuDNN 7.4. Seguir o tutorial nesse [LINK](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e)

É necessário fazer a instalação de algumas bibliotecas, que podem ser executadas através do arquivo 'requirements.txt'

A instalação do GYM é necessária e feita através do passo a passo:
```
git clone https://github.com/openai/gym

cd gym 

pip3 install -e .

apt-get install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg

pip3 install 'gym[all]'
```
-----
## TREINAMENTO

É possivel fazer alterações nas variáveis que estão descritas no início do código.
Para rodar o código de treinamento é necessário executar o seguinte comando:

`python3 breakout_train_dqn.py`

O treinamento é feito e o modelo é salvo na pasta 'saved_model'. O salvamento do modelo é feito de 1000 em 1000 iterações.

-----
## PLAY

Para fazer com que o modelo treinado seja capaz de jogar, é necessário executar o seguinte comando:
É necessário mudar dentro do código a pasta com o nome dado ao modelo.

`python3 breakout_play.py`

-----



- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request

:star:
