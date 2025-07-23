<p align="center">

  <h1 align="center">Beyond Bare Queries: <br>
Open-Vocabulary Object Grounding <br> with 3D Scene Graph</h1>
  <p align="center">
    <a href="https://github.com/linukc">Linok Sergey</a>
    ·
    <a href="https://github.com/wingrune">Tatiana Zemskova</a>
    ·
    Svetlana Ladanova
    ·
    Roman Titkov
    ·
    Dmitry Yudin
    <br>
    Maxim Monastyrny
    ·
    Aleksei Valenkov
  </p>

  <h4 align="center"><a href="https://linukc.github.io/BeyondBareQueries/">Project</a> | <a href="http://arxiv.org/abs/2406.07113">arXiv</a> | <a href="https://github.com/linukc/BeyondBareQueries">Code</a></h4>
  <div align="center"></div>
</p>

Репозиторий для кода для демонстрации работы пайплайна BBQ на роботе с отправкой запроса пользователя через Телеграм-бот.

В репозитории представлен модифицированный код пайплайна BBQ, ссылка на оригинальный репозиторий: https://github.com/linukc/BeyondBareQueries.

0. Сборка докер-образа с BBQ (можно оригинальный с perception_models)

1. Запуск контейнера BBQ:

```
./docker/start.sh
```

2. Запуск сущностей (запуск из контейнера, зайти в контейнер - `./docker/into.sh`). Каждая сущность запускается в отдельном окне терминала:

a. Телеграм-бот:

```
./docker/into.sh
./telegram/bot.sh
```

b. Сервер с PLM моделью (нужен будет HGtoken):

```
./docker/into.sh
conda activate perception_models
cd ~/BeyondBareQueries/BeyondBareQueries/bbq_core/models
CUDA_VISIBLE_DEVICES=0 uvicorn plm_server:app --host 0.0.0.0 --port 31623
```

c. Сервер с BBQ пайплайном ответа на вопрос (нужен будет HGtoken):

```
./docker/into.sh
./BeyondBareQueries/deductive_server.sh
```

d. Сервер с демо (gradio):

```
./docker/into.sh
conda activate perception_models
cd ~/BeyondBareQueries/
python gradio.py
```

После чего либо воспользоваться автоматическим пробросом портов от VS Code, либо самому пробросить порты так, чтобы можно было визуализировать демо в браузере компьютера, подключенного к серверу.

3. Настройка ROS
```
pip install empy==3.3.4
pip install rospkg
pip install --upgrade numpy==1.26.4
```

Собрать сообщенияs:
```
./docker_ros/build.sh (optional)
./docker_ros/start.sh (optional)
./docker_ros/into.sh
cd ..
mkdir -p custom_msg/src
cd custom_msg/src
git clone https://github.com/cog-model/husky_demo_scripts/tree/main/husky_deom_transport
cd ..
catkin_make
source devel/setup.bash
```

Процесс с BBQ пайплайном построения карты. Нужно запускать в контейнере с ros1 (папка docker_ros):

```
./docker_ros/build.sh (optional)
./docker_ros/start.sh (optional)
./docker_ros/into.sh
# настройка ROS
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://172.88.0.241:11311
export ROS_IP=10.43.71.82
#
./BeyondBareQueries/map_cycle.sh
```
Для отладки: rostopic pub /chatter std_msgs/String "data: 'hello world'"

4. Остановить контейнер с демо:

```
./docker/stop.sh
```

## Citation
If you find this work helpful, please consider citing our work as:
```
@misc{linok2024barequeriesopenvocabularyobject,
      title={Beyond Bare Queries: Open-Vocabulary Object Grounding with 3D Scene Graph}, 
      author={Sergey Linok and Tatiana Zemskova and Svetlana Ladanova and Roman Titkov and Dmitry Yudin and Maxim Monastyrny and Aleksei Valenkov},
      year={2024},
      eprint={2406.07113},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.07113}, 
}
```