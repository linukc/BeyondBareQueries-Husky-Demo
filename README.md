0. Сборка докер-образа с BBQ:

```
cd BeyondBareQueries
bash docker/build.sh
```

1. Запуск контейнера BBQ:

```
bash start.sh /datasets/
```

2. Запуск сущностей, живущих на компьютере с пайплайном (запуск из контейнера, зайти в контейнер - `bash ./into.sh`). Каждая сущность запускается в отдельном окне терминала:

a. Телеграм-бот:

```
bash ./into.sh
bash bot.sh
```

b. Сервер с PLM моделью:

```
bash ./into.sh
conda activate perception_models
cd ~/BeyondBareQueries/BeyondBareQueries/bbq/models
CUDA_VISIBLE_DEVICES=1 uvicorn plm_server:app --host 0.0.0.0 --port 31623
```

c. Сервер с BBQ пайплайном:

```
bash ./into.sh
cd ~/BeyondBareQueries/BeyondBareQueries
bash server.sh
```


d. Сервер с демо:

```
bash ./into.sh
conda activate perception_models
cd ~/BeyondBareQueries/BeyondBareQueries
python demo_local.py
```

После чего либо воспользоваться автоматическим пробросом портов от VS Code, либо самому пробросить порты так, чтобы можно было визуализировать демо в браузере компьютера, подключенного к серверу.

3. Запуск сервера, отдающего изображения:

a. Локальный сервер (по запросу отдаёт заранее сохраненные изображения):

```
bash ./into.sh
python local_camera_server.py
```


b. Сервер с камерами на роботе:

```
aima em load-env
cd ~/bbq_demo/
python3 camera_server.py
```

4. Остановить контейнер с демо:

```
bash ./stop.sh
```