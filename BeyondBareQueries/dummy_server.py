import os
import time

REMOTE_FILES = {
"RELEVANT_OBJECTS_3D": "/home/docker_user/BeyondBareQueries/outputs/3d_relevant_objects.gif",
"RELEVANT_OBJECTS_2D": "/home/docker_user/BeyondBareQueries/outputs/overlayed_masks_sam_and_graph_relevant_objects.png",
"RELEVANT_OBJECTS_TEXT": "/home/docker_user/BeyondBareQueries/outputs/relevant_objects.txt",
"RELEVANT_OBJECTS_TABLE": "/home/docker_user/BeyondBareQueries/outputs/table_relevant_objects.json",

"FINAL_ANSWER_3D": "/home/docker_user/BeyondBareQueries/outputs/3d_final_answer.gif",
"FINAL_ANSWER_2D": "/home/docker_user/BeyondBareQueries/outputs/overlayed_masks_sam_and_graph_final_answer.png",
"FINAL_ANSWER_TEXT": "/home/docker_user/BeyondBareQueries/outputs/final_answer.txt",
"FINAL_ANSWER_TABLE": "/home/docker_user/BeyondBareQueries/outputs/table_final_answer.json",
}
TEXT_FILE = "text.txt"


def main():
    print("update files ...")
    for key, path in REMOTE_FILES.items():
        time.sleep(0.8)
        os.utime(path, None)
        print(f"file {path} was updated")

def wait_for_message():
    """Ожидает изменения файла text.txt и вызывает main() при обновлении."""
    last_mod_time = None

    while True:
        if os.path.exists(TEXT_FILE):
            mod_time = os.path.getmtime(TEXT_FILE)
            if last_mod_time is None or mod_time > last_mod_time:
                last_mod_time = mod_time
                main()
        time.sleep(1)  # Проверка раз в секунду

if __name__ == "__main__":
    print("Ожидание сообщений от клиента...")
    wait_for_message()
